"""
siglip_rlt_embeddings.py
========================

Run-Length Tokenization (RLT) at the SigLIP embedding seam.

This module applies RLT *before* SigLIP's transformer blocks, so the expensive
attention runs over a reduced token set. It reuses SigLIP's own learned weights
(patch-embed, spatial position embedding, and the per-layer q/k/v/out_proj) and
only swaps the attention *kernel* for xformers' block-diagonal
memory_efficient_attention -- the same kernel the RLT reference uses
(rlt/src/models/vit_helpers.py:Attention).

Pipeline (one clip of T frames):

    pixel_values (T,3,H,W)
        |
        |  SigLIP embeddings  -> (T, P, C)   spatial pos already baked in
        |  RLT keep-mask      -> drop temporally-redundant patch tokens
        v
    gather survivors (N, C)               # N <= T*P, in (t,h,w) order
        + temporal position[frame_t]*s     # fixed sinusoid, scaled to SigLIP pos RMS
        |
        |  pack -> (1, N, C),  BlockDiagonalMask.from_seqlens([N])
        v
    SigLIP encoder blocks (xformers attn over the N survivors) -> (N, C)

What it deliberately ADDS on top of SigLIP (because SigLIP is per-image, RLT is
per-clip):
  * a FIXED temporal position table (sinusoidal over frames). SigLIP already bakes
    in SPATIAL position, so RLT adds only the temporal component -- scaled to the
    magnitude of SigLIP's own position_embedding, so it enters exactly as SigLIP's
    spatial position does (a raw unit sinusoid would vanish against the feature
    magnitude; the whole-feature RMS would be too large). This is the ONLY thing RLT
    adds at the seam: the paper reports the learnable run-length encoding
    contributes little, so it is not used here.
  * cross-frame attention over the packed survivors (RLT-faithful: one block per
    clip via BlockDiagonalMask).

NOTE (downstream): the output is a *ragged* (N, C) token set, not the dense
(T, P, C) grid the existing Video-XL-Pro pipeline (interpolate -> add_video ->
SAE -> 2D pool) expects. Wiring this into llava_arch is a separate step.
"""

from typing import Tuple

import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

try:
    from .rlt_static_tokens import (
        batched_find_idxs_to_keep,
        get_sinusoid_encoding,
    )
except ImportError:  # allow running this file directly as a script for testing
    from rlt_static_tokens import (
        batched_find_idxs_to_keep,
        get_sinusoid_encoding,
    )


class SiglipRLTEmbeddings(nn.Module):
    """RLT wrapper around a HuggingFace SiglipVisionModel.

    Args:
        vision_model: a transformers SiglipVisionModel (its .vision_model holds
            embeddings / encoder.layers / post_layernorm).
        threshold: RLT drop threshold (mean abs pixel change). Tune for the
            normalized SigLIP input range.
        patch_size: SigLIP patch size (14 for siglip-so400m-patch14-384).
        max_frames: size of the temporal position table.
        temporal_pos_scale: multiplier on the temporal-position magnitude AFTER it
            is rescaled to the RMS of SigLIP's own position_embedding. 1.0 => temporal
            position enters at the same magnitude SigLIP uses for spatial position;
            <1.0 => a gentler complement; 0 => disabled. This is the only RLT-specific
            knob; RLT adds no learnable state at the seam.
    """

    def __init__(
        self,
        vision_model,
        threshold: float = 0.1,
        patch_size: int = 14,
        max_frames: int = 512,
        temporal_pos_scale: float = 1.0,
        apply_post_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        vm = vision_model.vision_model
        self.embeddings = vm.embeddings
        self.encoder_layers = vm.encoder.layers
        self.post_layernorm = vm.post_layernorm

        self.embed_dim = self.embeddings.embed_dim
        self.threshold = threshold
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.temporal_pos_scale = temporal_pos_scale
        # Video-XL-Pro consumes SigLIP's hidden_states[-1] (the last encoder layer
        # output, BEFORE post_layernorm). Default off to match that; the projector
        # was trained on pre-LN features.
        self.apply_post_layernorm = apply_post_layernorm

        # Temporal position table phi_t (sinusoidal, frozen). SigLIP is per-image and
        # has no temporal axis; this restores "which frame" each survivor came from.
        # Non-persistent buffer: recomputed at load, moves with .to(device/dtype).
        self.register_buffer(
            "temporal_pos",
            get_sinusoid_encoding(max_frames, self.embed_dim)[0],          # (max_frames, C)
            persistent=False,
        )

        # NOTE: the paper's learnable run-length encoding phi_L is deliberately NOT
        # used -- the paper reports it contributes little, and RLT here adds only the
        # scale-matched fixed temporal position above. No trainable state at the seam.

    def _xformers_attention(self, self_attn, x: torch.Tensor, attn_bias) -> torch.Tensor:
        """SigLIP attention via xformers block-diagonal memory_efficient_attention.

        Reuses self_attn's learned q/k/v/out_proj; x is (1, N, C). Mirrors
        rlt/src/models/vit_helpers.py:Attention.forward (use_flash_attn branch).
        """
        B, N, C = x.shape
        H, d = self_attn.num_heads, self_attn.head_dim
        q = self_attn.q_proj(x).view(B, N, H, d)
        k = self_attn.k_proj(x).view(B, N, H, d)
        v = self_attn.v_proj(x).view(B, N, H, d)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self_attn.scale)
        out = out.reshape(B, N, C)
        return self_attn.out_proj(out)

    def _run_encoder(self, x: torch.Tensor, attn_bias) -> torch.Tensor:
        """Run SigLIP encoder layers with the xformers attention swap.

        Replicates SiglipEncoderLayer.forward exactly, only changing the kernel.
        """
        for layer in self.encoder_layers:
            residual = x
            x = layer.layer_norm1(x)
            x = self._xformers_attention(layer.self_attn, x, attn_bias)
            x = residual + x

            residual = x
            x = layer.layer_norm2(x)
            x = layer.mlp(x)
            x = residual + x
        if self.apply_post_layernorm:
            x = self.post_layernorm(x)
        return x

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: deliberately NOT wrapped in @torch.no_grad(), so gradients can reach
        downstream modules during fine-tuning. The added temporal position is a FIXED
        sinusoid rescaled by a DETACHED per-clip factor, so RLT introduces no trainable
        state of its own at the seam. What actually updates is controlled by
        requires_grad (the SigLIP backbone stays frozen). At eval, the caller's
        inference_mode/no_grad keeps this cheap.

        Args:
            pixel_values: (T, 3, H, W) frames for ONE clip.

        Returns:
            survivors: (N, C) encoded surviving tokens.
            token_mask: (T*P,) bool keep-mask in (t,h,w) order.
        """
        T = pixel_values.shape[0]
        dev, dtype = pixel_values.device, pixel_values.dtype

        # 1. RLT keep-mask from raw pixels. tubelet_size=1 -> consecutive frames.
        x5d = pixel_values.permute(1, 0, 2, 3).unsqueeze(0)            # (1, C, T, H, W)
        token_mask = batched_find_idxs_to_keep(
            x5d, threshold=self.threshold, tubelet_size=1, patch_size=self.patch_size
        )                                                              # (1, T*h*w)
        n_per_frame = token_mask.shape[1] // T
        h = w = int(round(n_per_frame ** 0.5))

        # 2. SigLIP patch-embed (+ spatial pos baked in). (T, P, C)
        emb = self.embeddings(pixel_values)
        P = emb.shape[1]
        assert P == n_per_frame, (
            f"SigLIP grid ({P}) != RLT keep-mask grid ({n_per_frame}); "
            f"check patch_size / image_size alignment."
        )

        # 3. Gather survivors in (t,h,w) order.
        mask_flat = token_mask.reshape(-1)                            # (T*P,)
        survivors = emb.reshape(T * P, self.embed_dim)[mask_flat]     # (N, C)

        # 4. Add a FIXED temporal position phi_t (which frame each survivor came from).
        #    SigLIP already provides SPATIAL position (baked into `emb`), so we add only
        #    the temporal component the paper's spatiotemporal encoding contributes on
        #    top. It is scaled to the magnitude SigLIP itself uses for position -- the
        #    RMS of SigLIP's own position_embedding.weight -- so temporal position
        #    enters exactly as SigLIP's spatial position does (faithful in spirit to the
        #    RLT paper, which adds position at the backbone's own position scale), rather
        #    than as a raw unit sinusoid (would vanish) or the whole-feature RMS (too
        #    large). The scale is DETACHED (it only rescales the frozen sinusoid).
        frame_idx = torch.arange(T, device=dev).view(T, 1).expand(T, P).reshape(-1)[mask_flat]
        pos = self.temporal_pos.to(device=dev, dtype=dtype)[frame_idx]   # (N, C)
        if self.temporal_pos_scale > 0 and survivors.numel() > 0:
            # Read SigLIP's spatial position encoding via the module forward (as HF
            # does) rather than .weight directly, so under DeepSpeed Zero-3 the sharded
            # weight is gathered by the nn.Embedding forward hook.
            sp = self.embeddings.position_embedding(self.embeddings.position_ids)  # (1, P, C)
            ref_rms = sp.detach().float().pow(2).mean().sqrt()
            pos_rms = pos.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            survivors = survivors + pos * (self.temporal_pos_scale * ref_rms / pos_rms).to(dtype)

        # 5. Pack and encode with PER-FRAME attention. SigLIP is a 2D image encoder
        #    (weights trained with WITHIN-frame attention), and Video-XL-Pro already
        #    defers whole-video attention to the LLM. So each frame's survivors
        #    attend only within that frame (per-frame block-diagonal = 2D /
        #    SigLIP-native). Survivors are in frame-major (t,h,w) order, so per-frame
        #    seqlens partition the packed sequence directly.
        seqlens = mask_flat.reshape(T, P).sum(dim=1).tolist()
        seqlens = [int(s) for s in seqlens if s > 0]     # skip fully-dropped frames
        attn_bias = BlockDiagonalMask.from_seqlens(seqlens)
        x = survivors.unsqueeze(0)                                    # (1, N, C)
        x = self._run_encoder(x, attn_bias).squeeze(0)               # (N, C)

        return x, mask_flat


if __name__ == "__main__":
    # ---- standalone integration test (random-weight tiny SigLIP, GPU) -------
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    assert torch.cuda.is_available(), "xformers memory_efficient_attention needs CUDA"
    dev = "cuda"
    dtype = torch.float16

    patch = 14
    cfg = SiglipVisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_channels=3, image_size=28, patch_size=patch,
    )
    vm = SiglipVisionModel(cfg).to(dev, dtype).eval()
    P = (28 // patch) ** 2
    h = w = 28 // patch
    print(f"tiny SigLIP: grid {h}x{w}={P}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

    rlt = SiglipRLTEmbeddings(vm, threshold=0.1, patch_size=patch, max_frames=64).to(dev, dtype).eval()

    # controlled clip: frame0 random; 1-2 redundant; top-left patch jumps at t=3; 4-5 redundant
    T = 6
    base = torch.randn(1, 3, 28, 28, device=dev, dtype=dtype)
    frames = base.repeat(T, 1, 1, 1)
    frames[3:, :, :patch, :patch] += 5.0

    survivors, mask_flat = rlt(frames)
    N = int(mask_flat.sum())
    print(f"survivors: {tuple(survivors.shape)}  (N={N}, dense would be {T*P})")

    assert survivors.shape == (N, cfg.hidden_size)
    assert torch.isfinite(survivors.float()).all(), "non-finite outputs from xformers attention"
    print("\nOK: SigLIP-RLT embeddings run end-to-end through xformers attention.")
