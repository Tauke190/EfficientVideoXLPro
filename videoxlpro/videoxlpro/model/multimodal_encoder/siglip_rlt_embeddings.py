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
    gather survivors (N, C)            # N <= T*P, in (t,h,w) order
        + temporal position[frame_t]   # NEW: SigLIP has no temporal axis
        + length embedding[run_len-1]  # NEW: how many frames the token spans
        |
        |  pack -> (1, N, C),  BlockDiagonalMask.from_seqlens([N])
        v
    SigLIP encoder blocks (xformers attn over the N survivors) -> (N, C)

What it deliberately ADDS on top of SigLIP (because SigLIP is per-image, RLT is
per-clip):
  * a temporal position table (sinusoidal over frames),
  * a LEARNABLE run-length encoding phi_L (paper sec 3.2), trained during
    fine-tuning; owned by the base model so it reaches the optimizer/checkpoint,
  * cross-frame attention over the packed survivors (RLT-faithful: one block per
    clip via BlockDiagonalMask).

NOTE (downstream): the output is a *ragged* (N, C) token set, not the dense
(T, P, C) grid the existing Video-XL-Pro pipeline (interpolate -> add_video ->
SAE -> 2D pool) expects. Wiring this into llava_arch is a separate step.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

try:
    from .rlt_static_tokens import (
        batched_find_idxs_to_keep,
        batched_get_token_lengths,
        get_sinusoid_encoding,
    )
except ImportError:  # allow running this file directly as a script for testing
    from rlt_static_tokens import (
        batched_find_idxs_to_keep,
        batched_get_token_lengths,
        get_sinusoid_encoding,
    )


def get_token_lengths_aligned(token_mask: torch.Tensor, T: int, h: int, w: int) -> torch.Tensor:
    """Run-length per surviving token, in (t, h, w) order to match the gather.

    Same run-length *values* as rlt_static_tokens.batched_get_token_lengths, but
    returned in (t, h, w) order so they line up with survivors gathered by
    `tokens.flatten[token_mask]`. (The reference returns them in (h, w, t) order;
    see the ordering caveat in rlt_static_tokens.py.)

    Built as a dense length-map then gathered with the same mask, so alignment is
    guaranteed by construction.
    """
    dev = token_mask.device
    m = token_mask.reshape(1, T, h, w)
    t_idx = torch.arange(T, device=dev).view(1, T, 1, 1).expand(1, T, h, w)
    big = T

    kept_idx = torch.where(m, t_idx, torch.full_like(t_idx, big))
    # nearest kept frame index at >= t (reverse cumulative min along time)
    next_incl = torch.flip(torch.cummin(torch.flip(kept_idx, dims=[1]), dim=1).values, dims=[1])
    # nearest kept frame strictly after t; pad tail with `big` to close last run
    tail = torch.full((1, 1, h, w), big, device=dev, dtype=next_incl.dtype)
    next_after = torch.cat([next_incl[:, 1:], tail], dim=1)

    length_map = torch.where(m, (next_after - t_idx).clamp(min=1), torch.zeros_like(t_idx))
    return length_map.flatten()[token_mask.flatten()]   # (N,), (t,h,w) order


class SiglipRLTEmbeddings(nn.Module):
    """RLT wrapper around a HuggingFace SiglipVisionModel.

    Args:
        vision_model: a transformers SiglipVisionModel (its .vision_model holds
            embeddings / encoder.layers / post_layernorm).
        threshold: RLT drop threshold (mean abs pixel change). Tune for the
            normalized SigLIP input range.
        patch_size: SigLIP patch size (14 for siglip-so400m-patch14-384).
        max_frames: size of the temporal position table.
        use_length_embed: add the run-length embedding (RLT's length encoding).
        length_embed: optional model-owned learnable run-length encoding phi_L,
            passed as an nn.Embedding (paper sec 3.2). When given, this wrapper
            references that module (so it stays in the base model's
            named_parameters() -> optimizer + checkpoint, and DeepSpeed Zero-3
            gathers it via the module's forward hook); when None, it creates its own
            zero-init embedding for standalone / eval use.
    """

    def __init__(
        self,
        vision_model,
        threshold: float = 0.1,
        patch_size: int = 14,
        max_frames: int = 512,
        use_length_embed: bool = True,
        apply_post_layernorm: bool = False,
        length_embed: Optional[nn.Module] = None,
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
        self.use_length_embed = use_length_embed
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

        # Run-length encoding phi_L (paper "Don't Look Twice", sec 3.2): a LEARNABLE
        # per-length bias the model trains during fine-tuning to compensate for the
        # info removed by temporal pruning. (The original frozen sinusoid disabled
        # exactly the mechanism the paper relies on.) Implemented as an nn.Embedding
        # so it is consumed via a module forward -- DeepSpeed Zero-3 gathers and
        # reduce-scatters its weight through the standard forward hook.
        #   * length_embed passed in -> reference the model-owned module. Stash it in
        #     __dict__ so this (un-registered) wrapper's .to()/.cuda() can't clone it
        #     and silently break sharing; gradients still flow to it.
        #   * length_embed is None   -> own zero-init embedding (standalone/eval). Zero
        #     init => no-op at the seam initially, so enabling RLT does not shock the
        #     pretrained features; fine-tuning learns the bias from zero.
        if length_embed is not None:
            self.__dict__["length_embed"] = length_embed
        else:
            self.length_embed = nn.Embedding(max_frames, self.embed_dim)
            nn.init.zeros_(self.length_embed.weight)

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

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        NOTE: deliberately NOT wrapped in @torch.no_grad(). The paper propagates
        gradients to the learnable run-length encoding phi_L during fine-tuning, so
        the encoder graph must be retained when training. What actually updates is
        controlled by requires_grad (the SigLIP backbone stays frozen; only phi_L
        and downstream modules train). At eval, the caller's inference_mode/no_grad
        keeps this cheap.

        Args:
            pixel_values: (T, 3, H, W) frames for ONE clip.

        Returns:
            survivors: (N, C) encoded surviving tokens.
            token_mask: (T*P,) bool keep-mask in (t,h,w) order.
            lengths: (N,) run-length per surviving token (t,h,w order).
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

        # 4. Add temporal position phi_t: per-survivor frame index. The buffer is
        #    placed on-device here (no blanket module .to() at build time), then
        #    indexed -- index and table must share a device.
        frame_idx = torch.arange(T, device=dev).view(T, 1).expand(T, P).reshape(-1)[mask_flat]
        temporal_pos = self.temporal_pos.to(device=dev, dtype=dtype)
        survivors = survivors + temporal_pos[frame_idx]

        # 5. Add learnable run-length encoding phi_L: run-length per survivor
        #    (aligned order). length_embed is the model-owned nn.Embedding; calling
        #    it (vs indexing) lets Zero-3 gather its weight, and the cast keeps
        #    autograd flowing back to it.
        lengths = get_token_lengths_aligned(mask_flat, T, h, w)       # (N,), in [1,T]
        if self.use_length_embed:
            survivors = survivors + self.length_embed(lengths - 1).to(device=dev, dtype=dtype)

        # 6. Pack and encode with PER-FRAME attention. SigLIP is a 2D image encoder
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

        return x, mask_flat, lengths


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

    survivors, mask_flat, lengths = rlt(frames)
    N = int(mask_flat.sum())
    print(f"survivors: {tuple(survivors.shape)}  (N={N}, dense would be {T*P})")
    print(f"lengths (t,h,w order): {lengths.tolist()}")
    print(f"sum lengths: {int(lengths.sum())} (== T*P = {T*P})")

    # cross-check: aligned length multiset == faithful reference multiset
    ref = batched_get_token_lengths(mask_flat.unsqueeze(0).cpu(), batch_size=1, input_shape=(T, h, w))
    assert sorted(lengths.cpu().tolist()) == sorted(ref.tolist()), "length multiset must match reference"

    assert survivors.shape == (N, cfg.hidden_size)
    assert int(lengths.sum()) == T * P
    assert torch.isfinite(survivors.float()).all(), "non-finite outputs from xformers attention"
    print("\nOK: SigLIP-RLT embeddings run end-to-end through xformers attention.")
