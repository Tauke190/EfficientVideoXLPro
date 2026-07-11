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
        batched_find_idxs_to_keep_ref,
        get_sinusoid_encoding,
    )
except ImportError:  # allow running this file directly as a script for testing
    from rlt_static_tokens import (
        batched_find_idxs_to_keep,
        batched_find_idxs_to_keep_ref,
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
        attn_mode: str = "reuse",
        mask_mode: str = "ref",
        refresh_every: int = 0,
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
        assert attn_mode in ("reuse", "per_frame"), f"unknown attn_mode {attn_mode!r}"
        self.attn_mode = attn_mode
        assert mask_mode in ("ref", "consec"), f"unknown mask_mode {mask_mode!r}"
        self.mask_mode = mask_mode
        self.refresh_every = refresh_every
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

    @staticmethod
    def _carry_idx(mask2d: torch.Tensor) -> torch.Tensor:
        """(T, P) index of the most recent frame that KEPT each spatial position.

        A dropped token is by definition unchanged since its last surviving copy, so
        this is the frame whose state it carries. Position (t, p) that survives maps to
        itself. RLT always keeps all of frame 0, so every position has a valid source.
        """
        T, P = mask2d.shape
        t_idx = torch.arange(T, device=mask2d.device).view(T, 1).expand(T, P)
        kept_t = torch.where(mask2d, t_idx, torch.full_like(t_idx, -1))
        return torch.cummax(kept_t, dim=0).values.clamp(min=0)

    def _reuse_index(self, mask2d: torch.Tensor, T: int, P: int):
        """Precompute the gather map from the packed survivor array to the dense grid.

        src_row[t*P + p] = row in the (N, ·) survivor array holding the token that slot
        (t, p) carries -- itself if it survived, else its last surviving copy. Built ONCE
        per clip so the per-layer inner loop is a single index_select with no boolean
        masking (torch bool-indexing calls nonzero() every time, which is slow and syncs).
        """
        dev = mask2d.device
        mask_flat = mask2d.reshape(-1)
        surv_idx = mask_flat.nonzero(as_tuple=True)[0]                     # (N,)

        carry = self._carry_idx(mask2d)                                    # (T,P) source frame
        flat_src = carry * P + torch.arange(P, device=dev).view(1, P)      # (T,P) flat (t,p)

        rank = torch.zeros(T * P, dtype=torch.long, device=dev)
        rank[surv_idx] = torch.arange(surv_idx.numel(), device=dev)
        src_row = rank[flat_src.reshape(-1)]                               # (T*P,) -> survivor row
        return surv_idx, src_row

    def _run_encoder_reuse(self, emb, mask2d, T, P):
        """SigLIP encoder with RLT token reuse (fixes survivor context starvation).

        Per frame, attention sees a FULL P-token key/value set: survivors contribute
        freshly-computed k/v, and dropped tokens contribute the k/v they had at their last
        surviving frame. That is exactly the state the dense model would compute for them,
        since RLT dropped them precisely because their content did not change -- and
        layer_norm / k_proj / v_proj are position-wise, so carrying the hidden state
        forward and carrying k/v forward are equivalent.

        The running state stays in packed survivor form (N, C), so EVERY linear op
        (q/k/v/out projections and the MLP) costs O(N), not O(T*P) -- the keep-rate saving
        is preserved in full. The only dense-sized work is the two gathers that build the
        attention key/value set.

        Attention stays strictly WITHIN a frame, matching the per-frame SigLIP that the
        rest of Video-XL-Pro was trained against; no cross-frame attention is introduced,
        so no temporal position embedding is required.

        At 100% keep this reduces to dense SigLIP exactly -- see the __main__ self-test.
        """
        C = self.embed_dim
        surv_idx, src_row = self._reuse_index(mask2d, T, P)

        # Frames with zero survivors contribute no queries; drop their (empty) block so
        # xformers never sees a zero-length sequence.
        active = mask2d.any(dim=1)                                        # (T,)
        n_act = int(active.sum())
        kv_rows = src_row.view(T, P)[active].reshape(-1)                  # (T_act*P,)
        attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen=[int(s) for s in mask2d.sum(dim=1)[active].tolist()],
            kv_seqlen=[P] * n_act,
        )

        x = emb.reshape(T * P, C).index_select(0, surv_idx)                # (N, C)
        N = x.shape[0]

        for layer in self.encoder_layers:
            a = layer.self_attn
            H, d = a.num_heads, a.head_dim

            h = layer.layer_norm1(x)                                       # (N, C)
            q = a.q_proj(h).view(1, N, H, d)
            # Gather each frame's full key/value set: survivors' own k/v, plus the k/v
            # carried by every dropped slot from its last surviving copy.
            k = a.k_proj(h).index_select(0, kv_rows).view(1, n_act * P, H, d)
            v = a.v_proj(h).index_select(0, kv_rows).view(1, n_act * P, H, d)

            o = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_bias, scale=a.scale
            ).reshape(N, C)

            x = x + a.out_proj(o)                                          # residual
            x = x + layer.mlp(layer.layer_norm2(x))                        # MLP, survivors only

        if self.apply_post_layernorm:
            x = self.post_layernorm(x)
        # Materialize the dense grid once, at the end: dropped slots take the final
        # encoded state of the survivor they carry.
        return x.index_select(0, src_row).view(T, P, C)

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
            dense: (T, P, C) encoded grid -- survivors at their own slots, dropped slots
                forward-filled from their last surviving copy. Callers that want the
                ragged survivor set gather it with token_mask.
            token_mask: (T*P,) bool keep-mask in (t,h,w) order.
        """
        T = pixel_values.shape[0]
        dev, dtype = pixel_values.device, pixel_values.dtype

        # 1. RLT keep-mask from raw pixels. tubelet_size=1 (SigLIP is a 2D encoder, so a
        #    token is one frame deep).
        #      "ref"    -- diff each patch against the frame its features will actually be
        #                  REUSED from, so drift is bounded by `threshold` by construction.
        #      "consec" -- legacy/paper: diff against frame t-1 while reusing from the last
        #                  SURVIVING frame. Those are different references, so slow drift is
        #                  never detected and accumulates without bound. Kept for ablation.
        x5d = pixel_values.permute(1, 0, 2, 3).unsqueeze(0)            # (1, C, T, H, W)
        if self.mask_mode == "ref":
            token_mask = batched_find_idxs_to_keep_ref(
                x5d, threshold=self.threshold, patch_size=self.patch_size,
                refresh_every=self.refresh_every,
            )
        else:
            token_mask = batched_find_idxs_to_keep(
                x5d, threshold=self.threshold, tubelet_size=1, patch_size=self.patch_size
            )                                                          # (1, T*h*w)
        n_per_frame = token_mask.shape[1] // T
        h = w = int(round(n_per_frame ** 0.5))

        # 2. SigLIP patch-embed (+ spatial pos baked in). (T, P, C)
        emb = self.embeddings(pixel_values)
        P = emb.shape[1]
        assert P == n_per_frame, (
            f"SigLIP grid ({P}) != RLT keep-mask grid ({n_per_frame}); "
            f"check patch_size / image_size alignment."
        )

        # 3. Keep-mask in (t,h,w) order.
        mask_flat = token_mask.reshape(-1)                            # (T*P,)

        # 4. Optionally add a FIXED temporal position phi_t (which frame each token came
        #    from), scaled to the RMS of SigLIP's own position_embedding so it enters at
        #    the magnitude SigLIP uses for position. The scale is DETACHED (it only
        #    rescales a frozen sinusoid), so RLT still adds no trainable state.
        #
        #    NOT NEEDED by either attn_mode here: both keep attention strictly within a
        #    frame, so a token's frame is never ambiguous to the encoder. It would only
        #    become load-bearing if survivors from different frames were ever packed into
        #    one attention block (the RLT paper's per-clip mask). Default 0.0 => disabled.
        emb_pos = emb
        if self.temporal_pos_scale > 0:
            pos = self.temporal_pos.to(device=dev, dtype=dtype)[:T]        # (T, C)
            # Read the spatial position encoding through the module forward (as HF does)
            # so DeepSpeed Zero-3 gathers the sharded weight via the nn.Embedding hook.
            sp = self.embeddings.position_embedding(self.embeddings.position_ids)  # (1,P,C)
            ref_rms = sp.detach().float().pow(2).mean().sqrt()
            pos_rms = pos.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            emb_pos = emb + pos.unsqueeze(1) * (
                self.temporal_pos_scale * ref_rms / pos_rms
            ).to(dtype)                                                    # (T, P, C)

        survivors = emb_pos.reshape(T * P, self.embed_dim)[mask_flat]      # (N, C)

        # 5. Encode. Attention is WITHIN-frame in both modes: SigLIP is a 2D image encoder
        #    whose weights were trained with within-frame attention, and Video-XL-Pro
        #    defers whole-video reasoning to the DTS/SAE stack and the LLM. The modes
        #    differ only in what each frame's survivors are allowed to attend TO.
        mask2d = mask_flat.view(T, P)
        if self.attn_mode == "reuse":
            # Survivors attend over the full P-token frame; dropped tokens supply the
            # keys/values they carry from their last surviving copy. Reduces to dense
            # SigLIP at 100% keep.
            dense = self._run_encoder_reuse(emb_pos, mask2d, T, P)
        else:
            # LEGACY "per_frame": survivors attend ONLY to each other. This starves them
            # of their own frame's context -- a frame keeping 40 of 729 patches runs all
            # encoder layers as a 40-token sequence, when the weights expect 729. Kept
            # only so the ablation can be reproduced; see _run_encoder_reuse.
            seqlens = [int(s) for s in mask2d.sum(dim=1).tolist() if s > 0]
            attn_bias = BlockDiagonalMask.from_seqlens(seqlens)
            surv = self._run_encoder(survivors.unsqueeze(0), attn_bias).squeeze(0)  # (N, C)
            _, src_row = self._reuse_index(mask2d, T, P)
            dense = surv.index_select(0, src_row).view(T, P, self.embed_dim)

        return dense, mask_flat


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

    # controlled clip: frame0 random; 1-2 redundant; top-left patch jumps at t=3; 4-5 redundant
    T = 6
    base = torch.randn(1, 3, 28, 28, device=dev, dtype=dtype)
    frames = base.repeat(T, 1, 1, 1)
    frames[3:, :, :patch, :patch] += 5.0

    def build(mode, thr=0.1):
        return SiglipRLTEmbeddings(vm, threshold=thr, patch_size=patch, max_frames=64,
                                   temporal_pos_scale=0.0, attn_mode=mode).to(dev, dtype).eval()

    for mode in ("reuse", "per_frame"):
        dense, mask_flat = build(mode)(frames)
        N = int(mask_flat.sum())
        print(f"{mode:<10} dense: {tuple(dense.shape)}  (survivors N={N} of {T*P})")
        assert dense.shape == (T, P, cfg.hidden_size)
        assert torch.isfinite(dense.float()).all(), "non-finite outputs from xformers attention"

    # The load-bearing invariant: with nothing dropped, RLT must be a no-op. Anything
    # else means the carried keys/values are not what the dense model would compute.
    truth = vm(frames, output_hidden_states=True).hidden_states[-1]      # (T, P, C)
    for mode in ("reuse", "per_frame"):
        dense, mask_flat = build(mode, thr=-1.0)(frames)                 # keep everything
        assert bool(mask_flat.all()), "threshold=-1 must keep every token"
        err = (dense.float() - truth.float()).abs().max().item()
        assert err < 1e-2, f"{mode}: 100% keep must reduce to dense SigLIP, got max|err|={err}"
        print(f"{mode:<10} 100% keep reproduces dense SigLIP (max|err|={err:.2e})")

    print("\nOK: SigLIP-RLT embeddings run end-to-end and are exact at 100% keep.")
