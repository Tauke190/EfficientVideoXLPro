"""
siglip_apt_embeddings.py
========================

Adaptive Patch Transformer (APT) at the SigLIP embedding seam.

This module re-embeds each frame with APT's hierarchical (quadtree) patchifier
*before* SigLIP's transformer blocks, so attention runs over a reduced,
content-adaptive token set.  It is the faithful APT mechanism (arXiv 2510.18091,
Eq. 2), NOT mean-pooling:

    E(P_i) = ZeroMLP( Conv2d^i( { E(P_j) | P_j subset P_i } ) ) + E( Resize_p(P_i) )

mapped onto SigLIP's own modules:
  * E              = SigLIP's frozen patch_embedding Conv2d (applied per 14px patch).
  * Resize_p(P_i)  = the coarse region resized to one 14px patch, embedded by E.
  * Conv2d^i       = a single trainable strided Conv2d (k2,s2) applied i times to
                     the grid of constituent sub-patch embeddings (self.patch_attn).
  * ZeroMLP        = a zero-initialized Linear (self.zero_conv).  Zero-init means
                     a coarse token starts equal to E(Resize_p(P_i)), so the
                     pretrained model is undisturbed; a short finetune (<= 1 epoch,
                     lr ~1e-6) folds in sub-patch detail.

Positions: SigLIP's learned position embedding is **interpolated/resampled** from
the base grid down to each coarse grid (timm.resample_abs_pos_embed), generalizing
position across scales while keeping spatial consistency.  No averaging.

Grid/divisibility: SigLIP's native 27x27 grid is odd (no clean 2x/4x quadtree).
We run APT at **392x392 -> 28x28** (28 = 4*7, divisible by both merge factors),
which needs only a one-time 27->28 interpolation of the base position embedding.

SigLIP is **cls-free** (num_positions == num_patches), so there is no class token
to carry (num_prefix_tokens = 0).

Trainable vs frozen: the SigLIP backbone (patch_embedding + encoder layers) is
frozen; only patch_attn and zero_conv train.

NOTE (downstream): forward returns ragged survivors plus the partition metadata.
apt_scatter_back() broadcasts them back to a dense (T, P, C) grid for the existing
Video-XL-Pro pipeline (interpolate -> add_video -> SAE -> 2D pool).  Wiring into
llava_arch is a separate step.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from timm.layers import resample_abs_pos_embed

try:
    from .apt_static_tokens import APTPatchTokenizer
except ImportError:  # allow running this file directly as a script for testing
    from apt_static_tokens import APTPatchTokenizer


def apt_scatter_back(
    encoded: torch.Tensor,
    output_mask: torch.Tensor,
    masks: Dict[int, torch.Tensor],
    base_patch_size: int,
    image_size: int,
) -> torch.Tensor:
    """Broadcast ragged APT survivors back to a dense (T, P, C) grid.

    APT's masks are a strict spatial partition: each base cell is covered by
    exactly one surviving patch (at scale 1, 2 or 3).  A coarse token is the
    representation of its whole region, so we **nearest-upsample** each coarse
    token over the base cells it covers.  Disjoint + exhaustive => every base
    cell written exactly once.

    Args:
        encoded: (N, C) encoded survivors, frame-major then scale-major raster.
        output_mask: (N,) scale codes {1, 2, ...} aligned with `encoded`.
        masks: dict {ps: (T, G_s, G_s)} 0/1 partition masks from the tokenizer.
        base_patch_size: base patch size p (14).
        image_size: APT input resolution (392) -> base grid G = image_size // p.

    Returns:
        (T, P, C) dense feature grid, P = G*G, matching one clip's vision output.
    """
    T = next(iter(masks.values())).shape[0]
    G = image_size // base_patch_size
    C = encoded.shape[-1]
    num_scales = len(masks)

    dense = encoded.new_zeros(T, G, G, C)
    covered = encoded.new_zeros(T, G, G)
    for idx in range(num_scales):
        ps = base_patch_size * 2 ** idx
        s = ps // base_patch_size
        code = idx + 1
        m = masks[ps].bool()                                   # (T, G_s, G_s)
        toks = encoded[output_mask == code]                    # (n, C) raster order
        grid = encoded.new_zeros(m.shape + (C,))               # (T, G_s, G_s, C)
        grid[m] = toks.to(grid.dtype)
        up = grid.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)   # (T, G, G, C)
        mup = m.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)     # (T, G, G)
        dense[mup] = up[mup]
        covered += mup.to(covered.dtype)

    assert torch.all(covered == 1), "apt_scatter_back: partition must cover each base cell exactly once"
    return dense.view(T, G * G, C)


class SiglipAPTEmbeddings(nn.Module):
    """APT wrapper around a HuggingFace SiglipVisionModel.

    Args:
        vision_model: a transformers SiglipVisionModel (.vision_model holds
            embeddings / encoder.layers / post_layernorm).
        thresholds: per-non-base-scale entropy thresholds, length num_scales-1.
        num_scales: number of patch scales (3 -> 14/28/56).
        base_patch_size: SigLIP patch size (14).
        image_size: APT input resolution (392 -> 28x28 base grid).
        apply_post_layernorm: Video-XL-Pro consumes hidden_states[-1] (pre-LN), so
            default False to match what the projector was trained on.
    """

    def __init__(
        self,
        vision_model,
        thresholds,
        num_scales: int = 3,
        base_patch_size: int = 14,
        image_size: int = 392,
        apply_post_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        vm = vision_model.vision_model
        self.embeddings = vm.embeddings
        self.encoder_layers = vm.encoder.layers
        self.post_layernorm = vm.post_layernorm

        self.embed_dim = self.embeddings.embed_dim
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.num_scales = num_scales
        self.apply_post_layernorm = apply_post_layernorm
        self.patch_sizes = [base_patch_size * (2 ** i) for i in range(num_scales)]

        # SigLIP's frozen patch embedder = APT's base embedder E.
        self.patch_embedding = self.embeddings.patch_embedding

        # APT tokenizer (pure tensor helper; no learnable params).
        self.tokenizer = APTPatchTokenizer(
            num_scales=num_scales,
            base_patch_size=base_patch_size,
            image_size=image_size,
            thresholds=thresholds,
        )

        # One-time interpolation of SigLIP's base position embedding to the
        # 28x28 grid the 392px input induces.  Frozen -> precompute as a buffer.
        with torch.no_grad():
            src = self.embeddings.position_embedding.weight.detach().float().unsqueeze(0)  # (1, 729, C)
            old_g = int(round(src.shape[1] ** 0.5))
            new_g = image_size // base_patch_size
            base_pos = resample_abs_pos_embed(
                src, new_size=(new_g, new_g), old_size=(old_g, old_g), num_prefix_tokens=0
            )                                                                              # (1, G*G, C)
        self.register_buffer("base_pos_embed", base_pos, persistent=False)
        self.base_grid_size = new_g

        # NEW trainable modules (the only params APT adds).
        self.patch_attn = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2)
        self.zero_conv = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    # ------------------------------------------------------------------ #
    # SigLIP attention/encoder reuse (verbatim from siglip_rlt_embeddings.py)
    # ------------------------------------------------------------------ #
    def _xformers_attention(self, self_attn, x: torch.Tensor, attn_bias) -> torch.Tensor:
        B, N, C = x.shape
        H, d = self_attn.num_heads, self_attn.head_dim
        q = self_attn.q_proj(x).view(B, N, H, d)
        k = self_attn.k_proj(x).view(B, N, H, d)
        v = self_attn.v_proj(x).view(B, N, H, d)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self_attn.scale)
        out = out.reshape(B, N, C)
        return self_attn.out_proj(out)

    def _run_encoder(self, x: torch.Tensor, attn_bias) -> torch.Tensor:
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

    # ------------------------------------------------------------------ #
    # APT hierarchical embedding (Eq. 2), re-pointed to SigLIP
    # ------------------------------------------------------------------ #
    def _embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """E(.): SigLIP's frozen patch conv applied per (-1, 3, p, p) patch -> (-1, C)."""
        return self.patch_embedding(patches).reshape(-1, self.embed_dim)

    def _embed(self, frames: torch.Tensor, input_dict: Dict) -> torch.Tensor:
        """Build the packed (1, N, C) survivor embeddings per APT Eq. 2."""
        batch_size = frames.shape[0]
        base_p = self.base_patch_size
        output_mask = input_dict["output_mask"]
        base_pos_embed = self.base_pos_embed.to(frames.dtype)            # (1, G*G, C)

        # Base scale (code 1): plain E + native (resampled) position embed.
        base16 = input_dict[f"resized_patches_{base_p}"]
        posmask_16 = input_dict[f"pos_embed_mask_{base_p}"]
        pos_embed16 = base_pos_embed.repeat(batch_size, 1, 1)[posmask_16]
        embed16 = self._embed_patches(base16) + pos_embed16

        expanded = torch.zeros(
            (output_mask.shape[0], self.embed_dim), device=frames.device, dtype=embed16.dtype
        )
        expanded[output_mask == 1] = embed16

        # Coarse scales (codes 2..num_scales): Eq. 2.
        for scale_idx, cur_patch_size in enumerate(self.patch_sizes[1:]):
            base_patches = input_dict[f"resized_patches_{cur_patch_size}"]
            full_patches = input_dict[f"full_patches_{cur_patch_size}"]
            pos_embed_masks = input_dict[f"pos_embed_mask_{cur_patch_size}"]

            new_grid_size = self.image_size // cur_patch_size
            resampled_pos_embed = resample_abs_pos_embed(
                base_pos_embed,
                new_size=(new_grid_size, new_grid_size),
                old_size=(self.base_grid_size, self.base_grid_size),
                num_prefix_tokens=0,
            )
            pos_embed = resampled_pos_embed.repeat(batch_size, 1, 1)[pos_embed_masks]

            if pos_embed_masks.sum() > 0:
                embed_scale = self._embed_patches(base_patches)          # E(Resize_p(P_i))
                n = cur_patch_size // base_p
                full_patches = full_patches.reshape(-1, 3, base_p, base_p)
                full_patches = self._embed_patches(full_patches).view(-1, n, n, self.embed_dim)
                full_patches = full_patches.permute(0, 3, 1, 2)          # (n_s, C, n, n)
                for _ in range(scale_idx + 1):                           # Conv2d^i
                    full_patches = self.patch_attn(full_patches)
                attn_scale = full_patches.squeeze(-1).squeeze(-1)        # (n_s, C)
                embed_scale = self.zero_conv(attn_scale) + embed_scale + pos_embed
            else:
                # No survivors at this scale: run zeros through the trainable
                # modules so their params stay in the autograd/DDP graph.
                dummy_base = torch.zeros((1, 3, base_p, base_p), device=frames.device, dtype=embed16.dtype)
                embed_scale = self._embed_patches(dummy_base)
                dummy_full = torch.zeros((1, 3, base_p * 2, base_p * 2), device=frames.device, dtype=embed16.dtype)
                dummy_full = self._embed_patches(dummy_full.reshape(-1, 3, base_p, base_p)).reshape(1, 2, 2, self.embed_dim)
                dummy_full = dummy_full.permute(0, 3, 1, 2)
                attn_scale = self.patch_attn(dummy_full).squeeze(-1).squeeze(-1)
                embed_scale = self.zero_conv(attn_scale) + embed_scale

            expanded[output_mask == (scale_idx + 2)] = embed_scale.to(expanded.dtype)

        return expanded.unsqueeze(0)                                     # (1, N, C)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], int, int]:
        """
        Args:
            pixel_values: (T, 3, H, W) frames for ONE clip (H=W=384 from the
                SigLIP processor; resized to self.image_size internally).

        Returns:
            survivors: (N, C) encoded surviving tokens.
            output_mask: (N,) scale codes aligned with survivors.
            masks: dict {ps: (T, G_s, G_s)} partition masks (for scatter-back).
            T: frames in the clip.
            P: dense patch count per frame (G*G).
        """
        T = pixel_values.shape[0]
        frames = pixel_values
        if frames.shape[-1] != self.image_size or frames.shape[-2] != self.image_size:
            frames = F.interpolate(
                frames.float(), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).to(pixel_values.dtype)

        input_dict = self.tokenizer(frames)
        expanded = self._embed(frames, input_dict)                       # (1, N, C)

        seqlens = [int(s) for s in input_dict["seqlens"] if s > 0]
        attn_bias = BlockDiagonalMask.from_seqlens(seqlens)
        survivors = self._run_encoder(expanded, attn_bias).squeeze(0)    # (N, C)

        P = (self.image_size // self.base_patch_size) ** 2
        return survivors, input_dict["output_mask"], input_dict["masks"], T, P


if __name__ == "__main__":
    # ---- standalone test: tiny random-weight SigLIP, CUDA --------------------
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    assert torch.cuda.is_available(), "xformers memory_efficient_attention needs CUDA"
    dev, dtype = "cuda", torch.float16

    base_p, img = 14, 56               # tiny: 56/14 = 4x4 base grid; scales 14/28/56
    num_scales = 3
    cfg = SiglipVisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_channels=3, image_size=img, patch_size=base_p,
    )
    vm = SiglipVisionModel(cfg).to(dev, dtype).eval()
    G = img // base_p
    print(f"tiny SigLIP: base grid {G}x{G}={G*G}, hidden={cfg.hidden_size}")

    apt = SiglipAPTEmbeddings(
        vm, thresholds=[5.0, 5.0], num_scales=num_scales,
        base_patch_size=base_p, image_size=img,
    ).to(dev, dtype).eval()

    # Clip where the left half is flat (low entropy -> merges to coarse) and the
    # right half is noise (stays at base scale).
    T = 3
    frames = torch.zeros(T, 3, img, img, device=dev, dtype=dtype)
    frames[:, :, :, img // 2:] = torch.rand(T, 3, img, img // 2, device=dev, dtype=dtype)
    frames = (frames - 0.5) / 0.5

    survivors, output_mask, masks, Tt, P = apt(frames)
    N = output_mask.numel()
    print(f"survivors: {tuple(survivors.shape)} (N={N}, dense would be {T*P})")
    counts = {int(c): int((output_mask == c).sum()) for c in output_mask.unique().tolist()}
    print(f"per-scale survivor counts (code:count): {counts}")
    assert N < T * P, "APT must reduce the token count"
    assert torch.isfinite(survivors.float()).all(), "non-finite survivors"

    # --- zero-init property (no initial degradation): at init, every coarse
    #     token == E(Resize_p(P_i)) + interpolated pos (zero_conv contributes 0).
    input_dict = apt.tokenizer(frames)
    expanded = apt._embed(frames, input_dict).squeeze(0)                 # (N, C)
    om = input_dict["output_mask"]
    for scale_idx, ps in enumerate(apt.patch_sizes[1:]):
        if (om == scale_idx + 2).sum() == 0:
            continue
        ref = apt._embed_patches(input_dict[f"resized_patches_{ps}"])
        new_g = img // ps
        rp = resample_abs_pos_embed(
            apt.base_pos_embed.to(dtype), new_size=(new_g, new_g),
            old_size=(G, G), num_prefix_tokens=0,
        )
        ref = ref + rp.repeat(T, 1, 1)[input_dict[f"pos_embed_mask_{ps}"]]
        got = expanded[om == scale_idx + 2]
        assert torch.allclose(got.float(), ref.float(), atol=1e-2), \
            f"zero-init: scale {ps} token must equal E(Resize)+pos at init"
    print("OK: zero-init -> coarse tokens equal E(Resize_p)+pos (no initial degradation).")

    # --- scatter-back tiles the dense grid exactly.
    dense = apt_scatter_back(survivors, output_mask, masks, base_p, img)
    assert dense.shape == (T, P, cfg.hidden_size), f"dense shape {tuple(dense.shape)} != {(T, P, cfg.hidden_size)}"
    assert torch.isfinite(dense.float()).all(), "non-finite dense grid"
    print(f"scatter-back: dense {tuple(dense.shape)} OK (every base cell covered once).")
    print("\nOK: SigLIP-APT embeddings run end-to-end (embed -> xformers attn -> scatter-back).")
