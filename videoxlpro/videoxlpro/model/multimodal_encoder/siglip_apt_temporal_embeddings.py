"""
siglip_apt_temporal_embeddings.py
==================================

Temporal-Anchored APT (TAPT): combines APT's per-frame spatial partition with
RLT-style temporal redundancy collapsing. See apt_temporal_static_tokens.py
for the classification math and its docstring for the precedence rationale
(spatial first, temporal second) and the REDUNDANT/FRESH/OVERRIDE cases.

This module **composes** an already-built SiglipAPTEmbeddings instance --
no changes to siglip_apt_embeddings.py are needed. It reuses:
  * apt.tokenizer          (APTPatchTokenizer -- entropy/partition + patch
                             grouping, including construct_patch_groups'
                             existing support for applying an EXTERNALLY
                             supplied mask to a frame's pixels, which is
                             exactly what's needed to compute the E(Resize_p)
                             "coarse anchor" term for a merged token whose
                             cell footprint may differ from frame t's own
                             partition, e.g. after an OVERRIDE event).
  * apt._embed_patches      (frozen SigLIP patch_embedding Conv2d -- never
                             shown anything but a native 14x14 patch).
  * apt.patch_attn / apt.zero_conv  (APT's trainable, zero-init spatial merge
                             -- currently untrained per the settled decision
                             not to backfill APT's training wiring here, so
                             these numerically contribute nothing beyond the
                             E(Resize_p) term today, but are wired in for
                             forward-compatibility: if APT training is ever
                             backfilled, TAPT benefits without a rewrite).
  * apt._run_encoder        (xformers block-diagonal attention over the
                             frozen SigLIP transformer layers).
  * apt.base_pos_embed / apt.base_grid_size (position embedding resampling).

New trainable parameter: a run-length embedding (reuse_len_embed), the exact
same "model-owned, zero-init, __dict__-stashed-when-injected" pattern as
RLT's phi_L (siglip_rlt_embeddings.py) -- tags every survivor with how many
frames it has stood in for the model to learn to read.

Everything through classification/run-length/origin-index is fully
vectorized (see apt_temporal_static_tokens.py). The only per-scale looping
here mirrors what SiglipAPTEmbeddings._embed already does (batch all events
of a given scale together), it is not a per-frame Python loop.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from timm.layers import resample_abs_pos_embed

try:
    from .siglip_apt_embeddings import SiglipAPTEmbeddings
    from .apt_temporal_static_tokens import (
        REDUNDANT, FRESH, OVERRIDE,
        dense_scale_code_grid, dirty_subtile_mask, shape_match_grid,
        reference_cell_dirty_stats, classify_cells,
        compute_origin_index, compute_run_lengths,
    )
except ImportError:  # allow running this file directly as a script for testing
    from siglip_apt_embeddings import SiglipAPTEmbeddings
    from apt_temporal_static_tokens import (
        REDUNDANT, FRESH, OVERRIDE,
        dense_scale_code_grid, dirty_subtile_mask, shape_match_grid,
        reference_cell_dirty_stats, classify_cells,
        compute_origin_index, compute_run_lengths,
    )


def apt_temporal_scatter_back(combined_survivors: torch.Tensor, origin_index: torch.Tensor,
                               T: int, P: int) -> torch.Tensor:
    """Broadcast TAPT's packed, encoded survivors back to a dense (T, P, C) grid.

    origin_index: (T, G, G) long, already fully resolved by
    SiglipAPTTemporalEmbeddings.forward -- for every (t, base cell), which row
    of combined_survivors currently supplies its value (REDUNDANT cells point
    at whichever FRESH/OVERRIDE event most recently refreshed that spatial
    position; FRESH/OVERRIDE cells point at their own row).
    """
    C = combined_survivors.shape[-1]
    G = origin_index.shape[-1]
    dense = combined_survivors[origin_index.reshape(-1)]   # (T*G*G, C)
    dense = dense.view(T, G * G, C)
    assert dense.shape[1] == P, f"apt_temporal_scatter_back: P mismatch {dense.shape[1]} != {P}"
    assert torch.isfinite(dense.float()).all(), "non-finite TAPT dense grid"
    return dense


class SiglipAPTTemporalEmbeddings(nn.Module):
    """Wraps a SiglipAPTEmbeddings instance to additionally collapse temporal
    redundancy across frames (see module docstring).

    Args:
        apt: an already-constructed SiglipAPTEmbeddings (its tokenizer,
            patch_attn, zero_conv, base_pos_embed, etc. are reused directly).
        threshold: mean-abs pixel-L1 threshold above which a base tile counts
            as "dirty" relative to the previous frame. Deliberately the SAME
            knob and SAME pixel scale as RLT's own rlt_threshold (SigLIP-
            normalized pixels, no un-normalize step -- see dirty_subtile_mask)
            rather than a second, differently-scaled threshold: both modes are
            answering the identical underlying question ("did this patch
            change vs. the previous frame?"), so config.rlt_threshold is
            reused directly at the llava_arch.py wiring level.
        majority_ratio: fraction of dirty sub-tiles above which a
            shape-mismatched cell is treated as independent (FRESH) rather
            than merge-overridden (OVERRIDE).
        max_frames: size of the run-length embedding table (a token can be
            tagged with a run-length up to max_frames-1's worth of reuse).
        reuse_len_embed: optional model-owned nn.Embedding (mirrors RLT's
            phi_L sharing pattern exactly, siglip_rlt_embeddings.py:157-161)
            -- stashed via __dict__ so this wrapper's own .to()/.cuda() can't
            clone the shared, checkpoint-tracked copy. When None, a
            standalone zero-init embedding is created (eval/testing only).
    """

    def __init__(
        self,
        apt: SiglipAPTEmbeddings,
        threshold: float = 0.1,
        majority_ratio: float = 0.5,
        max_frames: int = 512,
        reuse_len_embed: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.apt = apt
        self.threshold = threshold
        self.majority_ratio = majority_ratio
        self.max_frames = max_frames

        if reuse_len_embed is not None:
            self.__dict__["reuse_len_embed"] = reuse_len_embed
        else:
            self.reuse_len_embed = nn.Embedding(max_frames, apt.embed_dim)
            nn.init.zeros_(self.reuse_len_embed.weight)

    def _resize_to_apt_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        apt = self.apt
        if pixel_values.shape[-1] == apt.image_size and pixel_values.shape[-2] == apt.image_size:
            return pixel_values
        return F.interpolate(
            pixel_values.float(), size=(apt.image_size, apt.image_size),
            mode="bilinear", align_corners=False,
        ).to(pixel_values.dtype)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Args:
            pixel_values: (T, 3, H, W) frames for ONE clip.

        Returns:
            combined_survivors: (N, C) encoded packed tokens (N <= sum over
                scales of new-token event counts).
            origin_index: (T, G, G) long, for apt_temporal_scatter_back.
            T: frames in the clip.
            P: dense patch count per frame (G*G), matching one clip's vision
                output (same contract as apt_scatter_back / _rlt_scatter_back).
        """
        apt = self.apt
        T = pixel_values.shape[0]
        frames = self._resize_to_apt_input(pixel_values)
        base_p = apt.base_patch_size
        G = apt.image_size // base_p
        C = apt.embed_dim
        dev = frames.device

        input_dict = apt.tokenizer(frames)
        masks = input_dict["masks"]

        scale_grid = dense_scale_code_grid(masks, apt.patch_sizes, base_p)

        # Dirty check on the SAME (SigLIP-normalized) pixel scale RLT itself
        # uses -- no un-normalize step -- so self.threshold is directly the
        # shared config.rlt_threshold value, not a second differently-scaled
        # knob (see dirty_subtile_mask's docstring).
        dirty = dirty_subtile_mask(frames.float(), self.threshold, base_p)

        shape_match = shape_match_grid(scale_grid, masks, apt.patch_sizes, base_p)
        all_quiet, majority_dirty = reference_cell_dirty_stats(
            dirty, scale_grid, apt.patch_sizes, base_p, self.majority_ratio
        )
        cls = classify_cells(shape_match, all_quiet, majority_dirty)

        is_new_token = cls != REDUNDANT
        needs_fresh_embed = (cls == FRESH) | ((cls == OVERRIDE) & dirty)
        tile_source_frame = compute_origin_index(needs_fresh_embed)
        token_origin_frame = compute_origin_index(is_new_token)
        run_length = compute_run_lengths(is_new_token)

        # OVERRIDE events adopt frame t-1's own scale/boundary instead of
        # frame t's (mismatched) one; FRESH/REDUNDANT keep frame t's own.
        adopted_scale_grid = scale_grid.clone()
        adopted_scale_grid[1:] = torch.where(cls[1:] == OVERRIDE, scale_grid[:-1], scale_grid[1:])

        # ---- base-tile embedding cache: E(patch)+pos, computed once per
        # position wherever needed, gathered forward via tile_source_frame
        # for positions that reuse an earlier frame's still-valid embedding.
        patches = frames.unfold(2, base_p, base_p).unfold(3, base_p, base_p)   # (T,3,G,G,p,p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()               # (T,G,G,3,p,p)
        fresh_patches = patches[needs_fresh_embed].reshape(-1, 3, base_p, base_p)
        fresh_vals = apt._embed_patches(fresh_patches)                        # (Nfresh, C)
        pos_grid = apt.base_pos_embed.to(frames.dtype).view(G, G, C)
        pos_for_fresh = pos_grid.unsqueeze(0).expand(T, G, G, C)[needs_fresh_embed]
        fresh_vals = fresh_vals + pos_for_fresh

        full_tile_written = torch.zeros(T, G, G, C, device=dev, dtype=fresh_vals.dtype)
        full_tile_written[needs_fresh_embed] = fresh_vals
        gather_idx = tile_source_frame.unsqueeze(-1).expand(T, G, G, C)
        full_tile_embed = torch.gather(full_tile_written, 0, gather_idx)      # (T,G,G,C)

        # ---- merge pass, grouped by scale (scale-major order first; frame
        # index recorded per event so the packed sequence can be reordered to
        # frame-major -- required for per-frame BlockDiagonalMask boundaries).
        scale_tokens, scale_frame_idx, scale_events = [], [], []
        for idx, ps in enumerate(apt.patch_sizes):
            code = idx + 1
            s = ps // base_p
            coarse_adopted = adopted_scale_grid[:, ::s, ::s]
            coarse_new = is_new_token[:, ::s, ::s]
            event_mask = (coarse_adopted == code) & coarse_new            # (T, G//s, G//s)
            n_events = int(event_mask.sum().item())
            scale_events.append((event_mask, s))
            if n_events == 0:
                continue

            if s == 1:
                token_value = full_tile_embed[event_mask]                  # (n_events, C)
            else:
                children = full_tile_embed.view(T, G // s, s, G // s, s, C)
                children = children.permute(0, 1, 3, 2, 4, 5)              # (T,Gs,Gs,s,s,C)
                children = children[event_mask]                            # (n_events,s,s,C)
                attn_in = children.permute(0, 3, 1, 2).contiguous()        # (n_events,C,s,s)
                for _ in range(idx):
                    attn_in = apt.patch_attn(attn_in)
                attn_out = attn_in.squeeze(-1).squeeze(-1)                 # (n_events, C)
                merged = apt.zero_conv(attn_out)

                event_masks_for_groups = {
                    p: (event_mask.to(frames.dtype) if p == ps
                        else torch.zeros(T, apt.image_size // p, apt.image_size // p,
                                          device=dev, dtype=frames.dtype))
                    for p in apt.patch_sizes
                }
                patch_groups = apt.tokenizer.construct_patch_groups(frames, event_masks_for_groups)
                resize_patches = patch_groups[f"resized_patches_{ps}"]
                embed_scale = apt._embed_patches(resize_patches)           # (n_events, C)

                new_g = apt.image_size // ps
                resampled_pos = resample_abs_pos_embed(
                    apt.base_pos_embed.to(frames.dtype), new_size=(new_g, new_g),
                    old_size=(apt.base_grid_size, apt.base_grid_size), num_prefix_tokens=0,
                )
                pos_mask = patch_groups[f"pos_embed_mask_{ps}"]
                pos_for_events = resampled_pos.repeat(T, 1, 1)[pos_mask]

                token_value = merged + embed_scale.to(merged.dtype) + pos_for_events.to(merged.dtype)

            run_length_repr = run_length[:, ::s, ::s][event_mask]          # (n_events,)
            length_idx = (run_length_repr - 1).clamp(min=0, max=self.reuse_len_embed.num_embeddings - 1)
            token_value = token_value + self.reuse_len_embed(length_idx).to(token_value.dtype)

            frame_idx_for_events = torch.nonzero(event_mask, as_tuple=False)[:, 0]  # (n_events,)
            scale_tokens.append(token_value)
            scale_frame_idx.append(frame_idx_for_events)

        combined_scale_major = torch.cat(scale_tokens, dim=0)              # (N, C)
        frame_idx_scale_major = torch.cat(scale_frame_idx, dim=0)          # (N,)

        # Reorder to frame-major so per-frame blocks are contiguous for the
        # BlockDiagonalMask (mirrors the per-frame-only attention convention
        # both SiglipAPTEmbeddings and SiglipRLTEmbeddings already use).
        order = torch.argsort(frame_idx_scale_major, stable=True)
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=dev)
        combined_frame_major = combined_scale_major[order]

        sorted_frame_idx = frame_idx_scale_major[order]
        seqlens = [int((sorted_frame_idx == t).sum().item()) for t in range(T)]
        seqlens = [n for n in seqlens if n > 0]                            # skip fully-redundant frames

        # Rebuild packed_row_id (scale-major) -> remap to frame-major via inverse_order.
        packed_row_id = torch.full((T, G, G), -1, dtype=torch.long, device=dev)
        row_offset = 0
        for event_mask, s in scale_events:
            n_events = int(event_mask.sum().item())
            if n_events == 0:
                continue
            ids_coarse = torch.full_like(event_mask, -1, dtype=torch.long)
            ids_coarse[event_mask] = torch.arange(n_events, device=dev) + row_offset
            ids_base = ids_coarse.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
            packed_row_id = torch.where(ids_base >= 0, ids_base, packed_row_id)
            row_offset += n_events
        valid = packed_row_id >= 0
        packed_row_id = torch.where(valid, inverse_order[packed_row_id.clamp(min=0)],
                                     torch.full_like(packed_row_id, -1))

        origin_index = torch.gather(packed_row_id, 0, token_origin_frame)  # (T,G,G)
        assert (origin_index >= 0).all(), "every base cell must resolve to a valid packed row"

        attn_bias = BlockDiagonalMask.from_seqlens(seqlens)
        x = combined_frame_major.unsqueeze(0)                              # (1,N,C)
        combined_survivors = apt._run_encoder(x, attn_bias).squeeze(0)     # (N,C)

        P = G * G
        return combined_survivors, origin_index, T, P


if __name__ == "__main__":
    # ---- standalone test: tiny random-weight SigLIP, CUDA --------------------
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    assert torch.cuda.is_available(), "xformers memory_efficient_attention needs CUDA"
    dev, dtype = "cuda", torch.float16

    base_p, img = 14, 56               # 56/14=4x4 base grid; 2 scales (14/28)
    num_scales = 2
    cfg = SiglipVisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_channels=3, image_size=img, patch_size=base_p,
    )
    vm = SiglipVisionModel(cfg).to(dev, dtype).eval()

    apt = SiglipAPTEmbeddings(
        vm, thresholds=[5.0], num_scales=num_scales,
        base_patch_size=base_p, image_size=img,
    ).to(dev, dtype).eval()

    tapt = SiglipAPTTemporalEmbeddings(
        apt, threshold=0.1, majority_ratio=0.5, max_frames=16,   # 0.1: RLT's own default scale
    ).to(dev, dtype).eval()

    G = img // base_p

    # --- Test 1: a fully static clip (byte-identical frames) must collapse to
    #     exactly as many fresh events as plain per-frame APT would produce
    #     for frame 0 ALONE (whatever scale that frame's own entropy picks --
    #     not necessarily 1, since a 4x4 base grid with only 2 scales can't
    #     merge below four 28px cells), with every later frame fully
    #     redundant and broadcasting frame 0's values identically.
    torch.manual_seed(0)
    T = 5
    static_frame = torch.rand(1, 3, img, img, device=dev, dtype=dtype) * 0.02 + 0.49  # near-uniform -> low entropy -> merges
    static_clip = static_frame.repeat(T, 1, 1, 1)
    # Normalize to roughly SigLIP's expected range so the tokenizer's
    # un-normalize round-trips sensibly (mean/std=0.5 like the other smoke tests).
    static_clip_norm = (static_clip - 0.5) / 0.5

    survivors, origin_index, T_out, P = tapt(static_clip_norm)
    N = survivors.shape[0]
    ref0_surv0, ref0_mask0, _, _, _ = apt(static_clip_norm[0:1])
    expected_n = ref0_mask0.numel()
    print(f"[static clip] survivors: {tuple(survivors.shape)} (N={N}, frame0-alone would need "
          f"{expected_n}, dense would be {T*P})")
    assert N == expected_n, (
        f"a fully static clip must produce exactly frame 0's own per-frame-APT token count, "
        f"got {N} vs {expected_n}"
    )
    assert torch.isfinite(survivors.float()).all(), "non-finite survivors"

    dense = apt_temporal_scatter_back(survivors, origin_index, T_out, P)
    assert dense.shape == (T, P, cfg.hidden_size)
    for t in range(1, T):
        assert torch.allclose(dense[t].float(), dense[0].float(), atol=1e-2), (
            f"static clip: frame {t} must exactly match frame 0's broadcast value"
        )
    # Value-level correctness: frame 0's TAPT output must exactly match
    # plain per-frame APT's own merge/embed computation (patch_attn +
    # zero_conv + resize term + position embed), not just be self-consistent.
    from siglip_apt_embeddings import apt_scatter_back
    ref0_dense = apt_scatter_back(ref0_surv0, ref0_mask0, apt.tokenizer(static_clip_norm[0:1])["masks"], base_p, img)
    assert torch.allclose(dense[0].float(), ref0_dense[0].float(), atol=1e-2), (
        "static clip: frame 0's merged/embedded value must match plain per-frame APT exactly"
    )
    print(f"OK: fully static clip -> {N} fresh events (frame 0's own scale), "
          f"broadcast identically + value-matched to every frame.")

    # --- Test 2: max_window=1-equivalent sanity -- a 2-frame clip where frame
    #     1 is completely different (independent, high-entropy) content must
    #     produce 2 fresh events (no illegitimate collapsing across unrelated
    #     content), and running plain per-frame APT on each frame separately
    #     must give the same per-frame dense values TAPT produces.
    torch.manual_seed(1)
    f0 = torch.rand(1, 3, img, img, device=dev, dtype=dtype)
    f1 = torch.rand(1, 3, img, img, device=dev, dtype=dtype)
    clip2 = torch.cat([f0, f1], dim=0)
    clip2_norm = (clip2 - 0.5) / 0.5

    survivors2, origin2, T2, P2 = tapt(clip2_norm)
    dense2 = apt_temporal_scatter_back(survivors2, origin2, T2, P2)
    assert torch.isfinite(dense2.float()).all()

    # Independently run plain per-frame APT on each frame.
    from siglip_apt_embeddings import apt_scatter_back
    ref0_surv, ref0_mask, ref0_masks, ref0_T, ref0_P = apt(clip2_norm[0:1])
    ref0_dense = apt_scatter_back(ref0_surv, ref0_mask, ref0_masks, base_p, img)
    ref1_surv, ref1_mask, ref1_masks, ref1_T, ref1_P = apt(clip2_norm[1:2])
    ref1_dense = apt_scatter_back(ref1_surv, ref1_mask, ref1_masks, base_p, img)

    assert torch.allclose(dense2[0].float(), ref0_dense[0].float(), atol=1e-2), (
        "frame 0 of an all-independent clip must match plain per-frame APT"
    )
    assert torch.allclose(dense2[1].float(), ref1_dense[0].float(), atol=1e-2), (
        "frame 1 of an all-independent clip (fully independent content) must "
        "match plain per-frame APT applied to that frame alone"
    )
    print("OK: fully independent 2-frame clip matches plain per-frame APT on each frame.")

    # --- Test 3: mixed clip (fresh -> redundant -> fresh again). The middle
    #     frame contributes ZERO packed rows, exercising the "skip fully
    #     redundant frames in seqlens" path and the frame-major reordering
    #     when frames don't contribute contiguously in scale-major order.
    torch.manual_seed(2)
    g0 = torch.rand(1, 3, img, img, device=dev, dtype=dtype) * 0.02 + 0.3     # low entropy -> merges
    g2 = torch.rand(1, 3, img, img, device=dev, dtype=dtype)                  # unrelated, high entropy
    clip3 = torch.cat([g0, g0.clone(), g2], dim=0)                            # f1 identical to f0
    clip3_norm = (clip3 - 0.5) / 0.5

    survivors3, origin3, T3, P3 = tapt(clip3_norm)
    dense3 = apt_temporal_scatter_back(survivors3, origin3, T3, P3)

    ref0_surv3, ref0_mask3, ref0_masks3, _, _ = apt(clip3_norm[0:1])
    ref0_dense3 = apt_scatter_back(ref0_surv3, ref0_mask3, ref0_masks3, base_p, img)
    ref2_surv3, ref2_mask3, ref2_masks3, _, _ = apt(clip3_norm[2:3])
    ref2_dense3 = apt_scatter_back(ref2_surv3, ref2_mask3, ref2_masks3, base_p, img)

    assert torch.allclose(dense3[0].float(), ref0_dense3[0].float(), atol=1e-2), "f0 mismatch"
    assert torch.allclose(dense3[1].float(), dense3[0].float(), atol=1e-2), (
        "f1 (identical to f0) must be fully redundant -> broadcast f0's value"
    )
    assert torch.allclose(dense3[2].float(), ref2_dense3[0].float(), atol=1e-2), (
        "f2 (unrelated content) must be freshly embedded, matching plain per-frame APT"
    )
    print("OK: mixed fresh->redundant->fresh clip: middle frame contributes zero rows, "
          "endpoints value-match plain per-frame APT.")

    print("\nOK: SigLIP-APT-Temporal embeddings run end-to-end "
          "(classification -> embed/merge -> xformers attn -> scatter-back).")
