"""
apt_temporal_static_tokens.py
==============================

Pure-tensor primitives for combining APT's per-frame spatial partition with
RLT-style temporal redundancy collapsing ("APT-Temporal" / TAPT -- see
siglip_apt_temporal_embeddings.py for the SigLIP-specific wrapper). No SigLIP
weights here, same tier as apt_static_tokens.py / rlt_static_tokens.py.

Precedence (settled by design, not incidental): spatial redundancy is
resolved FIRST -- every frame gets its own independent entropy-driven
quadtree partition via APTPatchTokenizer, already computed batched across the
whole clip (compute_patch_entropy_batched / select_patches_by_threshold
already operate per-frame with zero cross-frame mixing, so masks[ps] already
comes out as (T, G_s, G_s) -- no new code needed for that step). Temporal
redundancy is resolved SECOND, per base cell, comparing frame t against frame
t-1:

  * shape match (frame t's cell exists at the same scale/boundary in frame
    t-1's own partition too) + zero dirty subtiles  -> REDUNDANT: collapse,
    no fresh embed, extend the running temporal lineage.
  * shape match + any dirty subtile                 -> FRESH: re-embed at
    frame t's own scale, exactly like ordinary per-frame APT.
  * shape mismatch (frame t's own entropy decision disagrees with frame
    t-1's) + majority of subtiles dirty              -> FRESH, same as above.
    This is the case pure spatial entropy can't catch on its own: a region
    can look spatially uniform *within* one frame (low local variance ->
    "mergeable") while having changed completely since the previous frame (a
    flat region's value shifting) -- the temporal signal overrides the
    spatial merge decision here.
  * shape mismatch + minority of subtiles dirty       -> OVERRIDE: adopt
    frame t-1's scale/boundary instead of frame t's own (mismatched) one,
    fresh-embed only the dirty minority sub-tiles, reuse cached embeddings
    for the quiet majority, tag as continuing the run.

The per-base-tile "did this change vs. the previous frame" signal is RLT's
own existing primitive, reused UNCHANGED (rlt_static_tokens.batched_find_idxs_to_keep,
tubelet_size=1): RLT's base-grid change detector becomes APT-temporal's dirty
signal. Its "always keep the first frame" convention also gives "frame 0 is
always fresh" for free (see classify_cells).

Everything here is fully vectorized across the whole clip -- shape/dirty
comparisons only ever reference frame t vs. frame t-1's own independently
computed partition/pixels (never a carried-forward "adopted" state), so no
Python-level loop over frames is needed for classification, run-length, or
origin-index bookkeeping. (Only the embedding computation in
siglip_apt_temporal_embeddings.py needs to gather variable-shaped groups of
cells per scale; that's a batched-per-scale operation, not a sequential one,
mirroring how SiglipAPTEmbeddings._embed already loops over scales.)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    from .rlt_static_tokens import batched_find_idxs_to_keep
except ImportError:  # allow running this file directly as a script for testing
    from rlt_static_tokens import batched_find_idxs_to_keep

# Per-base-cell classification codes (see module docstring).
REDUNDANT, FRESH, OVERRIDE = 0, 1, 2


def dense_scale_code_grid(masks: Dict[int, torch.Tensor], patch_sizes: List[int],
                           base_patch_size: int) -> torch.Tensor:
    """Per-base-cell scale code (1-based, base scale = 1) for every frame.

    masks: {ps: (T, G_s, G_s)} 0/1 partition masks (APTPatchTokenizer output).
    Returns (T, G, G) long: which scale covers each base cell, per frame.
    Masks form a strict partition (exhaustive + disjoint), so summing the
    upsampled per-scale codes is equivalent to a one-hot "pick" -- exactly the
    upsampling idiom apt_scatter_back already uses.
    """
    T = next(iter(masks.values())).shape[0]
    G = masks[base_patch_size].shape[-1]
    device = masks[base_patch_size].device
    grid = torch.zeros(T, G, G, dtype=torch.long, device=device)
    for idx, ps in enumerate(patch_sizes):
        code = idx + 1
        s = ps // base_patch_size
        up = masks[ps].repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
        grid = grid + up.long() * code
    return grid


def dirty_subtile_mask(frames: torch.Tensor, threshold: float,
                        base_patch_size: int) -> torch.Tensor:
    """Per-base-tile "changed vs. previous frame" mask -- RLT's own primitive,
    reused unchanged, on the SAME pixel scale RLT itself uses.

    frames: (T, 3, H, W) SigLIP-normalized pixels (the same tensor scale
        SiglipRLTEmbeddings.forward feeds straight into this same primitive,
        siglip_rlt_embeddings.py:218-220 -- no un-normalize step). This lets
        `threshold` be the literal config.rlt_threshold value, shared as ONE
        knob between RLT and APT-Temporal instead of two differently-scaled
        thresholds for the same underlying question ("did this patch change
        vs. the previous frame?").
    Returns (T, G, G) bool. Frame 0 is all True (RLT's own "always keep the
    first frame via a dummy 255-valued reference" convention, see
    rlt_static_tokens.batched_find_idxs_to_keep) -- this matches "frame 0
    always needs a fresh embed" for free.
    """
    x5d = frames.permute(1, 0, 2, 3).unsqueeze(0)             # (1,3,T,H,W)
    keep = batched_find_idxs_to_keep(
        x5d, threshold=threshold, tubelet_size=1, patch_size=base_patch_size
    )                                                          # (1, T*G*G) bool, (t,h,w) order
    T = frames.shape[0]
    G = x5d.shape[-1] // base_patch_size
    return keep.view(T, G, G)


def shape_match_grid(scale_grid: torch.Tensor, masks: Dict[int, torch.Tensor],
                      patch_sizes: List[int], base_patch_size: int) -> torch.Tensor:
    """Per-base-cell "frame t's own cell matches frame t-1's partition at the
    same scale/boundary" -- vectorized across all t=1..T-1 via a min/max-pool
    uniformity check. APT's masks are a grid-aligned quadtree partition, so a
    uniform scale code across an aligned footprint IS the same cell (no
    per-cell Python loop, no ambiguity).

    Returns (T, G, G) bool; row 0 (frame 0, no t-1 to compare against) is all
    False -- frame 0 is handled by classify_cells falling through to FRESH.
    """
    T, G, _ = scale_grid.shape
    prev = scale_grid[:-1].float()                            # scale_grid[t-1], aligned to t=1..T-1
    out = torch.zeros(T, G, G, dtype=torch.bool, device=scale_grid.device)
    acc = torch.zeros(T - 1, G, G, dtype=torch.bool, device=scale_grid.device)
    for idx, ps in enumerate(patch_sizes):
        code = idx + 1
        s = ps // base_patch_size
        cur_mask = masks[ps][1:].bool()                       # frame t's own cells at this scale (coarse grid)
        if s == 1:
            match = (prev == code) & cur_mask
        else:
            min_pool = -F.max_pool2d(-prev, kernel_size=s, stride=s)
            max_pool = F.max_pool2d(prev, kernel_size=s, stride=s)
            cell_match = (min_pool == code) & (max_pool == code) & cur_mask   # coarse grid
            match = cell_match.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)  # -> base grid
        acc = acc | match
    out[1:] = acc
    return out


def reference_cell_dirty_stats(dirty: torch.Tensor, scale_grid: torch.Tensor,
                                patch_sizes: List[int], base_patch_size: int,
                                majority_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-base-cell 'all quiet' / 'majority dirty', aggregated over the
    PREVIOUS frame's cell footprint (scale_grid[t-1]) -- not frame t's own.

    This is the candidate footprint actually being tested for continued
    redundancy: whether shape_match holds or not, the question is always "did
    the region frame t-1 treated as one cell change (a little / a lot)
    relative to frame t?". Using frame t's OWN (possibly already-split) cells
    for this aggregation would be wrong: if frame t's own entropy already
    split a footprint into several finer cells, each of those would trivially
    aggregate over itself alone (no vote at all), silently skipping the
    majority-vote override this function exists to compute.

    Returns (all_quiet, majority_dirty): each (T, G, G) bool, row 0 left False
    (no t-1 to compare against; classify_cells forces frame 0 to FRESH
    directly rather than relying on this function's degenerate output there).
    """
    T, G, _ = dirty.shape
    dirty_f = dirty.float()[1:]                                # dirty[t], aligned to t=1..T-1
    prev_code = scale_grid[:-1].float()                         # scale_grid[t-1]
    all_quiet = torch.zeros(T, G, G, dtype=torch.bool, device=dirty.device)
    majority_dirty = torch.zeros_like(all_quiet)
    acc_quiet = torch.zeros(T - 1, G, G, dtype=torch.bool, device=dirty.device)
    acc_majority = torch.zeros_like(acc_quiet)
    for idx, ps in enumerate(patch_sizes):
        code = idx + 1
        s = ps // base_patch_size
        if s == 1:
            frac = dirty_f
            code_mask = prev_code == code
        else:
            frac = F.avg_pool2d(dirty_f, kernel_size=s, stride=s)
            min_pool = -F.max_pool2d(-prev_code, kernel_size=s, stride=s)
            max_pool = F.max_pool2d(prev_code, kernel_size=s, stride=s)
            code_mask_coarse = (min_pool == code) & (max_pool == code)
            frac = frac.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
            code_mask = code_mask_coarse.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
        acc_quiet = acc_quiet | ((frac == 0) & code_mask)
        acc_majority = acc_majority | ((frac > majority_ratio) & code_mask)
    all_quiet[1:] = acc_quiet
    majority_dirty[1:] = acc_majority
    return all_quiet, majority_dirty


def classify_cells(shape_match: torch.Tensor, all_quiet: torch.Tensor,
                    majority_dirty: torch.Tensor) -> torch.Tensor:
    """(T,G,G) long in {REDUNDANT, FRESH, OVERRIDE}. Frame 0 (no t-1 to compare
    against) is forced to FRESH explicitly -- shape_match/all_quiet/majority_dirty
    are all meaningless there (reference_cell_dirty_stats leaves row 0 at its
    zero-initialized default), so this is an explicit special case rather than
    relying on degenerate all-False algebra to happen to fall through to FRESH.
    """
    redundant = shape_match & all_quiet
    fresh = (shape_match & ~all_quiet) | (~shape_match & majority_dirty)
    override = (~shape_match) & (~majority_dirty)
    out = torch.full_like(shape_match, REDUNDANT, dtype=torch.long)
    out = torch.where(fresh, torch.full_like(out, FRESH), out)
    out = torch.where(override, torch.full_like(out, OVERRIDE), out)
    out[0] = FRESH
    return out


def compute_origin_index(event_mask: torch.Tensor) -> torch.Tensor:
    """(T,G,G) bool "is this a new/refresh event here" -> (T,G,G) long: the
    most recent frame index t' <= t (per base cell) where event_mask was True.

    Mirrors llava_arch.py's _rlt_scatter_back cummax/gather pattern. Used both
    for (a) which packed merged-token row supplies (t,i,j)'s dense output
    value [event_mask = is_new_token], and (b) which frame's per-tile
    frozen-encoder embedding a base tile's cached value comes from
    [event_mask = needs_fresh_embed]. Requires event_mask[0] to be all True
    (guaranteed: frame 0 is always FRESH, and FRESH always needs_fresh_embed).
    """
    T = event_mask.shape[0]
    dev = event_mask.device
    t_idx = torch.arange(T, device=dev).view(T, 1, 1).expand_as(event_mask)
    kept_t = torch.where(event_mask, t_idx, torch.full_like(t_idx, -1))
    return torch.cummax(kept_t, dim=0).values


def compute_run_lengths(is_new_token: torch.Tensor) -> torch.Tensor:
    """(T,G,G) bool -> (T,G,G) long: run-length assigned at each new-token
    event (0 at non-event positions). Mirrors
    siglip_rlt_embeddings.get_token_lengths_aligned's cummin/flip trick,
    applied along dim 0 (T) instead of a flattened (t,h,w) index.
    """
    T = is_new_token.shape[0]
    dev = is_new_token.device
    t_idx = torch.arange(T, device=dev).view(T, 1, 1).expand_as(is_new_token)
    big = T
    kept_idx = torch.where(is_new_token, t_idx, torch.full_like(t_idx, big))
    next_incl = torch.flip(torch.cummin(torch.flip(kept_idx, dims=[0]), dim=0).values, dims=[0])
    tail = torch.full((1,) + is_new_token.shape[1:], big, device=dev, dtype=next_incl.dtype)
    next_after = torch.cat([next_incl[1:], tail], dim=0)
    length_map = torch.where(is_new_token, (next_after - t_idx).clamp(min=1), torch.zeros_like(t_idx))
    return length_map


if __name__ == "__main__":
    # ---- self-test: classification + run-length + origin-index consistency ----
    # 4x4 base grid (G=4), 2 scales (14/28) -> a 2x2 grid of 28px coarse cells,
    # each spanning a 2x2 group of base cells. 3-frame clip: f0/f1 fully merged
    # + identical (everything redundant). f2's masks/dirty are constructed BY
    # HAND (rather than fought out of real entropy thresholds, which are
    # finicky to tune blindly) so each quadrant exercises a specific path:
    #   TR quadrant: independently split to base scale, 3/4 sub-tiles dirty -> FRESH
    #   BL quadrant: independently split to base scale, 1/4 sub-tiles dirty -> OVERRIDE
    #   TL, BR quadrants: stay merged, no dirty sub-tiles -> REDUNDANT
    # This is a valid test of these primitives regardless of how masks/dirty
    # were obtained -- they don't care whether entropy or a human produced them.
    torch.manual_seed(0)
    base_p = 14
    patch_sizes = [14, 28]
    G = 4
    T = 3

    masks_14 = torch.zeros(T, G, G)
    masks_28 = torch.zeros(T, G // 2, G // 2)
    masks_28[0] = 1          # f0: fully merged
    masks_28[1] = 1          # f1: fully merged (identical to f0 -> redundant)
    masks_28[2, 0, 0] = 1    # f2 TL quadrant: stays merged
    masks_28[2, 1, 1] = 1    # f2 BR quadrant: stays merged
    masks_14[2, 0:2, 2:4] = 1   # f2 TR quadrant: split to base scale
    masks_14[2, 2:4, 0:2] = 1   # f2 BL quadrant: split to base scale
    masks = {14: masks_14, 28: masks_28}

    scale_grid = dense_scale_code_grid(masks, patch_sizes, base_p)
    print("scale_grid (1=14px,2=28px):\n", scale_grid)

    dirty = torch.zeros(T, G, G, dtype=torch.bool)
    dirty[0] = True                       # RLT convention: frame 0 always "dirty" (always kept)
    # f2 TR quadrant (rows 0-1, cols 2-3): 3/4 sub-tiles dirty -> majority.
    dirty[2, 0, 2] = dirty[2, 0, 3] = dirty[2, 1, 2] = True
    # f2 BL quadrant (rows 2-3, cols 0-1): 1/4 sub-tiles dirty -> minority.
    dirty[2, 2, 0] = True
    print("dirty:\n", dirty)

    shape_match = shape_match_grid(scale_grid, masks, patch_sizes, base_p)
    all_quiet, majority_dirty = reference_cell_dirty_stats(
        dirty, scale_grid, patch_sizes, base_p, majority_ratio=0.5
    )
    cls = classify_cells(shape_match, all_quiet, majority_dirty)
    print("classification (0=REDUNDANT,1=FRESH,2=OVERRIDE):\n", cls)

    assert (cls[0] == FRESH).all(), "frame 0 must always be FRESH"
    assert (cls[1] == REDUNDANT).all(), "f1 (identical to f0) must be fully redundant"
    assert (cls[2, 0:2, 2:4] == FRESH).all(), "TR quadrant (majority dirty) must be FRESH"
    assert (cls[2, 2:4, 0:2] == OVERRIDE).all(), "BL quadrant (minority dirty) must be OVERRIDE"
    assert (cls[2, 0:2, 0:2] == REDUNDANT).all(), "TL quadrant (untouched) must stay REDUNDANT"
    assert (cls[2, 2:4, 2:4] == REDUNDANT).all(), "BR quadrant (untouched) must stay REDUNDANT"

    is_new_token = cls != REDUNDANT
    run_len = compute_run_lengths(is_new_token)
    origin = compute_origin_index(is_new_token)
    print("run_len:\n", run_len)
    print("origin:\n", origin)

    # Consistency: every base cell's origin index must itself be a new-token event.
    for t in range(T):
        for i in range(G):
            for j in range(G):
                o = int(origin[t, i, j])
                assert bool(is_new_token[o, i, j]), "origin must point at a new-token event"
                assert o <= t, "origin must not point into the future"
    print("\nOK: apt_temporal_static_tokens primitives self-consistent "
          "(REDUNDANT/FRESH/OVERRIDE all exercised).")
