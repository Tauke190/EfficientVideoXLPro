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
t-1.

The reuse rule is a single conjunction:

    REUSE a cell iff  (a) frame t's own cell exists at the SAME scale and
                          boundary in frame t-1's partition  [shape_match],
                and   (b) NOTHING inside it changed vs. frame t-1  [all_quiet].

    Otherwise the cell is FRESH: re-embed at frame t's own scale, exactly like
    ordinary per-frame APT.

Both conditions are load-bearing and neither implies the other:

  * (a) alone is not enough -- a cell can keep its shape while its content
    changes completely.
  * (b) alone is not enough -- there is literally nothing to carry forward
    when the shapes disagree. Frame t wants (say) one 2p token; frame t-1
    only ever produced four p tokens covering that region. No token of the
    right shape exists at t-1, so "reuse" is not an option even when the
    pixels are identical.

  * (b) also catches what pure spatial entropy CANNOT see on its own: a region
    can look spatially uniform *within* one frame (low local variance ->
    "mergeable") while having changed completely since the previous frame (a
    flat region's value shifting). The temporal signal overrides the spatial
    merge decision there.

HISTORY -- why there is no third case. An earlier version had an OVERRIDE
class for (shape mismatch + only a MINORITY of sub-tiles dirty): merge at
frame t's own scale, but fresh-embed only the dirty minority and pull the
quiet majority from a cached per-base-tile embedding table. It was removed,
because it bought nothing:

  * OVERRIDE cells still EMIT A TOKEN (they were `cls != REDUNDANT`), so they
    still paid all of SigLIP's transformer blocks as a query. Collapsing
    OVERRIDE into FRESH leaves the token count, the keep rate, the attention
    cost, the run-lengths and the scatter-back all bit-for-bit identical.
  * The only thing it saved was the FROZEN patch conv on a few 14x14 tiles
    (~0.7M MACs) against ~410M MACs of transformer per token -- under 1%.
  * It cost real accuracy and real correctness. FRESH embeds the region from
    frame t's ACTUAL pixels; OVERRIDE stitched in cached embeddings that are
    merely within-threshold of them. And because OVERRIDE deliberately skipped
    the E(Resize_p) pixel anchor, relying on the zero-initialised merge alone,
    an OVERRIDE token at step 0 was EXACTLY a bare position embedding -- no
    visual content at all, silently blanking that region of the frame and
    destroying APT's zero-init identity property.
  * It also required a whole second carry-forward index (the per-base-tile
    embedding cache) and a majority_ratio hyperparameter, both of which are
    now gone.

Note what OVERRIDE never did: it never recovered a lost reuse opportunity. A
cell with (shape mismatch + nothing dirty) -- a region that did not change AT
ALL, whose quadtree boundaries merely wobbled across an entropy threshold --
was an OVERRIDE, i.e. still a full-price token. That case is the real ceiling
on TAPT's savings, and it is a PARTITION-STABILITY problem, not a
classification problem; see classification_stats(), which measures it directly.

The per-base-tile "did this change vs. the previous frame" signal is RLT's own
existing primitive (rlt_static_tokens.batched_find_idxs_to_keep{,_ref},
tubelet_size=1): RLT's base-grid change detector becomes APT-temporal's dirty
signal, sharing the SAME mask_mode/refresh_every knobs and the SAME default
("ref") RLT itself defaults to -- and, via dirty_subtile_mask_embed, the same
mask_space knob, so the dirty check can be run on base-tile embeddings rather
than raw pixels. That choice matters MORE here than it does for RLT alone: the
pixel test's sensitivity is proportional to the local spatial gradient, and
APT's merge criterion is low intensity dispersion, so spatial merging
systematically routes the temporal check into the regime where it is least able
to detect change. See dirty_subtile_mask_embed's docstring for the derivation
and for what the "l2" normalization does and does not fix.

See also batched_find_idxs_to_keep_ref's
docstring: the naive consecutive-frame diff ("consec") tests drift against
frame t-1 while a REDUNDANT cell is actually carried forward from whichever
frame it last changed in, possibly many frames back, so slow per-frame drift
under `threshold` accumulates unboundedly along a long carry chain -- exactly
the failure mode TAPT's own REDUNDANT run-lengths (compute_run_lengths) can
produce. "ref" bounds this by construction, diffing each tile against the
reference it would actually reuse. Its "always keep the first frame"
convention also gives "frame 0 is always fresh" for free (see classify_cells).

Everything here is fully vectorized across the whole clip -- shape/dirty
comparisons only ever reference frame t vs. frame t-1's own independently
computed partition/pixels (never a carried-forward "adopted" state), so no
Python-level loop over frames is needed for classification, run-length, or
origin-index bookkeeping. (Only the embedding computation in
siglip_apt_temporal_embeddings.py needs to gather variable-shaped groups of
cells per scale; that's a batched-per-scale operation, not a sequential one,
mirroring how SiglipAPTEmbeddings._embed already loops over scales.)
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from .rlt_static_tokens import (
        batched_find_idxs_to_keep, batched_find_idxs_to_keep_ref, find_idxs_to_keep_embed,
    )
except ImportError:  # allow running this file directly as a script for testing
    from rlt_static_tokens import (
        batched_find_idxs_to_keep, batched_find_idxs_to_keep_ref, find_idxs_to_keep_embed,
    )

# Per-base-cell classification codes (see module docstring).
REDUNDANT, FRESH = 0, 1


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
                        base_patch_size: int, mask_mode: str = "ref",
                        refresh_every: int = 0) -> torch.Tensor:
    """Per-base-tile "changed vs. previous frame" mask -- RLT's own primitive,
    on the SAME pixel scale RLT itself uses.

    frames: (T, 3, H, W) SigLIP-normalized pixels (the same tensor scale
        SiglipRLTEmbeddings.forward feeds straight into this same primitive,
        siglip_rlt_embeddings.py:218-220 -- no un-normalize step). This lets
        `threshold` be the literal config.rlt_threshold value, shared as ONE
        knob between RLT and APT-Temporal instead of two differently-scaled
        thresholds for the same underlying question ("did this patch change
        vs. the previous frame?").
    mask_mode: "ref" (default, matches SiglipRLTEmbeddings' own default) diffs
        each tile against the reference it would actually be carried forward
        from, bounding drift by `threshold` regardless of run length; "consec"
        is the legacy paper behaviour (diff against frame t-1 only), which lets
        slow per-frame drift accumulate unboundedly across a long REDUNDANT
        run -- see batched_find_idxs_to_keep_ref's docstring and the module
        docstring above. `refresh_every` (ref mode only) force-refreshes every
        Nth frame regardless of drift; 0 disables.
    Returns (T, G, G) bool. Frame 0 is all True (RLT's own "always keep the
    first frame via a dummy reference" convention, see
    rlt_static_tokens.batched_find_idxs_to_keep{,_ref}) -- this matches "frame 0
    always needs a fresh embed" for free.
    """
    assert mask_mode in ("ref", "consec"), f"unknown mask_mode {mask_mode!r}"
    x5d = frames.permute(1, 0, 2, 3).unsqueeze(0)             # (1,3,T,H,W)
    if mask_mode == "ref":
        keep = batched_find_idxs_to_keep_ref(
            x5d, threshold=threshold, patch_size=base_patch_size,
            refresh_every=refresh_every,
        )                                                      # (1, T*G*G) bool, (t,h,w) order
    else:
        keep = batched_find_idxs_to_keep(
            x5d, threshold=threshold, tubelet_size=1, patch_size=base_patch_size
        )                                                      # (1, T*G*G) bool, (t,h,w) order
    T = frames.shape[0]
    G = x5d.shape[-1] // base_patch_size
    return keep.view(T, G, G)


@torch.no_grad()
def dirty_subtile_mask_embed(tile_embed: torch.Tensor, threshold: float,
                              metric: str = "l2", mask_mode: str = "ref",
                              refresh_every: int = 0) -> torch.Tensor:
    """Per-base-tile "changed vs. the previous frame" mask computed on base-tile
    EMBEDDINGS instead of raw pixels -- the APT-Temporal counterpart of RLT's
    mask_space="embed" (rlt_static_tokens.find_idxs_to_keep_embed), sharing the
    same knobs and the same "ref" default.

    WHY THIS MATTERS MORE UNDER COMPOSITION THAN IT DOES FOR RLT ALONE. The
    pixel test's sensitivity is not uniform across the frame -- it is
    proportional to the local spatial gradient. Under brightness constancy
    (I_t = -grad(I) . v, the assumption every motion-estimation method rests
    on) the per-tile statistic dirty_subtile_mask thresholds obeys

        mean|I_t| = mean|grad(I) . v| <= |v| * mean|grad(I)|

    so a region with little spatial detail registers little intensity change
    NO MATTER HOW MUCH MOVED THROUGH IT. This is the aperture problem at its
    limit: a perfectly flat region cannot express its own motion in intensity
    differences at all.

    That is a mild nuisance for RLT alone, where the test runs on a uniform
    base grid. It is a systematic bias under APT-Temporal, because APT's merge
    criterion IS low intensity dispersion (compute_patch_entropy_batched
    thresholds the Shannon entropy of a cell's grayscale histogram). So APT
    sorts cells by exactly the quantity that governs the pixel test's SNR and
    hands the bottom of that ordering to the dirty check: P(false REDUNDANT |
    merged) is structurally higher than P(false REDUNDANT | base scale), and
    the two criteria stop being independent -- APT's drop set is close to a
    SUBSET of the pixel test's, so composing them yields far less than the
    product of their individual savings.

    Testing embeddings does not repeal the physics, but it moves the decision
    onto the quantity that is actually reused (a REDUNDANT cell carries an
    embedding forward, not pixels -- see find_idxs_to_keep_embed's docstring),
    and the frozen patch conv responds to structure the raw intensity
    difference cannot separate from grain.

    HONEST LIMITATION, so the ablation is read correctly: metric="l2"
    normalizes by the CLIP's MEAN patch-embed norm -- one global scalar. That
    makes the threshold dimensionless and transferable, but it is still an
    ABSOLUTE test, so it does NOT undo the gradient attenuation above: a flat
    tile's small change stays small relative to the clip average. metric=
    "cosine" normalizes per tile and therefore does push back on it directly,
    which is very likely part of what find_idxs_to_keep_embed's docstring
    records as cosine "over-keeping" flat low-norm regions such as sky. That
    over-keeping is not purely a defect -- it is a blunt correction for a test
    that genuinely cannot see through flat regions -- but it IS blunt, since it
    also over-keeps flat regions that really are static, and pays real savings
    for it. A per-cell relative test against the merged token each cell is
    carried forward from is the sharper instrument, and needs merged-token
    identity to be stable across frames first; not implemented here.

    Args:
        tile_embed: (T, G, G, C) frozen patch-embed output for EVERY base tile,
            with NO position embedding added. APT-Temporal builds this itself
            rather than subtracting a position embedding back off (the way
            SiglipRLTEmbeddings._embed_keep_mask must), so the "position is
            constant per slot and must not enter the metric's norms"
            requirement in find_idxs_to_keep_embed holds by construction.
        threshold: embedding-distance drop threshold. NOT on the pixel scale of
            dirty_subtile_mask's threshold -- see find_idxs_to_keep_embed.
        metric: "l2" (default) or "cosine".
        mask_mode, refresh_every: as dirty_subtile_mask.

    Returns (T, G, G) bool, same layout/polarity as dirty_subtile_mask (True =
    changed). Frame 0 is all True, matching that function's convention.
    """
    assert tile_embed.dim() == 4, f"tile_embed must be (T,G,G,C), got {tuple(tile_embed.shape)}"
    T, G, _, C = tile_embed.shape
    keep = find_idxs_to_keep_embed(
        tile_embed.reshape(T, G * G, C),
        threshold=threshold, metric=metric,
        mask_mode=mask_mode, refresh_every=refresh_every,
    )                                          # (T*G*G,) bool, (t,h,w) order
    return keep.view(T, G, G)


@torch.no_grad()
def detect_cuts(dirty: torch.Tensor, cut_threshold: float = 0.8) -> torch.Tensor:
    """(T,G,G) dirty mask -> (T,) bool: frames where essentially EVERYTHING changed.

    No scene-detection machinery: at a shot boundary every base tile trips the
    dirty test at once, so the mask RLT already computes is the detector. It is
    also available before the partition exists (dirty_subtile_mask{,_embed} work
    at base scale and never reference masks), which is what makes cut-aligned
    windowing possible in the first place -- see window_ids.

    NOTE what this deliberately does NOT try to be: a semantically correct shot
    detector. A fast pan, a whip, a flash also trip every tile, and those are
    ALSO frames where nothing was reusable. The only thing this has to find is
    "frames where reuse was going to fail anyway", which is a far easier target
    than shot-boundary detection and degrades gracefully -- a false positive
    costs nothing (see window_ids), and a false negative costs at most one
    window's worth of over-conservative partition.

    cut_threshold: fraction of dirty base tiles above which the frame is a
        boundary. The per-frame dirty fraction on edited video is strongly
        bimodal (within-shot vs. cut), so read the value off a histogram of
        your own footage rather than trusting this default.
    """
    return dirty.flatten(1).float().mean(dim=1) > cut_threshold


@torch.no_grad()
def window_ids(cut: torch.Tensor, max_window: int) -> torch.Tensor:
    """(T,) bool cut mask -> (T,) long window id, ids consecutive from 0.

    A new window starts at a cut, or once the current one hits `max_window`
    frames. max_window=1 puts every frame in its own window, which makes the
    windowed partition degenerate to the per-frame partition exactly -- that is
    the default, so windowing is inert until switched on.

    WHY BOUNDARIES AT CUTS ARE FREE. A window boundary costs the first frame of
    the window: with no previous partition to match, shape_match fails and every
    cell is FRESH. At a cut, every cell is dirty, so every cell is FRESH anyway
    -- the boundary's cost and the cut's cost are the same cost, paid once. The
    window structure and the video's own structure want the boundary in the same
    place.

    WHY max_window EXISTS. Within one long shot the content still drifts, and
    the windowed partition takes a max over the whole window (see
    window_max_entropy), so a very long window is very conservative -- one busy
    stretch forces a region fine-grained for all of it. This bounds that.
    Distinct from mask_mode="ref", which bounds how far a carried tile's PIXELS
    drift; this bounds how stale the PARTITION is allowed to get.
    """
    assert max_window >= 1, f"max_window must be >= 1, got {max_window}"
    T = int(cut.shape[0])
    is_cut = cut.tolist()                       # one sync; T is ~10^2, the loop is int work
    ids, cur, run = [], 0, 0
    for t in range(T):
        if t > 0 and (is_cut[t] or run >= max_window):
            cur += 1
            run = 0
        ids.append(cur)
        run += 1
    return torch.tensor(ids, dtype=torch.long, device=cut.device)


@torch.no_grad()
def window_max_entropy(importance_maps: Dict[int, torch.Tensor],
                        seg_id: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Per-scale entropy maps -> the same maps with each window's MAX broadcast
    over every frame in that window, so select_patches_by_threshold produces ONE
    partition per window instead of one per frame.

    MAX, not mean, and the direction matters: a region merges only if EVERY
    frame in the window agreed it was mergeable. Any frame that wanted fine
    detail there keeps it fine for the whole window. That is conservative in the
    safe direction -- the failure mode is emitting more tokens than a per-frame
    partition would, never blurring away detail some frame needed.

    It also keeps the border convention intact: compute_patch_entropy_batched
    marks incomplete border patches with pad_value=1e6 so they never merge, and
    a max preserves that.

    Args:
        importance_maps: {ps: (T, G_s, G_s)} from
            APTPatchTokenizer.compute_importance_maps.
        seg_id: (T,) long window id from window_ids.

    Returns {ps: (T, G_s, G_s)}, same shapes -- so this is a drop-in before
    select_patches_by_threshold and every downstream shape is unchanged.

    Implemented as a slice-per-window amax rather than a segmented scatter
    reduce: window_ids only ever produces CONTIGUOUS runs, so slicing is exact,
    and it avoids torch's index_reduce/scatter_reduce beta-API warning firing on
    every forward. The loop is over windows (few), not frames.
    """
    ids = seg_id.tolist()
    bounds, start = [], 0
    for t in range(1, len(ids) + 1):
        if t == len(ids) or ids[t] != ids[start]:
            bounds.append((start, t))
            start = t

    out = {}
    for ps, e in importance_maps.items():
        agg = torch.empty_like(e)
        for lo, hi in bounds:
            agg[lo:hi] = e[lo:hi].amax(dim=0, keepdim=True)
        out[ps] = agg
    return out


@torch.no_grad()
def survivor_aligned_masks(dirty: torch.Tensor, run_len: torch.Tensor,
                            patch_sizes: List[int], base_patch_size: int,
                            run_tol: int = 0, persist: bool = False) -> Dict[int, torch.Tensor]:
    """Grid-aligned partition built from RLT SURVIVORSHIP instead of entropy.

    Drop-in replacement for apt_static_tokens.select_patches_by_threshold: same
    {ps: (T, G_s, G_s)} strict-partition contract, so every consumer downstream
    (dense_scale_code_grid, shape_match_grid, cell_all_quiet, classify_cells,
    the per-scale merge pass, apt_temporal_scatter_back) is unchanged.

    WHY THIS ORDERING EXISTS. The entropy partition merges on spatial flatness,
    and flat regions are disproportionately the ones RLT was going to drop
    anyway -- so the two criteria compete for the same redundancy, and APT's
    merge lands on cells that would have cost nothing. Measured on MLVU: coarse
    (flat) cells are temporally quiet 51.6% of the time against 38.2% for fine
    cells, and a REDUNDANT cell emits zero tokens whatever its scale, so merging
    a still region buys nothing and costs fidelity.

    Building the partition from survivorship inverts that. Only tiles RLT could
    NOT drop are eligible to merge, so the spatial saving is additive on top of
    the temporal one by construction rather than overlapping with it.

    THE RULE. A coarse cell at scale s exists iff

        (a) all s*s base tiles under it SURVIVED (dirty)      [nothing to reuse]
        (b) their run lengths agree within `run_tol`          [same lifetime]

    (b) is load-bearing and is not implied by (a). Merging a tile that lives one
    frame with one that lives ten forces the merged token to refresh every
    frame: the shortest lifetime in the group dominates.

    What that costs is FIDELITY, not tokens -- and the distinction matters,
    because the token effect runs the OTHER way. One merged token that refreshes
    every frame still beats four separate ones, so loosening run_tol reduces the
    token count (measured, 10 MLVU videos x 128 frames, persist=False):

        run_tol=0    278 tokens/frame   missed_reuse 0.042
        run_tol=1    256                             0.077
        run_tol=4    243                             0.102
        run_tol=999  241                             0.110

    The price of a loose tolerance is that the long-lived tiles -- which could
    have been carried at full resolution for free -- get dragged into a cell that
    is re-encoded every frame at reduced resolution, and that more aggressive
    merging churns the partition, so missed_reuse climbs. Requiring agreement
    also keeps a merged token's own run length unambiguous (children that
    disagree have no single correct l to carry).

    THE MOTION-STOPS COST, and what `persist` does about it. With persist=False
    a tile that was DIRTY at t-1 (hence merged, possibly inside a coarse cell)
    and goes QUIET at t is a base-scale cell at t, so shape_match against t-1's
    coarse cell fails and EVERY tile under that cell pays a fresh re-encode. For
    a 4-tile block that merges at t and then stays still for 5 frames that is
    1 + 4 = 5 tokens, against 4 for never merging at all and 1 with persistence
    -- i.e. merging something that is about to go still actively costs more than
    leaving it alone. persist=True keeps the cell alive while its contents stay
    quiet, so the merged token is carried for its whole run.

    Measured on 10 MLVU videos x 128 frames (survivor mode, run_tol=0):

        persist=False   260 tokens/frame   66.8% dropped   3.06x   missed 0.042
        persist=True    228 tokens/frame   71.0% dropped   4.09x   missed 0.000

    (last two columns: mean compression ratio of the token covering a base cell,
    and classification_stats' missed_reuse.) Persistence drives missed_reuse to
    exactly zero -- the fragmentation was its entire source in this mode -- and
    it is a TRADE: the region stays at merged resolution through the still
    stretch instead of being refreshed at full resolution when it fragments.
    Both settings still represent content more finely than the entropy partition
    does (6.12x on the same clips), while emitting far fewer tokens.

    Args:
        dirty: (T, G, G) bool, True = tile changed = SURVIVED RLT. Straight from
            dirty_subtile_mask{,_embed}; frame 0 is all True by their convention.
        run_len: (T, G, G) long from compute_run_lengths(dirty) -- forward run
            length at each surviving tile, 0 where the tile did not survive.
        patch_sizes: ascending, base first (apt.patch_sizes).
        base_patch_size: the base scale.
        run_tol: max allowed spread (max - min) of run length inside a cell.
            0 = exact agreement.

    Returns {ps: (T, G_s, G_s)} float 0/1 masks forming a strict partition.
    """
    d, r = dirty.float(), run_len.float()
    coarse_sizes = patch_sizes[1:]

    # Candidate coarse cells, per scale, independently: a FRESH merge.
    cand: Dict[int, torch.Tensor] = {}
    for ps in coarse_sizes:
        s = ps // base_patch_size
        all_survive = F.avg_pool2d(d, kernel_size=s, stride=s) == 1.0        # (a)
        hi = F.max_pool2d(r, kernel_size=s, stride=s)
        lo = -F.max_pool2d(-r, kernel_size=s, stride=s)
        cand[ps] = all_survive & ((hi - lo) <= float(run_tol))                # (b)

    def _coarser_wins(cur: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Resolve overlap among coarse scales; the coarsest claims first. Same
        idiom and ordering as select_patches_by_threshold, so the result is
        disjoint exactly as that function's is."""
        for i in range(len(coarse_sizes) - 1, 0, -1):
            c = coarse_sizes[i]
            for j in range(i):
                sm = coarse_sizes[j]
                f = c // sm
                up = cur[c].repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)
                cur[sm] = cur[sm] & ~up[..., :cur[sm].shape[-2], :cur[sm].shape[-1]]
        return cur

    if persist:
        # A coarse cell stays alive while everything inside it stays quiet, so a
        # merged token can be CARRIED for the length of its run instead of
        # fragmenting the moment its tiles stop changing. Without this, condition
        # (a) fails as soon as the motion stops, the cell splits back into base
        # cells, shape_match against the previous frame's coarse cell fails, and
        # every tile inside pays a fresh re-encode -- which costs MORE than never
        # having merged (measured: 5 tokens vs 4 for a 4-tile block that merges
        # then goes still for 5 frames; with persistence it is 1).
        #
        # Sequential over frames because frame t's partition depends on t-1's
        # RESOLVED partition, not on its candidates -- a cell only persists if it
        # actually existed after _coarser_wins. T is ~10^2 and the body is a
        # handful of elementwise ops, the same shape of loop
        # batched_find_idxs_to_keep_ref already runs.
        #
        # This is a TRADE, not a free win: the region stays at merged resolution
        # for the whole still stretch instead of being refreshed at full
        # resolution when it fragments. Fewer tokens, coarser representation.
        quiet = {ps: F.avg_pool2d((~dirty).float(), kernel_size=ps // base_patch_size,
                                   stride=ps // base_patch_size) == 1.0
                 for ps in coarse_sizes}
        out = {ps: torch.zeros_like(cand[ps]) for ps in coarse_sizes}
        prev = {ps: torch.zeros_like(cand[ps][0]) for ps in coarse_sizes}
        for t in range(dirty.shape[0]):
            cur = _coarser_wins({ps: cand[ps][t] | (prev[ps] & quiet[ps][t])
                                 for ps in coarse_sizes})
            for ps in coarse_sizes:
                out[ps][t] = cur[ps]
                prev[ps] = cur[ps]
        coarse = out
    else:
        coarse = _coarser_wins({ps: cand[ps].clone() for ps in coarse_sizes})

    masks: Dict[int, torch.Tensor] = {ps: coarse[ps].float() for ps in coarse_sizes}
    # Base scale takes whatever no coarse cell claimed -> exhaustive by construction.
    base = torch.ones_like(dirty, dtype=torch.float32)
    for ps in coarse_sizes:
        s = ps // base_patch_size
        up = masks[ps].repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
        base = base * (1 - up[:, :base.shape[1], :base.shape[2]])
    masks[patch_sizes[0]] = base
    return masks


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


def cell_all_quiet(dirty: torch.Tensor, masks: Dict[int, torch.Tensor],
                    patch_sizes: List[int], base_patch_size: int) -> torch.Tensor:
    """Per-base-cell "nothing inside this cell changed", aggregated over FRAME
    T'S OWN cell footprint (masks[ps][t]) -- deliberately NOT the previous
    frame's footprint.

    An earlier version aggregated over scale_grid[t-1]'s footprint instead,
    reasoning that "the candidate being tested for redundancy is whatever t-1
    treated as one cell." That's wrong: when frame t's own entropy decides to
    MERGE a region that frame t-1 had split into multiple, differently-scaled
    sub-cells, aggregating per t-1's (heterogeneous) structure produces a
    DIFFERENT classification for different sub-tiles within what is otherwise
    ONE of frame t's own cells -- which breaks the invariant the
    embedding/merge pass depends on (every base cell in one of frame t's own
    cells must resolve to the same decision, since they all become ONE token
    together). Confirmed on real video: this produced base cells with no valid
    packed row at all.

    Aggregating over frame t's own footprint instead guarantees uniformity by
    construction, since shape_match and all_quiet are then BOTH keyed to the
    exact same masks[ps][t] restriction.

    Returns (T, G, G) bool, broadcast uniformly over frame t's own cell
    footprint at whichever scale it is.
    """
    T, G, _ = dirty.shape
    dirty_f = dirty.float()
    all_quiet = torch.zeros(T, G, G, dtype=torch.bool, device=dirty.device)
    for ps in patch_sizes:
        s = ps // base_patch_size
        cur_mask = masks[ps].bool()                            # coarse grid (G/s, G/s)
        if s == 1:
            frac = dirty_f
            mask_base = cur_mask
        else:
            frac = F.avg_pool2d(dirty_f, kernel_size=s, stride=s)                # coarse grid
            frac = frac.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)   # -> base grid
            mask_base = cur_mask.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)  # -> base grid
        all_quiet = all_quiet | ((frac == 0) & mask_base)
    return all_quiet


def classify_cells(shape_match: torch.Tensor, all_quiet: torch.Tensor) -> torch.Tensor:
    """(T,G,G) long in {REDUNDANT, FRESH}.

    REDUNDANT (reuse the carried token, emit nothing) iff the cell kept its
    shape AND nothing inside it changed; FRESH (re-embed at frame t's own
    scale) otherwise. See the module docstring for why both conditions are
    required and why there is no third class.

    Frame 0 (no t-1 to compare against) is forced to FRESH explicitly --
    shape_match[0] is meaningless there (shape_match_grid leaves row 0 at its
    zero-initialized default, since there's no t-1), so this is an explicit
    special case rather than relying on degenerate algebra to happen to fall
    through to FRESH.
    """
    out = torch.where(
        shape_match & all_quiet,
        torch.full_like(shape_match, REDUNDANT, dtype=torch.long),
        torch.full_like(shape_match, FRESH, dtype=torch.long),
    )
    out[0] = FRESH
    return out


def classification_stats(shape_match: torch.Tensor, all_quiet: torch.Tensor) -> Dict[str, float]:
    """Diagnostics over frames 1..T-1 (frame 0 is unconditionally FRESH and
    would only dilute the rates).

    The number that matters is `missed_reuse`: base cells where NOTHING changed
    (all_quiet) but the quadtree boundaries disagree with the previous frame
    (not shape_match), so the cell cannot be carried forward and pays for a
    full token anyway. These are pure loss -- a region of video that did not
    change, re-encoded from scratch because an entropy value wobbled across a
    threshold between frames.

    A high missed_reuse says the ceiling on TAPT's savings is set by PARTITION
    INSTABILITY, not by how much the video actually moves, and points at
    threshold hysteresis / windowed re-partitioning rather than at anything in
    the classifier.
    """
    if shape_match.shape[0] < 2:
        return {}
    sm, aq = shape_match[1:], all_quiet[1:]
    n = float(sm.numel())
    return {
        "shape_match": float(sm.sum()) / n,        # partition agreed with t-1
        "all_quiet": float(aq.sum()) / n,          # nothing in the cell changed
        "redundant": float((sm & aq).sum()) / n,   # both -> token reused (the actual saving)
        "missed_reuse": float((~sm & aq).sum()) / n,  # quiet but unreusable -> partition instability
    }


def compute_origin_index(event_mask: torch.Tensor) -> torch.Tensor:
    """(T,G,G) bool "is this a new/refresh event here" -> (T,G,G) long: the
    most recent frame index t' <= t (per base cell) where event_mask was True.

    Mirrors the RLT carry-forward cummax/gather pattern (siglip_rlt_embeddings._carry_idx).
    Used to resolve which packed merged-token row supplies (t,i,j)'s dense output
    value [event_mask = is_new_token]. Requires event_mask[0] to be all True
    (guaranteed: frame 0 is always FRESH).
    """
    T = event_mask.shape[0]
    dev = event_mask.device
    t_idx = torch.arange(T, device=dev).view(T, 1, 1).expand_as(event_mask)
    kept_t = torch.where(event_mask, t_idx, torch.full_like(t_idx, -1))
    return torch.cummax(kept_t, dim=0).values


def compute_run_lengths(is_new_token: torch.Tensor) -> torch.Tensor:
    """(T,G,G) bool -> (T,G,G) long: run-length assigned at each new-token
    event (0 at non-event positions). Uses a reverse-cummin/flip trick along
    dim 0 (T): the run-length of a new-token event at frame t is the distance to
    the next new-token event after t.
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
    # each spanning a 2x2 group of base cells. 3-frame clip: f0/f1 IDENTICAL
    # (TL/BR merged to 28px, TR/BL split to base -- everything redundant). f2's
    # masks/dirty are constructed BY HAND (rather than fought out of real
    # entropy thresholds, which are finicky to tune blindly) so each quadrant
    # exercises a specific path:
    #   TR quadrant: f2 MERGES to 28px (f1 had it split to base -> mismatch),
    #                3/4 constituent tiles dirty          -> FRESH (content changed)
    #   BL quadrant: f2 MERGES to 28px (f1 had it split to base -> mismatch),
    #                ZERO tiles dirty                     -> FRESH, and this is the
    #                MISSED REUSE case: nothing changed, but there is no token of
    #                the right shape at f1 to carry forward. Pure partition-jitter
    #                loss -- exactly what classification_stats().missed_reuse counts.
    #   TL, BR quadrants: stay merged throughout, no dirty sub-tiles -> REDUNDANT
    # This is a valid test of these primitives regardless of how masks/dirty
    # were obtained -- they don't care whether entropy or a human produced them.
    torch.manual_seed(0)
    base_p = 14
    patch_sizes = [14, 28]
    G = 4
    T = 3

    masks_14 = torch.zeros(T, G, G)
    masks_28 = torch.zeros(T, G // 2, G // 2)
    for t in (0, 1):
        masks_28[t, 0, 0] = 1        # TL: merged
        masks_28[t, 1, 1] = 1        # BR: merged
        masks_14[t, 0:2, 2:4] = 1    # TR: split to base (f0 == f1)
        masks_14[t, 2:4, 0:2] = 1    # BL: split to base (f0 == f1)
    masks_28[2, 0, 0] = 1    # f2 TL: stays merged
    masks_28[2, 1, 1] = 1    # f2 BR: stays merged
    masks_28[2, 0, 1] = 1    # f2 TR: NOW MERGES to 28px (mismatch vs f1's split)
    masks_28[2, 1, 0] = 1    # f2 BL: NOW MERGES to 28px (mismatch vs f1's split)
    masks = {14: masks_14, 28: masks_28}

    scale_grid = dense_scale_code_grid(masks, patch_sizes, base_p)
    print("scale_grid (1=14px,2=28px):\n", scale_grid)

    dirty = torch.zeros(T, G, G, dtype=torch.bool)
    dirty[0] = True                       # RLT convention: frame 0 always "dirty" (always kept)
    # f2 TR quadrant (rows 0-1, cols 2-3): 3/4 sub-tiles dirty -> content changed.
    dirty[2, 0, 2] = dirty[2, 0, 3] = dirty[2, 1, 2] = True
    # f2 BL quadrant (rows 2-3, cols 0-1): NOTHING dirty -> the missed-reuse case.
    print("dirty:\n", dirty)

    shape_match = shape_match_grid(scale_grid, masks, patch_sizes, base_p)
    all_quiet = cell_all_quiet(dirty, masks, patch_sizes, base_p)
    cls = classify_cells(shape_match, all_quiet)
    print("classification (0=REDUNDANT,1=FRESH):\n", cls)

    assert (cls[0] == FRESH).all(), "frame 0 must always be FRESH"
    assert (cls[1] == REDUNDANT).all(), "f1 (identical to f0) must be fully redundant"
    assert (cls[2, 0:2, 2:4] == FRESH).all(), "TR quadrant (content changed) must be FRESH"
    assert (cls[2, 2:4, 0:2] == FRESH).all(), (
        "BL quadrant: quiet but shape-mismatched -> FRESH (nothing of the right shape to carry)"
    )
    assert (cls[2, 0:2, 0:2] == REDUNDANT).all(), "TL quadrant (untouched) must stay REDUNDANT"
    assert (cls[2, 2:4, 2:4] == REDUNDANT).all(), "BR quadrant (untouched) must stay REDUNDANT"

    stats = classification_stats(shape_match, all_quiet)
    print("stats (frames 1..T-1):", {k: round(v, 3) for k, v in stats.items()})
    # BL quadrant of f2 = 4 of the 32 base cells over frames 1..2: quiet, but not reusable.
    assert abs(stats["missed_reuse"] - 4 / 32) < 1e-6, (
        f"BL quadrant of f2 must be counted as missed reuse, got {stats['missed_reuse']}"
    )

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
          "(REDUNDANT/FRESH + missed-reuse accounting).")
