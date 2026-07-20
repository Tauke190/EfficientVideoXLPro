"""
siglip_apt_temporal_embeddings.py
==================================

Temporal-Anchored APT (TAPT): combines APT's per-frame spatial partition with
RLT-style temporal redundancy collapsing. See apt_temporal_static_tokens.py
for the classification math and its docstring for the precedence rationale
(spatial first, temporal second) and the REDUNDANT/FRESH rule.

The reuse rule, in one line: a cell is REDUNDANT (carried forward, emitting no
token) iff its shape matches frame t-1's partition AND nothing inside it
changed. Otherwise it is FRESH and re-embedded at frame t's own scale, exactly
as ordinary per-frame APT would.

This module **composes** an already-built SiglipAPTEmbeddings instance --
no changes to siglip_apt_embeddings.py are needed. It reuses:
  * apt.tokenizer          (APTPatchTokenizer -- entropy/partition + patch
                             grouping, including construct_patch_groups'
                             existing support for applying an externally
                             supplied mask to a frame's pixels, used here to
                             compute the E(Resize_p) "coarse anchor" term for
                             just the subset of frame t's own cells that emit a
                             token this frame).
  * apt._embed_patches      (frozen SigLIP patch_embedding Conv2d -- never
                             shown anything but a native 14x14 patch).
  * apt.patch_attn / apt.zero_conv  (APT's trainable, zero-init spatial merge,
                             used verbatim -- every coarse token here is APT's
                             Eq. 2 unchanged, so with zero_conv zero-init a
                             coarse token starts at exactly E(Resize_p) + pos
                             and TAPT is an identity perturbation of the
                             pretrained model at step 0).
  * apt._run_encoder        (xformers block-diagonal attention over the
                             frozen SigLIP transformer layers).
  * apt.base_pos_embed / apt.base_grid_size (position embedding resampling).

No trainable parameter of its own: like plain RLT (siglip_rlt_embeddings.py),
TAPT adds no learnable state at the temporal seam. The only trained modules are
APT's spatial patch_attn/zero_conv, so TAPT is exactly "APT's merge + RLT's
reuse" and its trainable surface is identical to an APT-only run. (A zero-init
run-length embedding tagging each survivor with how many frames it stood in for
used to live here; it was removed -- RLT drops the paper's equivalent for
contributing little, and keeping it made TAPT's trained modules differ from
APT's, confounding the comparison between them.)

Everything through classification/run-length/origin-index is fully
vectorized (see apt_temporal_static_tokens.py). The only per-scale looping
here mirrors what SiglipAPTEmbeddings._embed already does (batch all events
of a given scale together), it is not a per-frame Python loop.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from timm.layers import resample_abs_pos_embed

try:
    from .apt_static_tokens import select_patches_by_threshold
    from .siglip_apt_embeddings import SiglipAPTEmbeddings
    from .apt_temporal_static_tokens import (
        REDUNDANT, FRESH,
        dense_scale_code_grid, dirty_subtile_mask, dirty_subtile_mask_embed,
        shape_match_grid, cell_all_quiet, classify_cells, classification_stats,
        compute_origin_index, compute_run_lengths, detect_cuts, window_ids,
        window_max_entropy, survivor_aligned_masks,
    )
except ImportError:  # allow running this file directly as a script for testing
    from apt_static_tokens import select_patches_by_threshold
    from siglip_apt_embeddings import SiglipAPTEmbeddings
    from apt_temporal_static_tokens import (
        REDUNDANT, FRESH,
        dense_scale_code_grid, dirty_subtile_mask, dirty_subtile_mask_embed,
        shape_match_grid, cell_all_quiet, classify_cells, classification_stats,
        compute_origin_index, compute_run_lengths, detect_cuts, window_ids,
        window_max_entropy, survivor_aligned_masks,
    )


def apt_temporal_scatter_back(combined_survivors: torch.Tensor, origin_index: torch.Tensor,
                               T: int, P: int) -> torch.Tensor:
    """Broadcast TAPT's packed, encoded survivors back to a dense (T, P, C) grid.

    origin_index: (T, G, G) long, already fully resolved by
    SiglipAPTTemporalEmbeddings.forward -- for every (t, base cell), which row
    of combined_survivors currently supplies its value (REDUNDANT cells point
    at whichever FRESH event most recently refreshed that spatial position;
    FRESH cells point at their own row).
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
        mask_mode, refresh_every: same knobs and same "ref" default as
            SiglipRLTEmbeddings (siglip_rlt_embeddings.py), reused directly at
            the llava_arch.py wiring level from config.rlt_mask_mode /
            config.rlt_refresh_every -- see dirty_subtile_mask's docstring for
            why "ref" (diff against the carried reference, not frame t-1) is
            the correct default for a REDUNDANT run of any length.
        mask_space: WHERE the dirty check compares base tiles -- "pixel"
            (default, the paper's test, uses `threshold`) or "embed" (uses
            `embed_threshold`/`embed_metric`, comparing frozen patch-embed
            outputs). Again the SAME knob RLT uses (config.rlt_mask_space),
            for the same reason the threshold is shared: both are answering the
            identical question. It carries more weight here than it does for
            RLT alone -- the pixel test's sensitivity scales with the local
            spatial gradient, and APT merges on low intensity dispersion, so
            spatial merging systematically feeds the temporal check the tiles
            it is least able to read. See dirty_subtile_mask_embed's docstring.
        embed_threshold, embed_metric: shared with RLT
            (config.rlt_embed_threshold / config.rlt_embed_metric); used only
            when mask_space="embed". NOT interchangeable with `threshold`,
            which is a pixel-scale value.
        window: how many frames share ONE partition. 1 (default) = today's
            behaviour, a fresh independent partition per frame, and the
            windowing code is then exactly inert. >1 computes one partition per
            window from the window's MAX entropy, which makes shape_match true
            by construction inside a window and so converts `missed_reuse`
            (cells that did not change but could not be carried because the
            quadtree boundaries wobbled) into actual reuse. The cost is a
            partition that is window-optimal rather than frame-optimal, i.e.
            slightly more tokens per frame, growing with `window`. Windows also
            break at detected cuts, so this is a MAXIMUM length, not a fixed
            stride -- see window_ids.
        cut_threshold: fraction of dirty base tiles above which a frame starts a
            new window. Only consulted when window > 1. See detect_cuts for why
            a boundary at a cut is free, and why this does not need to be a
            semantically correct shot detector.
        partition_mode: "entropy" (default, plain APT: per-frame quadtree from
            spatial flatness) or "survivor" (RLT-first: merge only what RLT
            could not drop, so spatial saving is additive on top of temporal
            reuse rather than competing with it). See survivor_aligned_masks.
        run_tol: survivor mode only -- max spread of forward run length allowed
            inside one merged cell. 0 = exact agreement.
        persist: survivor mode only -- keep a coarse cell alive while everything
            inside it stays quiet, so a merged token is carried for its whole
            run instead of fragmenting the moment its tiles stop changing.
            Without it, merging a block that is about to go still costs MORE
            than never merging it (the split forces a fresh re-encode of every
            tile inside). Drives missed_reuse to ~0. It is a trade, not a free
            win: the region then stays at merged resolution through the still
            stretch rather than refreshing at full resolution. Default off so
            the un-persisted result stays reproducible.

    `self.last_windows` holds the number of windows the last clip was split
    into (== T when window=1), so a profile run can report the realized cut rate
    alongside the classification stats.
    (No run-length embedding argument: TAPT carries no learnable temporal state,
    matching plain RLT -- see the module docstring.)

    After each forward(), `self.last_stats` holds the classification diagnostics
    for the clip just encoded (see apt_temporal_static_tokens.classification_stats);
    `missed_reuse` there is the fraction of base cells that did not change at all
    but still had to pay for a token because the quadtree boundaries wobbled.
    """

    def __init__(
        self,
        apt: SiglipAPTEmbeddings,
        threshold: float = 0.2,
        attn_mode: str = "reuse",
        mask_mode: str = "ref",
        refresh_every: int = 0,
        mask_space: str = "pixel",
        embed_threshold: float = 0.34,
        embed_metric: str = "l2",
        window: int = 1,
        cut_threshold: float = 0.8,
        partition_mode: str = "entropy",
        run_tol: int = 0,
        persist: bool = False,
    ) -> None:
        super().__init__()
        self.apt = apt
        self.threshold = threshold
        assert attn_mode in ("reuse", "per_frame"), f"unknown attn_mode {attn_mode!r}"
        self.attn_mode = attn_mode
        assert mask_mode in ("ref", "consec"), f"unknown mask_mode {mask_mode!r}"
        self.mask_mode = mask_mode
        self.refresh_every = refresh_every
        assert mask_space in ("pixel", "embed"), f"unknown mask_space {mask_space!r}"
        self.mask_space = mask_space
        self.embed_threshold = embed_threshold
        assert embed_metric in ("l2", "cosine"), f"unknown embed_metric {embed_metric!r}"
        self.embed_metric = embed_metric
        assert window >= 1, f"window must be >= 1, got {window}"
        self.window = window
        self.cut_threshold = cut_threshold
        assert partition_mode in ("entropy", "survivor"), \
            f"unknown partition_mode {partition_mode!r}"
        self.partition_mode = partition_mode
        self.run_tol = run_tol
        self.persist = persist
        self.last_stats: Dict[str, float] = {}
        self.last_windows: int = 0

    @property
    def active_threshold(self) -> float:
        """The threshold actually in force, given mask_space. For logging.

        Mirrors SiglipRLTEmbeddings.active_threshold so llava_arch's TAPT log
        line can report the same way the RLT one does.
        """
        return self.embed_threshold if self.mask_space == "embed" else self.threshold

    def _kv_rows_per_frame(self, origin_index: torch.Tensor, events_per_frame):
        """Per frame, the packed rows that make up its FULL partition.

        origin_index[t, i, j] is the packed row covering base cell (i, j) at frame t --
        that cell's own fresh event, or the token it carries from its last event. A coarse
        token spans several base cells, so it appears many times in origin_index; unique()
        collapses it back to ONE key.

        That dedup is what keeps the invariant: when every cell is FRESH (nothing reused),
        unique(origin_index[t]) is exactly frame t's APT partition, so this reduces to plain
        APT. Gathering all G*G base cells instead would give a coarse token attention mass
        proportional to its area and would NOT reduce to APT.

        Frames with zero events contribute no queries, so their kv block is dropped too --
        q_seqlen and kv_seqlen MUST describe the same frames, in the same order.

        Returns (kv_rows concatenated frame-major, kv_seqlen, q_seqlen) over active frames.
        """
        rows, kv_seqlens, q_seqlens = [], [], []
        for t, n_ev in enumerate(events_per_frame):
            if n_ev == 0:                              # fully-redundant frame: no queries
                continue
            r = torch.unique(origin_index[t])          # sorted, deduped packed rows
            rows.append(r)
            kv_seqlens.append(int(r.numel()))
            q_seqlens.append(int(n_ev))
        return torch.cat(rows), kv_seqlens, q_seqlens

    def _run_encoder_reuse(self, x, events_per_frame, origin_index):
        """SigLIP encoder where each frame's events attend over its FULL partition.

        Same fix as siglip_rlt_embeddings._run_encoder_reuse. The legacy path packs only the
        EVENT tokens and gives each frame a block containing just those -- so a frame whose
        scene barely changed encodes its handful of new tokens against almost nothing, while
        the SigLIP weights expect a whole frame. Here the queries are still only the events
        (so cost still scales with the event rate), but the keys/values are the frame's whole
        partition: fresh events plus the tokens carried from their last event.

        Reusing a carried token's k/v is exact whenever its content is unchanged, and the
        scale always lines up: classify_cells only marks a cell REDUNDANT when shape_match
        holds against t-1, so by transitivity along the carry chain the token at origin_index
        has the same scale/boundary the cell needs at t. No resampling is possible or needed.
        """
        apt = self.apt
        kv_rows, kv_seqlens, q_seqlens = self._kv_rows_per_frame(origin_index, events_per_frame)
        assert sum(q_seqlens) == x.shape[0], (
            f"q_seqlen total {sum(q_seqlens)} != packed events {x.shape[0]}; the packed "
            f"sequence must be frame-major and cover exactly the event tokens."
        )
        attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlens, kv_seqlen=kv_seqlens)

        N, C = x.shape
        for layer in apt.encoder_layers:
            a = layer.self_attn
            H, d = a.num_heads, a.head_dim

            h = layer.layer_norm1(x)                                   # (N, C) events only
            q = a.q_proj(h).view(1, N, H, d)
            # layer_norm/k_proj/v_proj are position-wise, so projecting the packed rows and
            # THEN gathering is identical to carrying hidden states forward and projecting.
            k = a.k_proj(h).index_select(0, kv_rows).view(1, -1, H, d)
            v = a.v_proj(h).index_select(0, kv_rows).view(1, -1, H, d)

            o = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_bias, scale=a.scale
            ).reshape(N, C)

            x = x + a.out_proj(o)                                      # residual, events only
            x = x + layer.mlp(layer.layer_norm2(x))                    # MLP, events only

        if getattr(apt, "apply_post_layernorm", False):
            x = apt.post_layernorm(x)
        return x

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
            combined_survivors: (N, C) encoded packed tokens (N = number of FRESH
                cells across the clip, summed over scales).
            origin_index: (T, G, G) long, for apt_temporal_scatter_back.
            T: frames in the clip.
            P: dense patch count per frame (G*G), matching one clip's vision
                output (same contract as apt_scatter_back).
        """
        apt = self.apt
        T = pixel_values.shape[0]
        frames = self._resize_to_apt_input(pixel_values)
        base_p = apt.base_patch_size
        G = apt.image_size // base_p
        C = apt.embed_dim
        dev = frames.device

        # ORDER: dirty -> cuts -> partition -> classification. The dirty check
        # runs at base scale and never references the partition
        # (dirty_subtile_mask{,_embed} both take base_patch_size), so it can be
        # computed first -- which is what lets cut detection drive the window
        # boundaries the partition is then built over. Nothing here is circular.
        #
        # Base tiles as (T,G,G,3,p,p). Needed here rather than just before the
        # merge pass because mask_space="embed" runs the dirty check on their
        # embeddings, which therefore have to exist BEFORE classification.
        patches = frames.unfold(2, base_p, base_p).unfold(3, base_p, base_p)   # (T,3,G,G,p,p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()               # (T,G,G,3,p,p)

        if self.mask_space == "embed":
            # Embed EVERY base tile, not just the FRESH ones, because which
            # tiles are FRESH is what this test is about to decide. That costs
            # the frozen patch conv on the redundant tiles too -- ~0.7M MACs
            # each against the ~410M MACs of transformer a token costs, i.e.
            # well under 1% (the same arithmetic that retired the OVERRIDE
            # class; see apt_temporal_static_tokens' HISTORY note). The
            # decision is nearly free to make on the right quantity, so there
            # is no reason to make it on a pixel proxy whose sensitivity APT
            # has already biased -- see dirty_subtile_mask_embed's docstring.
            #
            # No position embedding is added yet, which is exactly what
            # find_idxs_to_keep_embed requires; the merge pass adds it below.
            raw_tile = apt._embed_patches(
                patches.reshape(-1, 3, base_p, base_p)
            ).view(T, G, G, C)                                                 # (T,G,G,C)
            dirty = dirty_subtile_mask_embed(
                raw_tile, self.embed_threshold, metric=self.embed_metric,
                mask_mode=self.mask_mode, refresh_every=self.refresh_every,
            )
        else:
            # Dirty check on the SAME (SigLIP-normalized) pixel scale RLT itself
            # uses -- no un-normalize step -- so self.threshold is directly the
            # shared config.rlt_threshold value, not a second differently-scaled
            # knob (see dirty_subtile_mask's docstring).
            raw_tile = None
            dirty = dirty_subtile_mask(frames.float(), self.threshold, base_p,
                                        mask_mode=self.mask_mode, refresh_every=self.refresh_every)

        # ---- partition, over windows rather than per frame.
        # window=1 (default) puts every frame in its own window, so
        # window_max_entropy is a no-op max over one frame and this reduces to
        # the per-frame partition EXACTLY -- windowing is inert until enabled.
        #
        # shape_match_grid is deliberately still called. Inside a window every
        # frame carries the identical partition, so it is satisfied for free
        # (that is the entire point); at a window boundary the partition changes
        # and it correctly reports FRESH; and if two adjacent windows happen to
        # agree, reuse crosses the boundary at no extra cost. Keeping it also
        # keeps `missed_reuse` meaningful as the diagnostic that motivated this.
        if self.partition_mode == "survivor":
            # RLT-first: the partition is built from what RLT could NOT drop, so
            # spatial merging is additive on top of temporal reuse instead of
            # competing with it for the same redundancy. `window` does not apply
            # -- there is no entropy map to pool, and survivorship is already a
            # per-frame temporal quantity. See survivor_aligned_masks, including
            # its KNOWN COST note about the motion-stops transition (watch
            # missed_reuse).
            masks = survivor_aligned_masks(
                dirty, compute_run_lengths(dirty), apt.patch_sizes, base_p,
                run_tol=self.run_tol, persist=self.persist,
            )
            self.last_windows = T
        else:
            importance = apt.tokenizer.compute_importance_maps(frames)
            if self.window > 1:
                seg_id = window_ids(detect_cuts(dirty, self.cut_threshold), self.window)
                importance = window_max_entropy(importance, seg_id)
                self.last_windows = int(seg_id.max().item()) + 1
            else:
                self.last_windows = T
            masks = select_patches_by_threshold(importance, thresholds=apt.tokenizer.thresholds)

        scale_grid = dense_scale_code_grid(masks, apt.patch_sizes, base_p)

        shape_match = shape_match_grid(scale_grid, masks, apt.patch_sizes, base_p)
        # Aggregated over frame t's OWN cell footprint (masks[ps][t]), not
        # frame t-1's -- see cell_all_quiet's docstring for why the latter
        # produces inconsistent classification within one of frame t's cells.
        all_quiet = cell_all_quiet(dirty, masks, apt.patch_sizes, base_p)
        cls = classify_cells(shape_match, all_quiet)
        self.last_stats = classification_stats(shape_match, all_quiet)

        is_new_token = cls == FRESH                    # == (cls != REDUNDANT)
        token_origin_frame = compute_origin_index(is_new_token)

        # ---- base-tile embedding: E(patch)+pos for every tile that belongs to a
        # FRESH cell. cls is uniform over each of frame t's own cells (shape_match
        # and all_quiet are both keyed to masks[ps][t]), so every base tile under a
        # FRESH cell is itself FRESH -- which means the non-FRESH slots left at zero
        # below are never read by the merge pass. (A violation of that invariant
        # would surface as the `origin_index >= 0` assert further down.)
        if raw_tile is not None:
            # mask_space="embed" already ran the frozen conv over every tile;
            # select rather than embed a second time.
            fresh_vals = raw_tile[is_new_token]                                # (Nfresh, C)
        else:
            fresh_vals = apt._embed_patches(
                patches[is_new_token].reshape(-1, 3, base_p, base_p)
            )                                                                  # (Nfresh, C)
        pos_grid = apt.base_pos_embed.to(frames.dtype).view(G, G, C)
        fresh_vals = fresh_vals + pos_grid.unsqueeze(0).expand(T, G, G, C)[is_new_token]

        tile_embed = torch.zeros(T, G, G, C, device=dev, dtype=fresh_vals.dtype)
        tile_embed[is_new_token] = fresh_vals

        # ---- merge pass, grouped by scale (scale-major order first; frame
        # index recorded per event so the packed sequence can be reordered to
        # frame-major -- required for per-frame BlockDiagonalMask boundaries).
        scale_tokens, scale_frame_idx, scale_events = [], [], []
        zero3_pad = None
        for idx, ps in enumerate(apt.patch_sizes):
            code = idx + 1
            s = ps // base_p
            coarse_code = scale_grid[:, ::s, ::s]
            coarse_new = is_new_token[:, ::s, ::s]
            event_mask = (coarse_code == code) & coarse_new                # (T, G//s, G//s)
            n_events = int(event_mask.sum().item())
            scale_events.append((event_mask, s))

            # patch_attn / zero_conv / _embed_patches are ZeRO-3-
            # PARTITIONED: every call issues an all-gather. A rank whose clip happens to
            # have no cells at this scale must STILL issue them, in the same order, or the
            # collective sequence desyncs across ranks and the job deadlocks (NCCL
            # watchdog after ddp_timeout, naming neither rank nor cause). So the calls
            # below are unconditional, on dummy tensors when n_events == 0 -- the
            # same fix already applied inside siglip_apt_embeddings._embed.
            if s == 1:
                # Base tiles were embedded unconditionally above, so this branch calls
                # nothing partitioned of its own; the dummy only feeds a grad-free
                # zero into zero3_pad, which is harmless because s==1 touches no
                # partitioned module.
                token_value = (tile_embed[event_mask] if n_events > 0        # (n_events, C)
                               else tile_embed.new_zeros(1, C))
            else:
                if n_events > 0:
                    children = tile_embed.view(T, G // s, s, G // s, s, C)
                    children = children.permute(0, 1, 3, 2, 4, 5)           # (T,Gs,Gs,s,s,C)
                    children = children[event_mask]                         # (n_events,s,s,C)
                    attn_in = children.permute(0, 3, 1, 2).contiguous()     # (n_events,C,s,s)
                else:
                    attn_in = torch.zeros(1, C, s, s, device=dev, dtype=tile_embed.dtype)

                for _ in range(idx):                                        # Conv2d^i
                    attn_in = apt.patch_attn(attn_in)
                merged = apt.zero_conv(attn_in.squeeze(-1).squeeze(-1))     # (n_events|1, C)

                # E(Resize_p(P_i)) -- the coarse region resized to one base patch and
                # embedded by the frozen conv. Applied to EVERY event, which is what
                # keeps APT's zero-init identity: zero_conv is zero-init, so a coarse
                # token starts at exactly E(Resize_p) + pos, i.e. plain APT, i.e. no
                # perturbation of the pretrained model at step 0. (The removed OVERRIDE
                # class skipped this term and relied on the zero-init merge alone, which
                # made its tokens a bare position embedding at step 0 -- see the history
                # note in apt_temporal_static_tokens.py.)
                group_masks = {
                    p: (event_mask.to(frames.dtype) if p == ps
                        else torch.zeros(T, apt.image_size // p, apt.image_size // p,
                                          device=dev, dtype=frames.dtype))
                    for p in apt.patch_sizes
                }
                patch_groups = apt.tokenizer.construct_patch_groups(frames, group_masks)
                resize_patches = patch_groups[f"resized_patches_{ps}"]
                if n_events == 0:
                    resize_patches = torch.zeros(1, 3, base_p, base_p, device=dev,
                                                 dtype=frames.dtype)
                embed_scale = apt._embed_patches(resize_patches).to(merged.dtype)

                token_value = merged + embed_scale
                if n_events > 0:
                    new_g = apt.image_size // ps
                    resampled_pos = resample_abs_pos_embed(
                        apt.base_pos_embed.to(frames.dtype), new_size=(new_g, new_g),
                        old_size=(apt.base_grid_size, apt.base_grid_size), num_prefix_tokens=0,
                    )
                    pos_grid_ps = resampled_pos.view(new_g, new_g, C)
                    pos_for_events = pos_grid_ps.unsqueeze(0).expand(T, new_g, new_g, C)[event_mask]
                    token_value = token_value + pos_for_events.to(merged.dtype)

            if n_events == 0:
                # Padding calls only. Fold them into the graph with weight 0 so
                # patch_attn/zero_conv stay reachable from the loss on
                # this rank -- a forward with no corresponding backward leaves their grad
                # hooks unfired, desyncing the gradient reduce-scatter and reintroducing
                # the very deadlock the unconditional calls exist to prevent.
                pad = token_value.sum()
                zero3_pad = pad if zero3_pad is None else zero3_pad + pad
                continue

            frame_idx_for_events = torch.nonzero(event_mask, as_tuple=False)[:, 0]  # (n_events,)
            scale_tokens.append(token_value)
            scale_frame_idx.append(frame_idx_for_events)

        combined_scale_major = torch.cat(scale_tokens, dim=0)              # (N, C)
        if zero3_pad is not None:
            combined_scale_major = combined_scale_major + 0.0 * zero3_pad
        frame_idx_scale_major = torch.cat(scale_frame_idx, dim=0)          # (N,)

        # Reorder to frame-major so per-frame blocks are contiguous for the
        # BlockDiagonalMask (mirrors the per-frame-only attention convention
        # both SiglipAPTEmbeddings and SiglipRLTEmbeddings already use).
        order = torch.argsort(frame_idx_scale_major, stable=True)
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=dev)
        combined_frame_major = combined_scale_major[order]

        sorted_frame_idx = frame_idx_scale_major[order]
        # Unfiltered, length T. "reuse" needs this to keep q_seqlen and kv_seqlen describing
        # the SAME frames; the legacy path filters the zeros out just below.
        events_per_frame = [int((sorted_frame_idx == t).sum().item()) for t in range(T)]
        seqlens = [n for n in events_per_frame if n > 0]                   # skip fully-redundant frames

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

        if self.attn_mode == "reuse":
            # Events attend over their frame's FULL partition (fresh + carried tokens),
            # not just the handful of other events in that frame. Reduces to plain APT
            # when nothing is reused -- see _run_encoder_reuse / _kv_rows_per_frame.
            combined_survivors = self._run_encoder_reuse(
                combined_frame_major, events_per_frame, origin_index
            )                                                              # (N,C)
        else:
            # LEGACY "per_frame": each frame's events attend ONLY to each other. A frame
            # that changed little encodes its few new tokens against almost nothing, while
            # the SigLIP weights expect a whole frame's worth of context. Kept for ablation.
            attn_bias = BlockDiagonalMask.from_seqlens(seqlens)
            x = combined_frame_major.unsqueeze(0)                          # (1,N,C)
            combined_survivors = apt._run_encoder(x, attn_bias).squeeze(0) # (N,C)

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
        apt, threshold=0.1,                  # 0.1: RLT's own default scale
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
    print(f"[static clip] stats: {tapt.last_stats}")

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

    # --- Test 2: a 2-frame clip where frame 1 is completely different
    #     (independent, high-entropy) content must produce 2 frames' worth of
    #     fresh events (no illegitimate collapsing across unrelated content),
    #     and running plain per-frame APT on each frame separately must give the
    #     same per-frame dense values TAPT produces.
    torch.manual_seed(1)
    f0 = torch.rand(1, 3, img, img, device=dev, dtype=dtype)
    f1 = torch.rand(1, 3, img, img, device=dev, dtype=dtype)
    clip2 = torch.cat([f0, f1], dim=0)
    clip2_norm = (clip2 - 0.5) / 0.5

    survivors2, origin2, T2, P2 = tapt(clip2_norm)
    dense2 = apt_temporal_scatter_back(survivors2, origin2, T2, P2)
    assert torch.isfinite(dense2.float()).all()

    # Independently run plain per-frame APT on each frame.
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

    # --- Test 4 (the zero-init identity): with zero_conv zero-init, EVERY coarse
    #     token -- not just some of them -- must equal E(Resize_p) + pos + psi(0),
    #     i.e. exactly what plain APT produces. This is the property the removed
    #     OVERRIDE class broke: its tokens skipped the E(Resize_p) anchor and so
    #     collapsed to a bare position embedding at step 0. Tests 1-3 above already
    #     assert TAPT == plain APT value-for-value on every frame they cover, which
    #     is that property; this makes the reason explicit.
    assert float(apt.zero_conv.weight.abs().sum()) == 0.0, "zero_conv must be zero-init"
    assert float(apt.zero_conv.bias.abs().sum()) == 0.0, "zero_conv must be zero-init"
    print("OK: zero-init at the seam -> every TAPT token reduces to plain APT's "
          "E(Resize_p)+pos at step 0 (asserted value-for-value by tests 1-3).")

    print("\nOK: SigLIP-APT-Temporal embeddings run end-to-end "
          "(classification -> embed/merge -> xformers attn -> scatter-back).")
