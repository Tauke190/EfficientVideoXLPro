#!/usr/bin/env python
"""
visualize_apt_video.py
======================

Visualize APT's adaptive (hierarchical / quadtree) patch partition on a video.

For each sampled frame this:
  1. resizes the frame to the APT processing grid (default 392x392 -> 28x28, the
     same geometry the SigLIP-APT encoder runs at),
  2. computes per-patch Shannon entropy at each scale and selects a strict
     spatial partition into 14 / 28 / 56 px patches (low-entropy regions merge
     into coarse patches), and
  3. overlays the chosen patches, color-coded per scale, plus per-frame stats
     (retained fraction = survivors / dense grid).

It reuses the exact primitives the encoder uses
(videoxlpro/model/multimodal_encoder/apt_static_tokens.py), so what you see is
what the model partitions.

Duration controls (how much of the video to visualize):
  --start      seconds to start from         (default 0)
  --duration   length of span in seconds     (default: whole video)
  --fps        output sampling rate          (default 4.0 frames/s of span)
  --num_frames exact number of frames to sample within the span (overrides --fps)

Example:
  python visualize_apt_video.py --video train.mp4 --start 2 --duration 5 --fps 6 \
      --thresholds 4.0 5.0 --out assets/apt_vis.mp4
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu


class H264Writer:
    """Write RGB frames to an H.264 mp4 (VS Code / browser playable) via ffmpeg,
    falling back to OpenCV mp4v if ffmpeg is unavailable."""

    def __init__(self, path, fps, size):
        W, H = size
        ff = shutil.which("ffmpeg") or "/home/av354855/miniconda3/bin/ffmpeg"
        self.proc = self.cv = None
        if os.path.exists(ff):
            cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0", "-an",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            except Exception:
                self.proc = None
        if self.proc is None:
            print("ffmpeg unavailable; falling back to mp4v (may not preview in VS Code)")
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, rgb):
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        else:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()

# Reuse the encoder's APT primitives (no duplicate math).
_ENC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "videoxlpro", "videoxlpro", "model", "multimodal_encoder",
)
sys.path.insert(0, _ENC_DIR)
from apt_static_tokens import compute_patch_entropy_batched, select_patches_by_threshold  # noqa: E402
from apt_temporal_static_tokens import (  # noqa: E402
    REDUNDANT, FRESH,
    dense_scale_code_grid, dirty_subtile_mask, shape_match_grid,
    cell_all_quiet, classify_cells,
)

# Per-scale overlay colors in RGB (finest -> coarsest).
DEFAULT_COLORS = [
    (60, 220, 60),    # 14px  fine detail (green)
    (70, 170, 255),   # 28px  (orange)
    (255, 70, 70),    # 56px  coarse / merged (red)
    (210, 70, 255),   # 112px (purple) -- if num_scales > 3
]

# APT-Temporal classification tint colors (RGB), drawn as a translucent fill
# under the scale-color outlines.
#   REDUNDANT     -- dim gray: carried forward from the last event, no compute.
#   FRESH         -- no tint: content genuinely changed, re-embedded (as it should be).
#   MISSED_REUSE  -- bright magenta: the cell did NOT change at all, but its quadtree
#                    boundaries disagree with the previous frame's, so there is no token
#                    of the right shape to carry forward and it pays for a full token
#                    anyway. Pure loss to PARTITION INSTABILITY -- these are the cells
#                    threshold hysteresis / windowed re-partitioning would recover.
TAPT_TINT_REDUNDANT = (80, 80, 80)
TAPT_TINT_MISSED = (255, 0, 220)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize APT adaptive patches on a video.")
    p.add_argument("--video", type=str, default="train.mp4", help="Input video path.")
    p.add_argument("--out", type=str, default="assets/apt_vis.mp4", help="Output annotated mp4.")
    # duration / sampling
    p.add_argument("--start", type=float, default=0.0, help="Start time (seconds).")
    p.add_argument("--duration", type=float, default=None, help="Span to visualize (seconds); default whole video.")
    p.add_argument("--fps", type=float, default=4.0, help="Output sampling rate (frames per second of span).")
    p.add_argument("--num_frames", type=int, default=None, help="Exact #frames within the span (overrides --fps).")
    p.add_argument("--out_fps", type=float, default=None, help="Playback fps of the output video (default = --fps).")
    # APT params (match the model defaults)
    p.add_argument("--image_size", type=int, default=392, help="APT processing size (square). 392 -> 28x28 grid.")
    p.add_argument("--patch_size", type=int, default=14, help="Base patch size.")
    p.add_argument("--num_scales", type=int, default=3, help="Number of scales (3 -> 14/28/56).")
    p.add_argument("--thresholds", type=float, nargs="+", default=[4.0, 4.0],
                   help="Entropy thresholds, one per non-base scale (len = num_scales-1).")
    # APT-Temporal (TAPT) overlay
    p.add_argument("--temporal", action="store_true",
                   help="Also classify each cell REDUNDANT/FRESH vs. the previous frame (see "
                        "apt_temporal_static_tokens.py) and tint the overlay, highlighting "
                        "cells that did not change but still cost a token (missed reuse).")
    p.add_argument("--pixel_threshold", type=float, default=12.0,
                   help="0-255-scale mean-abs pixel L1 vs. the previous frame (TAPT dirty signal).")
    p.add_argument("--mask_mode", type=str, default="ref", choices=["ref", "consec"],
                   help="TAPT dirty-check mode: 'ref' (default, matches config.rlt_mask_mode's "
                        "default) diffs each tile against the reference it would actually be "
                        "carried forward from, bounding drift regardless of run length; 'consec' "
                        "is the legacy per-frame-t-1 diff, which lets slow drift accumulate "
                        "unboundedly across a long REDUNDANT run.")
    p.add_argument("--refresh_every", type=int, default=0,
                   help="ref mode only: force-refresh every Nth frame regardless of drift. 0 disables.")
    # rendering
    p.add_argument("--canvas", type=int, default=784, help="Output frame size (square); patches scaled to match.")
    p.add_argument("--thickness", type=int, default=1, help="Patch outline thickness.")
    p.add_argument("--save_frames", type=str, default=None, help="Optional dir to also dump annotated frames as JPGs.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sample_frames(video_path, start, duration, fps, num_frames):
    """Return (frames_rgb_uint8 [N,H,W,3], timestamps [N]) for the requested span."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    avg_fps = float(vr.get_avg_fps()) or 30.0

    start_f = max(0, int(round(start * avg_fps)))
    if duration is not None and duration > 0:
        end_f = min(total, int(round((start + duration) * avg_fps)))
    else:
        end_f = total
    end_f = max(end_f, start_f + 1)

    if num_frames is not None and num_frames > 0:
        idxs = np.linspace(start_f, end_f - 1, num_frames, dtype=int)
    else:
        step = max(1, int(round(avg_fps / max(fps, 1e-6))))
        idxs = np.arange(start_f, end_f, step, dtype=int)
    idxs = np.clip(idxs, 0, total - 1)

    frames = vr.get_batch(idxs.tolist()).asnumpy()       # (N, H, W, 3) RGB uint8
    timestamps = [round(i / avg_fps, 2) for i in idxs.tolist()]
    print(f"video={video_path} total={total} avg_fps={avg_fps:.2f} -> sampled {len(idxs)} "
          f"frames over [{start:.2f}s, {timestamps[-1]:.2f}s]")
    return frames, timestamps


def compute_partition(frame_rgb, image_size, patch_size, num_scales, thresholds, device):
    """Resize a frame to the APT grid and return per-scale 0/1 masks (Gs, Gs)."""
    sq = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(sq).to(device).permute(2, 0, 1).unsqueeze(0).float()   # (1,3,H,W) in [0,255]
    maps = compute_patch_entropy_batched(t, patch_size=patch_size, num_scales=num_scales)
    masks = select_patches_by_threshold(maps, thresholds=thresholds)
    return sq, {ps: masks[ps][0].detach().cpu().numpy() for ps in masks}


def compute_temporal_classification(sq_frames, patch_sizes, base_patch_size, num_scales,
                                     thresholds, pixel_threshold, device,
                                     mask_mode="ref", refresh_every=0):
    """Classify every base cell of every sampled frame as REDUNDANT/FRESH vs. the
    previous frame (see apt_temporal_static_tokens.py). Fully vectorized across the
    whole sampled sequence -- same primitives the encoder uses, so what's rendered is
    what SiglipAPTTemporalEmbeddings would actually do.

    sq_frames: list of (image_size, image_size, 3) uint8 RGB arrays (already
        resized to the APT processing grid by the per-frame compute_partition
        calls in main()).

    Returns (cls, missed), both (T, G, G) numpy arrays. `missed` marks cells that are
    all-quiet (nothing changed) but shape-mismatched, so they cannot be carried forward
    and cost a full token for no reason -- the partition-instability tax.
    """
    stacked = np.stack(sq_frames, axis=0)                                 # (T,H,W,3) uint8
    t = torch.from_numpy(stacked).to(device).permute(0, 3, 1, 2).float()  # (T,3,H,W) in [0,255]
    maps = compute_patch_entropy_batched(t, patch_size=base_patch_size, num_scales=num_scales)
    masks = select_patches_by_threshold(maps, thresholds=thresholds)

    scale_grid = dense_scale_code_grid(masks, patch_sizes, base_patch_size)
    dirty = dirty_subtile_mask(t, pixel_threshold, base_patch_size,
                                mask_mode=mask_mode, refresh_every=refresh_every)
    shape_match = shape_match_grid(scale_grid, masks, patch_sizes, base_patch_size)
    all_quiet = cell_all_quiet(dirty, masks, patch_sizes, base_patch_size)
    cls = classify_cells(shape_match, all_quiet)
    missed = (~shape_match) & all_quiet
    missed[0] = False                       # frame 0 has no predecessor; nothing was "missed"
    return cls.detach().cpu().numpy(), missed.detach().cpu().numpy()


def draw_temporal_tint(disp, cls_frame, missed_frame, base_patch_size, proc, canvas, alpha=0.35):
    """Alpha-blend a translucent per-base-cell tint onto `disp`: gray where the cell was
    carried forward (REDUNDANT), magenta where it did not change yet still cost a token
    (missed reuse), untinted where it genuinely changed. Drawn BEFORE the scale-color
    outlines in draw_overlay so those remain crisp on top."""
    scale = canvas / proc
    side = int(round(base_patch_size * scale))
    G = cls_frame.shape[0]
    overlay = disp.copy()
    any_tint = False
    for i in range(G):
        for j in range(G):
            if int(cls_frame[i, j]) == REDUNDANT:
                color = TAPT_TINT_REDUNDANT
            elif missed_frame is not None and bool(missed_frame[i, j]):
                color = TAPT_TINT_MISSED
            else:
                continue
            any_tint = True
            x1, y1 = int(round(j * side)), int(round(i * side))
            cv2.rectangle(overlay, (x1, y1), (x1 + side - 1, y1 + side - 1), color, -1)
    if any_tint:
        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, dst=disp)
    return disp


def draw_overlay(sq_rgb, masks, patch_sizes, colors, thickness, canvas, ts, thresholds,
                  cls_frame=None, missed_frame=None, base_patch_size=None):
    """Draw color-coded adaptive patches + stats onto an upscaled canvas (RGB).

    cls_frame (optional): (G,G) REDUNDANT/FRESH classification for this frame vs. the
    previous one (see compute_temporal_classification) -- tinted underneath the
    scale-color outlines when provided, along with missed_frame (quiet but unreusable).
    """
    proc = sq_rgb.shape[0]
    scale = canvas / proc
    disp = cv2.resize(sq_rgb, (canvas, canvas), interpolation=cv2.INTER_LINEAR)

    if cls_frame is not None:
        disp = draw_temporal_tint(disp, cls_frame, missed_frame, base_patch_size, proc, canvas)

    counts = {}
    dense = (proc // patch_sizes[0]) ** 2
    for ps, color in zip(patch_sizes, colors):
        m = masks[ps]
        side = int(round(ps * scale))
        ys, xs = np.where(m > 0)
        counts[ps] = int(len(xs))
        for i, j in zip(ys.tolist(), xs.tolist()):
            x1, y1 = int(round(j * side)), int(round(i * side))
            cv2.rectangle(disp, (x1, y1), (x1 + side - 1, y1 + side - 1), color, thickness)

    survivors = sum(counts.values())
    retained = survivors / dense

    # Stats banner.
    lines = [
        f"t={ts:.2f}s   tokens={survivors}/{dense} ({100*retained:.1f}% kept)",
        f"thr={thresholds}",
    ]
    if cls_frame is not None:
        n_redundant = int((cls_frame == REDUNDANT).sum())
        n_fresh = int((cls_frame == FRESH).sum())
        n_missed = int(missed_frame.sum()) if missed_frame is not None else 0
        lines.append(
            f"TAPT: redundant={n_redundant} fresh={n_fresh} (missed_reuse={n_missed})"
        )
    y = 22
    for ln in lines:
        cv2.putText(disp, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    # Legend.
    for ps, color in zip(patch_sizes, colors):
        cv2.rectangle(disp, (8, y - 12), (24, y + 2), color, -1)
        txt = f"{ps}px x{counts[ps]}"
        cv2.putText(disp, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    if cls_frame is not None:
        for color, txt in (
            (TAPT_TINT_REDUNDANT, "REDUNDANT (reused, no compute)"),
            (TAPT_TINT_MISSED, "MISSED REUSE (unchanged, paid anyway)"),
        ):
            cv2.rectangle(disp, (8, y - 12), (24, y + 2), color, -1)
            cv2.putText(disp, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
    return disp, survivors, dense


def main():
    args = parse_args()
    assert len(args.thresholds) == args.num_scales - 1, (
        f"need num_scales-1 = {args.num_scales - 1} thresholds, got {args.thresholds}"
    )
    assert args.image_size % (args.patch_size * 2 ** (args.num_scales - 1)) == 0, (
        f"image_size {args.image_size} must be divisible by patch_size*2^(num_scales-1) "
        f"= {args.patch_size * 2 ** (args.num_scales - 1)} for a clean quadtree"
    )
    patch_sizes = [args.patch_size * (2 ** i) for i in range(args.num_scales)]
    colors = DEFAULT_COLORS[: args.num_scales]

    frames, timestamps = sample_frames(args.video, args.start, args.duration, args.fps, args.num_frames)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    out_fps = args.out_fps or args.fps
    writer = H264Writer(args.out, out_fps, (args.canvas, args.canvas))

    # Pass 1: resize + per-frame entropy partition for every sampled frame
    # (independent per frame -- exactly what the encoder's per-frame APT does).
    sq_frames, per_frame_masks = [], []
    for frame in frames:
        sq, masks = compute_partition(
            frame, args.image_size, args.patch_size, args.num_scales, args.thresholds, args.device
        )
        sq_frames.append(sq)
        per_frame_masks.append(masks)

    cls = missed = None
    if args.temporal:
        # Pass 2 (temporal, needs the whole sequence at once): classify every
        # base cell of every frame REDUNDANT/FRESH vs. the previous one.
        cls, missed = compute_temporal_classification(
            sq_frames, patch_sizes, args.patch_size, args.num_scales, args.thresholds,
            args.pixel_threshold, args.device,
            mask_mode=args.mask_mode, refresh_every=args.refresh_every,
        )
        n_redundant = int((cls == REDUNDANT).sum())
        n_fresh = int((cls == FRESH).sum())
        n_missed = int(missed.sum())
        n_total = cls.size
        print(f"[TAPT] classification over {len(frames)} frames: "
              f"redundant={n_redundant}/{n_total} ({100*n_redundant/n_total:.1f}%, reused -- the saving) "
              f"fresh={n_fresh}")
        # The diagnostic that matters: cells that did NOT change but still cost a token,
        # because the quadtree redrew its boundaries between frames. Recovering these is
        # a partition-stability problem (threshold hysteresis / windowed re-partitioning),
        # not a classification one.
        print(f"[TAPT] MISSED REUSE: {n_missed}/{n_total} ({100*n_missed/n_total:.1f}%) cells "
              f"were unchanged but unreusable (partition instability). Ceiling on reuse is "
              f"{100*(n_redundant + n_missed)/n_total:.1f}% if the partition were stable.")

    grand_keep = grand_dense = 0
    for k, (sq, masks, ts) in enumerate(zip(sq_frames, per_frame_masks, timestamps)):
        cls_frame = cls[k] if cls is not None else None
        missed_frame = missed[k] if missed is not None else None
        disp, keep, dense = draw_overlay(
            sq, masks, patch_sizes, colors, args.thickness, args.canvas, ts, args.thresholds,
            cls_frame=cls_frame, missed_frame=missed_frame, base_patch_size=args.patch_size,
        )
        grand_keep += keep
        grand_dense += dense
        writer.write(disp)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"frame_{k:04d}.jpg"),
                        cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        print(f"  frame {k:03d} t={ts:6.2f}s  kept {keep}/{dense} ({100*keep/dense:.1f}%)")

    writer.close()
    print(f"\nWrote {len(frames)} frames -> {args.out} @ {out_fps:g} fps")
    if grand_dense:
        print(f"AVERAGE retained = {grand_keep}/{grand_dense} ({100*grand_keep/grand_dense:.1f}%) "
              f"=> ~{grand_keep // max(len(frames),1)} tokens/frame vs {grand_dense // max(len(frames),1)} dense")


if __name__ == "__main__":
    main()
