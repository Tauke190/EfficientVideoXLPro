#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/av354855/miniconda3/envs/videoxlpro/bin/python}"

VIDEO="${1:-test_order_20.mp4}"
THRESHOLDS="${2:-4:6}"
STEM="$(basename "${VIDEO%.*}")"
OUT="${REPO_ROOT}/assets/apt_${STEM}.mp4"

cd "${REPO_ROOT}"
"${PYTHON}" visualize_apt_video.py \
  --temporal \
  --num_scales 3 \
  --pixel_threshold 25.5 \
  --video "${VIDEO}" \
  --num_frames 4000 \
  --out "${OUT}" \
  --out_fps 20 \
  --thresholds $(echo "${THRESHOLDS}" | tr ':,' '  ')
