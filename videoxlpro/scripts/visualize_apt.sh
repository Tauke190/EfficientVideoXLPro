#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/av354855/miniconda3/envs/videoxlpro/bin/python}"

VIDEO="${1:-test_order_20.mp4}"
THRESHOLDS="${2:-6.5:7.5}"
DURATION="${3:-50}"
START="${4:-0}"
FPS="${5:-4}"

STEM="$(basename "${VIDEO%.*}")"
OUT="${REPO_ROOT}/assets/apt_${STEM}.mp4"

cd "${REPO_ROOT}"
"${PYTHON}" visualize_apt_video.py \
  --video "${VIDEO}" \
  --out "${OUT}" \
  --start "${START}" \
  --duration "${DURATION}" \
  --fps "${FPS}" \
  --thresholds $(echo "${THRESHOLDS}" | tr ':,' '  ')
