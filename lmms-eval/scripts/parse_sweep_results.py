#!/usr/bin/env python3
"""Turn eval_videoxl_pro_F.sh's per-combo logs into an accuracy + token-drop table."""
import argparse
import csv
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ACC_RE = re.compile(r"Average Performance Across All Task Categories:\s*([\d.]+)%")
TOKENS_RE = re.compile(r"tokens kept=(\d+)/(\d+)")
VIDEOS_RE = re.compile(r"\[(?:RLT|APT|APT-Temporal)\] videos=(\d+)")

METHODS = ["rlt", "apt", "apt_rlt"]
METHOD_LABELS = {"rlt": "RLT", "apt": "APT", "apt_rlt": "APT+RLT"}
FRAME_COUNTS = [128, 256, 512, 1024, 2048]


def parse_log(path: Path):
    if not path.exists():
        return None
    text = ANSI_RE.sub("", path.read_text(errors="replace"))

    # One GRAND TOTAL block per accelerate rank (--num_processes=3): each rank only
    # saw its own shard of the --limit videos, so the run total is a sum, not a pick.
    token_matches = TOKENS_RE.findall(text)
    if not token_matches:
        return None
    kept = sum(int(k) for k, _ in token_matches)
    dense = sum(int(d) for _, d in token_matches)

    acc_matches = ACC_RE.findall(text)
    accuracy = float(acc_matches[-1]) if acc_matches else None

    return {
        "kept": kept,
        "dense": dense,
        "drop_pct": 100.0 * (1 - kept / dense) if dense else None,
        "accuracy_pct": accuracy,
        "videos": sum(int(v) for v in VIDEOS_RE.findall(text)),
        "ranks_reporting": len(token_matches),
    }


def find_latest_sweep_dir(base: Path) -> Path:
    candidates = sorted(base.glob("sweep_*"))
    if not candidates:
        sys.exit(f"No sweep_* directories found under {base}")
    return candidates[-1]


def fmt(value, suffix=""):
    return f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else "N/A"


def render_table(title, results, key):
    lines = [
        f"### {title}",
        "",
        "| Frames | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
        "|---" * (len(METHODS) + 1) + "|",
    ]
    for frames in FRAME_COUNTS:
        row = [str(frames)]
        for method in METHODS:
            r = results[(method, frames)]
            row.append(fmt(r[key], "%") if r else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", nargs="?", default=None,
                     help="Directory written by eval_videoxl_pro_F.sh (default: latest sweep_* under logs/videoxlpro_mlvu)")
    args = ap.parse_args()

    base = Path("logs/videoxlpro_mlvu")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_sweep_dir(base)
    print(f"Reading logs from {sweep_dir}\n")

    results = {
        (method, frames): parse_log(sweep_dir / f"{method}_f{frames}.log")
        for method in METHODS
        for frames in FRAME_COUNTS
    }

    csv_path = sweep_dir / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "frames", "accuracy_pct", "drop_pct", "tokens_kept",
                    "tokens_dense", "videos", "ranks_reporting"])
        for method in METHODS:
            for frames in FRAME_COUNTS:
                r = results[(method, frames)]
                if r is None:
                    w.writerow([method, frames, "", "", "", "", "", 0])
                else:
                    w.writerow([method, frames, r["accuracy_pct"], r["drop_pct"],
                                r["kept"], r["dense"], r["videos"], r["ranks_reporting"]])

    # accuracy_pct is mlvu_aggregate_results_test's macro-avg as-printed: at --limit 100
    # it divides by all 9 MLVU categories even though only the first few are sampled,
    # so it understates true accuracy by roughly (9 / categories actually sampled).
    # It's still valid to compare column-to-column here since every cell used the same
    # limit -- just don't read it as an absolute accuracy number.
    md = render_table("Accuracy (MLVU, limit=100, macro-avg as logged -- see note below)", results, "accuracy_pct")
    md += "\n\n" + render_table("Encoder token drop %", results, "drop_pct")

    missing = [f"{METHOD_LABELS[m]}@{f}" for m in METHODS for f in FRAME_COUNTS if results[(m, f)] is None]
    if missing:
        md += "\n\nMissing/failed runs (no GRAND TOTAL block found -- check the log): " + ", ".join(missing)

    md += ("\n\nNote: accuracy_pct is understated -- mlvu_aggregate_results_test divides by "
           "all 9 MLVU categories regardless of how many --limit 100 actually sampled. "
           "Comparable across this table's columns, not as an absolute number.")

    md_path = sweep_dir / "results_table.md"
    md_path.write_text(md + "\n")

    print(md)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
