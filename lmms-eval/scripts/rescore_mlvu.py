"""
Re-score an existing MLVU_Test samples .jsonl log using the corrected A-F extractor.

Usage:
    python scripts/rescore_mlvu.py <path_to_samples_mlvu_test.jsonl>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

TASK_TYPES = {
    "anomaly_reco", "count", "ego", "needleQA",
    "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA",
}


def rescore(jsonl_path: str):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    category2score = {t: {"correct": 0, "answered": 0} for t in TASK_TYPES}
    unknown_tasks = set()

    for s in samples:
        resp = s["filtered_resps"]
        pred = extract_mcq_answer(resp, choices=["A", "B", "C", "D", "E", "F"])
        answer = s["target"]
        task_type = s["mlvu_percetion_score"]["task_type"]

        if task_type not in category2score:
            unknown_tasks.add(task_type)
            continue

        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += int(pred == answer)

    if unknown_tasks:
        print(f"[warn] unknown task types (skipped): {unknown_tasks}")

    print(f"\nResults on {len(samples)} samples")
    print(f"{'Task':<20} {'Correct':>7} {'Total':>7} {'Acc':>8}")
    print("-" * 46)

    task_scores = {}
    for task in sorted(TASK_TYPES):
        c = category2score[task]["correct"]
        a = category2score[task]["answered"]
        acc = 100 * c / a if a > 0 else 0.0
        task_scores[task] = acc
        print(f"{task:<20} {c:>7} {a:>7} {acc:>7.1f}%")

    avg = sum(task_scores.values()) / len(TASK_TYPES)
    print("-" * 46)
    print(f"{'mlvu_perception_score':<20} {'':>7} {'':>7} {avg:>7.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/rescore_mlvu.py <samples_mlvu_test.jsonl>")
        sys.exit(1)
    rescore(sys.argv[1])
