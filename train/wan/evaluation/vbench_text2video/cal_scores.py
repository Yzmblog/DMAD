#!/usr/bin/env python3
"""Compute VBench final scores from an evaluation results directory.

Reads all ``*_eval_results.json`` files in the given directory, prints a table
of all 16 dimension scores (raw + normalized), and computes Quality Score,
Semantic Score, and Total Score.

Usage:
    python evaluation/vbench_text2video/cal_scores.py --result_dir <eval_results_dir>
"""

import argparse
import json
import os

QUALITY_DIMS = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_DIMS = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

NORMALIZE = {
    "subject consistency": (0.1462, 1.0),
    "background consistency": (0.2615, 1.0),
    "temporal flickering": (0.6293, 1.0),
    "motion smoothness": (0.706, 0.9975),
    "dynamic degree": (0.0, 1.0),
    "aesthetic quality": (0.0, 1.0),
    "imaging quality": (0.0, 1.0),
    "object class": (0.0, 1.0),
    "multiple objects": (0.0, 1.0),
    "human action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance style": (0.0009, 0.2855),
    "temporal style": (0.0, 0.364),
    "overall consistency": (0.0, 0.364),
}

QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def load_scores(result_dir: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith("_eval_results.json"):
            continue
        with open(os.path.join(result_dir, fname)) as f:
            data = json.load(f)
        for key, val in data.items():
            dim_name = key.replace("_", " ")
            if isinstance(val, list) and len(val) >= 1:
                scores[dim_name] = val[0]
            elif isinstance(val, (int, float)):
                scores[dim_name] = val
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute VBench final scores")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing *_eval_results.json files")
    args = parser.parse_args()

    scores = load_scores(args.result_dir)

    all_dims = QUALITY_DIMS + SEMANTIC_DIMS
    found = [d for d in all_dims if d in scores]
    missing = [d for d in all_dims if d not in scores]

    if missing:
        print(f"Warning: missing {len(missing)} dimensions: {', '.join(missing)}\n")

    header = f"{'Dimension':<28s} {'Raw':>8s} {'Norm':>8s} {'Weight':>6s} {'Group':<10s}"
    print(header)
    print("-" * len(header))

    norm_scores: dict[str, float] = {}
    for dim in all_dims:
        if dim not in scores:
            continue
        raw = scores[dim]
        lo, hi = NORMALIZE[dim]
        norm = (raw - lo) / (hi - lo) if hi > lo else 0.0
        norm = max(0.0, min(1.0, norm))
        norm_scores[dim] = norm
        w = DIM_WEIGHT[dim]
        group = "Quality" if dim in QUALITY_DIMS else "Semantic"
        print(f"  {dim:<26s} {raw:8.4f} {norm:8.4f} {w:6.1f} {group:<10s}")

    print()

    quality_num = sum(norm_scores.get(d, 0) * DIM_WEIGHT[d] for d in QUALITY_DIMS)
    quality_den = sum(DIM_WEIGHT[d] for d in QUALITY_DIMS if d in norm_scores)
    quality = quality_num / quality_den if quality_den else 0.0

    semantic_num = sum(norm_scores.get(d, 0) * DIM_WEIGHT[d] for d in SEMANTIC_DIMS)
    semantic_den = sum(DIM_WEIGHT[d] for d in SEMANTIC_DIMS if d in norm_scores)
    semantic = semantic_num / semantic_den if semantic_den else 0.0

    total = (QUALITY_WEIGHT * quality + SEMANTIC_WEIGHT * semantic) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    print(f"  {'Quality Score':<26s} {quality:8.4f}   (weighted avg of {len([d for d in QUALITY_DIMS if d in norm_scores])} quality dims)")
    print(f"  {'Semantic Score':<26s} {semantic:8.4f}   (weighted avg of {len([d for d in SEMANTIC_DIMS if d in norm_scores])} semantic dims)")
    print(f"  {'Total Score':<26s} {total:8.4f}   (quality*{QUALITY_WEIGHT} + semantic*{SEMANTIC_WEIGHT}) / {QUALITY_WEIGHT + SEMANTIC_WEIGHT}")


if __name__ == "__main__":
    main()
