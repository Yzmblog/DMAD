#!/usr/bin/env python3
"""Build the prompt list and caption map from UltraVideo's short.csv.

Outputs (in --out-dir):
    prompts_dedup.txt                 one caption per line ("Detailed Description", deduplicated, first-occurrence order)
    ultravideo_index_to_clipid.jsonl  {"index": caption index, "clip_ids": [clips with that caption]}  (dmad.real_caption_map)

Usage:
    python data_process/build_prompt_list.py --csv /path/to/UltraVideo/short.csv --out-dir /path/to/ultravideo_prompts
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMN = "Detailed Description"
EXPECTED_PROMPTS = 42158


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="UltraVideo short.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--expect", type=int, default=EXPECTED_PROMPTS, help="Expected number of unique captions (0 disables the check).")
    return p.parse_args()


def main():
    args = parse_args()
    csv.field_size_limit(10**9)
    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if COLUMN not in rows[0]:
        sys.exit(f"ERROR: {args.csv} has no {COLUMN!r} column")

    index_of, prompts, clip_ids = {}, [], []
    for row in rows:
        caption = (row[COLUMN] or "").strip()
        clip = (row["clip_id"] or "").strip()
        if not caption or not clip:
            continue
        if caption not in index_of:
            index_of[caption] = len(prompts)
            prompts.append(caption)
            clip_ids.append([])
        clip_ids[index_of[caption]].append(clip)
    if args.expect and len(prompts) != args.expect:
        sys.exit(f"ERROR: built {len(prompts)} prompts, expected {args.expect}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = [" ".join(p.split()) for p in prompts]  # one prompt per line: collapse internal newlines/whitespace
    (out_dir / "prompts_dedup.txt").write_text("\n".join(flat) + "\n", encoding="utf-8")
    with (out_dir / "ultravideo_index_to_clipid.jsonl").open("w", encoding="utf-8") as handle:
        for index, clips in enumerate(clip_ids):
            handle.write(json.dumps({"index": index, "clip_ids": clips}, ensure_ascii=False) + "\n")
    print(f"rows={len(rows)} prompts={len(prompts)} clips={sum(len(c) for c in clip_ids)} -> {out_dir}")


if __name__ == "__main__":
    main()
