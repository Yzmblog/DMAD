#!/usr/bin/env python3
"""List the caption indices that training needs: the captions of all encoded real clips of one frame tier,
plus the first few prompts that the trainer uses for periodic visualization.

Only these prompts have to be text-encoded (encode_prompts.py --index-file) and sampled by the teacher
(gen_teacher_data.py reads the resulting prompt cache).

Usage:
    python data_process/build_pool_indices.py --real-latents /path/to/real_latents \
        --caption-map /path/to/ultravideo_prompts/ultravideo_index_to_clipid.jsonl --out /path/to/pool_indices.txt
"""

import argparse
import glob
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-latents", required=True, help="Output dir of encode_real_videos.py (manifest_rank*.jsonl).")
    p.add_argument("--caption-map", required=True, help="ultravideo_index_to_clipid.jsonl from build_prompt_list.py.")
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=124, help="Frame tier used for training.")
    p.add_argument("--num-viz-prompts", type=int, default=4, help="Prompts 0..N-1 are always included (visualization).")
    return p.parse_args()


def main():
    args = parse_args()
    clip2idx = {}
    for line in open(args.caption_map, encoding="utf-8"):
        row = json.loads(line)
        for clip in row["clip_ids"]:
            clip2idx.setdefault(clip.rsplit(".", 1)[0], int(row["index"]))
    indices, n_clips = set(range(args.num_viz_prompts)), 0
    for manifest in sorted(glob.glob(os.path.join(args.real_latents, "manifest_rank*.jsonl"))):
        for line in open(manifest, encoding="utf-8"):
            row = json.loads(line)
            if "skip" in row or row.get("num_frames") != args.num_frames:
                continue
            n_clips += 1
            if row["clip_id"] in clip2idx:
                indices.add(clip2idx[row["clip_id"]])
    with open(args.out, "w") as f:
        f.write("".join(f"{i}\n" for i in sorted(indices)))
    print(f"{n_clips} clips @ {args.num_frames}f -> {len(indices)} caption indices -> {args.out}")


if __name__ == "__main__":
    main()
