"""Add caption-paired real-video latents to the teacher-sample shards (the real side of the DMAD critic).

For every item of a teacher shard written by build_synthetic_dataset.py ({idx:09d}.{latent.pt,embed.pt,prompt.txt}),
look up the UltraVideo clip of that prompt (index_to_clipid.jsonl from build_prompt_list.py; item index = global prompt
index), VAE-encode it into the teacher latent space (Wan2pt1VAEInterface.encode on [-1, 1] video, times sigma_data),
and write an output shard with all original members plus {idx:09d}.real_latent.pt and {idx:09d}.real_meta.json.

Preprocessing (deterministic):
  * temporal: num_frames frames sampled uniformly over the whole clip (the teacher compresses the whole caption into
    one 5 s clip, so the real clip is time-normalized the same way); the speed factor is recorded in real_meta.json.
  * spatial: bilinear resize (antialiased) to cover the target size, then center crop (832x480 for 480p 16:9).
  * color: HDR clips (PQ/HLG/bt2020, about 4% of UltraVideo) are tone-mapped to SDR bt709 with ffmpeg before encoding.

One process per GPU; completed output shards are skipped on rerun.
"""

import argparse
import io
import json
import os
import tarfile

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface


def load_index_map(map_file):
    idx_to_clip = {}
    with open(map_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # merged duplicates carry several clip_ids; the first is the canonical source
            idx_to_clip[row["index"]] = row["clip_ids"][0]
    return idx_to_clip


MODEL_FPS = 16.0


def sample_frame_indices(n, fps, num_frames):
    """num_frames indices spread uniformly over the n source frames, and the resulting speed factor
    (displayed duration / source duration; < 1 = sped up)."""
    sel = np.round(np.linspace(0, n - 1, num_frames)).astype(np.int64)
    speed = ((num_frames - 1) / MODEL_FPS) / ((n - 1) / fps)
    meta = {"mode": "stretch", "speed_factor": round(float(speed), 4), "src_frames": int(n), "src_fps": round(float(fps), 3)}
    return sel, meta


_HDR_MARKERS = ("smpte2084", "arib-std-b67", "bt2020")
# linearize (PQ/HLG auto-detected from stream tags) -> 709 primaries -> hable tonemap -> 709 gamma
_TONEMAP_VF = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable,zscale=t=bt709"


def _read_frames_tonemapped(path, w, h):
    import subprocess

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path,
           "-vf", _TONEMAP_VF, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tonemap decode failed for {path}: {proc.stderr.decode()[-500:]}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)


def read_video_frames(path, num_frames):
    """Sample num_frames RGB frames -> ((T,H,W,3) uint8, meta dict)."""
    info = iio.immeta(path)
    is_hdr = any(m in info.get("pix_fmt", "") for m in _HDR_MARKERS)
    if is_hdr:
        w, h = info["size"]
        frames = _read_frames_tonemapped(path, w, h)
    else:
        frames = iio.imread(path, index=None)  # (N,H,W,3) uint8
    n = frames.shape[0]
    if n == 0:
        raise ValueError(f"no frames decoded from {path}")
    sel, meta = sample_frame_indices(n, info["fps"], num_frames)
    meta["hdr_tonemapped"] = bool(is_hdr)
    return frames[sel], meta


def preprocess(frames_thwc, target_w, target_h):
    """(T,H,W,3) uint8 -> (1,3,T,H',W') float in [-1,1], resize-to-cover + center crop."""
    v = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2).float() / 127.5 - 1.0  # (T,3,H,W)
    h, w = v.shape[-2:]
    scale = max(target_w / w, target_h / h)
    rh, rw = max(target_h, round(h * scale)), max(target_w, round(w * scale))
    v = F.interpolate(v, size=(rh, rw), mode="bilinear", align_corners=False, antialias=True)
    top, left = (rh - target_h) // 2, (rw - target_w) // 2
    v = v[:, :, top : top + target_h, left : left + target_w]
    return v.permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,T,H,W)


def output_shard_done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def process_shard(shard_path, out_path, idx_to_clip, args, tokenizer, limit_state):
    """Copy every member of shard_path into out_path and add {idx}.real_latent.pt."""
    tmp_path = out_path + ".tmp"
    with tarfile.open(shard_path, "r") as src:
        members = src.getmembers()
        indices = sorted({int(m.name.split(".")[0]) for m in members if m.name.split(".")[0].isdigit()})
        if args.limit_items > 0:
            room = args.limit_items - limit_state["done"]
            indices = indices[:room]
            members = [m for m in members if int(m.name.split(".")[0]) in set(indices)]
        if not indices:
            return 0

        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        by_idx = {}
        for m in members:
            head = m.name.split(".")[0]
            if head.isdigit():
                by_idx.setdefault(int(head), []).append(m)
        with tarfile.open(tmp_path, "w") as dst:
            # webdataset groups consecutive members with the same key prefix: write each index's members together
            for idx in tqdm(indices, desc=os.path.basename(shard_path)):
                clip = idx_to_clip.get(idx)
                if clip is None:
                    raise KeyError(f"index {idx} missing from {args.map_file}")
                clip_path = os.path.join(args.video_root, clip)
                frames, meta = read_video_frames(clip_path, args.num_frames)
                video = preprocess(frames, w, h).to(device="cuda", dtype=torch.bfloat16)
                latent = tokenizer.encode(video) * args.sigma_data  # (1,16,T_lat,h,w)
                buf = io.BytesIO()
                torch.save(latent[0].cpu(), buf)
                for m in by_idx[idx]:  # original latent/embed/prompt, source order
                    dst.addfile(m, src.extractfile(m))
                for name, data in (
                    (f"{idx:09d}.real_latent.pt", buf.getvalue()),
                    (f"{idx:09d}.real_meta.json", json.dumps({"clip_id": clip, **meta}).encode("utf-8")),
                ):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    dst.addfile(ti, io.BytesIO(data))
    os.rename(tmp_path, out_path)
    limit_state["done"] += len(indices)
    return len(indices)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_shards", nargs="+", required=True, help="teacher tar shard paths, in order")
    p.add_argument("--map_file", required=True, help="index_to_clipid.jsonl (build_prompt_list.py)")
    p.add_argument("--video_root", required=True, help="folder with the UltraVideo clips (clips_short_960)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--resolution", default="480p")
    p.add_argument("--aspect_ratio", default="16:9")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--sigma_data", type=float, default=1.0)
    p.add_argument("--limit_items", type=int, default=0, help="stop after this many items (smoke test; the last shard is partial); 0 = all")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    idx_to_clip = load_index_map(args.map_file)
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    limit_state = {"done": 0}
    failed = []
    for shard_path in args.teacher_shards:
        out_path = os.path.join(args.output_dir, os.path.basename(shard_path))
        if output_shard_done(out_path):
            print(f"skip completed {out_path}", flush=True)
            continue
        try:
            n = process_shard(shard_path, out_path, idx_to_clip, args, tokenizer, limit_state)
        except Exception as e:
            # a bad or missing clip fails its whole shard (every row needs real_latents); continue with the rest
            failed.append(os.path.basename(shard_path))
            print(f"FAILED {out_path}: {type(e).__name__}: {e}", flush=True)
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
            continue
        print(f"{out_path}: +{n} items (total {limit_state['done']})", flush=True)
        if args.limit_items > 0 and limit_state["done"] >= args.limit_items:
            print("limit_items reached, stopping")
            break
    if failed:
        print(f"DONE WITH {len(failed)} FAILED SHARDS: {failed}", flush=True)
    else:
        print("DONE, all shards ok", flush=True)


if __name__ == "__main__":
    main()
