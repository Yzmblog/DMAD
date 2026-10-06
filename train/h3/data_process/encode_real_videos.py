"""Encode UltraVideo clips into MiniMax-H3 video + audio latents (the real-data pool of the GAN critic).

Clips are read directly from the UltraVideo zips. For each clip the longest frame tier that fits
(default 124, then 107; both 17n+5 at 24 fps) is used, with a centered time window and the aligned audio.
One .pt per clip is written to <output_dir>/latents/:

    {"video_latent": bf16 [24, 5n+2, 48, 84]  (normalized: (z - mean) / std),
     "audio_latent": bf16 [2, n_audio, 32]    (stereo as two mono encodes, normalized),
     "meta": {..., "num_frames": tier}}

plus a per-rank manifest_rank{R}.jsonl (skipped clips are recorded with a reason).
Video frames are cover-scaled and center-cropped to 768x1344; HDR clips are tonemapped to SDR bt709.
Resumable: clips whose .pt already exists are skipped.

Usage (one process per GPU):
    torchrun --nproc_per_node=8 data_process/encode_real_videos.py \
        --model_dir /path/to/MiniMax-H3 --zip_dir /path/to/UltraVideo/clips_short_1920 --output_dir /path/to/real_latents
"""

import argparse
import glob
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from imageio_ffmpeg import get_ffmpeg_exe

FPS = 24
AUDIO_SR = 32000
AUDIO_HOP = 800
HEIGHT, WIDTH = 768, 1344
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="MiniMax-H3 root with vae/ and audio_vae/")
    ap.add_argument("--zip_dir", required=True, help="Directory with the UltraVideo clips_short_1920_*.zip files")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--zip_glob", default="clips_short_1920_*.zip")
    ap.add_argument("--num_frames_tiers", type=int, nargs="+", default=[124, 107],
                    help="Candidate clip lengths in frames (each 17n+5), tried longest first.")
    ap.add_argument("--max_clips", type=int, default=None, help="Encode only the first N clips (for a quick test).")
    return ap.parse_args()


def probe(ffmpeg, path):
    """Returns (duration_s, is_hdr, error)."""
    txt = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    if "Audio:" not in txt:
        return None, False, "no_audio"
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)
    if not m:
        return None, False, "no_duration"
    h, mnt, s = m.groups()
    is_hdr = any(k in txt for k in ("smpte2084", "arib-std-b67", "bt2020"))
    return int(h) * 3600 + int(mnt) * 60 + float(s), is_hdr, None


def decode_window(ffmpeg, path, t0, num_frames, is_hdr=False):
    """Decode num_frames video frames (24 fps, cover-scaled + center-cropped) and the aligned 32 kHz stereo audio."""
    vf = f"fps={FPS},scale=w={WIDTH}:h={HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}"
    if is_hdr:  # HDR10/HLG -> SDR bt709 (hable tonemap)
        vf += ",zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv"
    dur = num_frames / FPS
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.4f}", "-i", path,
         "-t", f"{dur + 0.25:.4f}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    frame_bytes = WIDTH * HEIGHT * 3
    n = len(r.stdout) // frame_bytes
    if n < num_frames:
        return None, None, f"short_decode_{n}"
    video = np.frombuffer(r.stdout[: num_frames * frame_bytes], dtype=np.uint8).reshape(num_frames, HEIGHT, WIDTH, 3)

    num_samples = int(round(num_frames / FPS * AUDIO_SR))
    ra = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.4f}", "-i", path,
         "-t", f"{dur + 0.25:.4f}", "-vn", "-ac", "2", "-ar", str(AUDIO_SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    audio = np.frombuffer(ra.stdout, dtype=np.float32).reshape(-1, 2)
    if audio.shape[0] < num_samples:
        if audio.shape[0] < num_samples - AUDIO_SR // 10:  # more than 100 ms missing
            return None, None, f"short_audio_{audio.shape[0]}"
        audio = np.pad(audio, ((0, num_samples - audio.shape[0]), (0, 0)))
    return video, audio[:num_samples].T.copy(), None  # [T, H, W, 3] uint8, [2, samples] float32


def load_vaes(model_dir, device):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    # Both VAEs keep fp32 weights; encoding runs under bf16 autocast.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{model_dir}/vae").to(device).eval()
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(f"{model_dir}/audio_vae").to(device).eval()
    vm = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    vs = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    am = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
    as_ = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
    return vae, avae, (vm, vs), (am, as_)


@torch.inference_mode()
def encode_clip(video_u8, audio_f32, vae, avae, vnorm, anorm, device):
    x = torch.from_numpy(video_u8.copy()).to(device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    pm = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    x = (x - pm) / ps  # the H3 video VAE expects ImageNet-normalized pixels
    with torch.autocast("cuda", torch.bfloat16):
        z = vae.encode(x).latent_dist.mode()
    z = (z.float() - vnorm[0]) / vnorm[1]  # [1, 24, 5n+2, 48, 84]

    a = torch.from_numpy(audio_f32.copy()).to(device).unsqueeze(1)  # [2, 1, samples]: stereo as a batch of 2
    za = avae.encode(a).latent_dist.mode().float()
    za = (za - anorm[0]) / anorm[1]  # [2, 32, n]
    # The audio VAE pads to a multiple of the hop and can emit one extra latent; crop to the packed length.
    target = int(round(video_u8.shape[0] / FPS * AUDIO_SR / AUDIO_HOP))
    za = za[:, :, :target]
    return z[0].to(torch.bfloat16).cpu(), za.permute(0, 2, 1).to(torch.bfloat16).cpu()


def main():
    args = parse_args()
    tiers = sorted(args.num_frames_tiers, reverse=True)
    for nf in tiers:
        if nf % 17 != 5:
            raise ValueError(f"num_frames tier must be 17n+5, got {nf}")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    ffmpeg = get_ffmpeg_exe()

    out_dir = Path(args.output_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)
    catalog = []
    for zp in sorted(glob.glob(os.path.join(args.zip_dir, args.zip_glob))):
        with zipfile.ZipFile(zp) as z:
            catalog.extend((zp, n) for n in sorted(z.namelist()) if n.endswith(".mp4"))
    if args.max_clips:
        catalog = catalog[: args.max_clips]
    mine = catalog[rank::world]
    print(f"[rank {rank}/{world}] {len(mine)}/{len(catalog)} clips", flush=True)

    vae, avae, vnorm, anorm = load_vaes(args.model_dir, device)
    manifest = open(out_dir / f"manifest_rank{rank}.jsonl", "a")
    done = skipped = 0
    for zp, name in mine:
        clip_id = os.path.splitext(os.path.basename(name))[0]
        out_pt = out_dir / "latents" / f"{clip_id}.pt"
        if out_pt.exists():
            done += 1
            continue
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        try:
            with zipfile.ZipFile(zp) as z, os.fdopen(fd, "wb") as f:
                f.write(z.read(name))
            dur, is_hdr, err = probe(ffmpeg, tmp)
            num_frames = t0 = None
            if err is None:
                num_frames = next((nf for nf in tiers if dur >= nf / FPS + 0.05), None)
                if num_frames is None:
                    err = f"too_short_{dur:.2f}"
            if err is None:
                t0 = max(0.0, (dur - num_frames / FPS) / 2)
                video, audio, err = decode_window(ffmpeg, tmp, t0, num_frames, is_hdr)
            if err is not None:
                manifest.write(json.dumps({"clip_id": clip_id, "skip": err}) + "\n")
                skipped += 1
                continue
            zv, za = encode_clip(video, audio, vae, avae, vnorm, anorm, device)
            meta = {"clip_id": clip_id, "src_zip": os.path.basename(zp), "duration": round(dur, 3), "hdr_tonemapped": is_hdr,
                    "t0": round(t0, 3), "num_frames": num_frames, "fps": FPS, "height": HEIGHT, "width": WIDTH,
                    "video_latent_shape": list(zv.shape), "audio_latent_shape": list(za.shape)}
            torch.save({"video_latent": zv, "audio_latent": za, "meta": meta}, out_pt)
            manifest.write(json.dumps(meta) + "\n")
            done += 1
        except Exception as e:  # one bad clip must not stop the pass
            manifest.write(json.dumps({"clip_id": clip_id, "skip": f"error:{str(e)[:150]}"}) + "\n")
            skipped += 1
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        if (done + skipped) % 100 == 0:
            manifest.flush()
            print(f"[rank {rank}] done={done} skipped={skipped}", flush=True)
    manifest.close()
    print(f"[rank {rank}] finished: done={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
