import argparse
import json
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import distributed, log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.model_utils import init_weights_on_device, load_state_dict, load_state_dict_from_dcp
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding


def save_video(video, save_path, fps):
    """Write a [C, T, H, W] video in [0, 1] as an mp4 through ffmpeg."""
    frames = (rearrange(video.cpu().float().clamp(0, 1).numpy() * 255, "c t h w -> t h w c") + 0.5).astype(np.uint8)
    h, w = frames.shape[1:3]
    header = b"P6\n%d %d\n255\n" % (w, h)  # frames are piped as PPM images (a headerless raw pipe can be misread)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-vcodec", "ppm", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", save_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(header + f.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    proc.wait()
    if proc.returncode != 0 or os.path.getsize(save_path) < 1024:
        raise RuntimeError(f"ffmpeg failed for {save_path} (rc={proc.returncode}): {err[:500]}")

torch._dynamo.config.suppress_errors = True

TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}
RECTIFIED_FLOW_T_SCALING = 1000.0

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

_DEFAULT_MID_T = [15 / 16, 5 / 6, 5 / 8]

# ---------------------------------------------------------------------------
# Prompt loading & partitioning
# ---------------------------------------------------------------------------


def load_prompts(path: str, use_augmented: bool = True) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for entry in data:
        gen_prompt = entry.get("augmented_prompt", entry["prompt"]) if use_augmented else entry["prompt"]
        item = {
            "prompt": entry["prompt"],
            "gen_prompt": gen_prompt,
            "dimensions": entry.get("dimensions", []),
        }
        if "auxiliary_info" in entry:
            item["auxiliary_info"] = entry["auxiliary_info"]
        prompts.append(item)
    return prompts


def partition_prompts(prompts: list[dict], rank: int, world_size: int) -> list[dict]:
    return prompts[rank::world_size]


# ---------------------------------------------------------------------------
# Model construction & weight loading
# ---------------------------------------------------------------------------


def build_dit(arch: str, model_size: str):
    from rcm.networks.wan2pt1 import WanModel

    cfgs = {
        "1.3B": L(WanModel)(
            dim=1536, eps=1e-06, ffn_dim=8960, freq_dim=256, in_dim=16, model_type="t2v", num_heads=12, num_layers=30, out_dim=16, text_len=512
        ),
        "14B": L(WanModel)(
            dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256, in_dim=16, model_type="t2v", num_heads=40, num_layers=40, out_dim=16, text_len=512
        ),
    }
    with init_weights_on_device():
        net = instantiate(cfgs[model_size]).eval()
    return net


def load_dit_weights(net, dit_path: str, arch: str):
    if os.path.isdir(dit_path):
        state_dict = load_state_dict_from_dcp(dit_path)
    else:
        state_dict = load_state_dict(dit_path)

    for key in ["generator", "generator_ema"]:
        if key in state_dict.keys():
            state_dict = state_dict[key]
    for prefix in ["model.", "net.", "_fsdp_wrapped_module."]:
        state_dict = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    for k, v in state_dict.items():
        if k.endswith("patch_embedding.weight"):
            state_dict[k] = v.reshape(net.patch_embedding.weight.shape)
        if k.endswith("patch_embedding.bias"):
            state_dict[k] = v.reshape(net.patch_embedding.bias.shape)
    net.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict


# ---------------------------------------------------------------------------
# Timestep schedules
# ---------------------------------------------------------------------------


def parse_rf_time(value: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_rf_time_values(values) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (str, int, float)):
        values = [values]
    parsed = []
    for value in values:
        if isinstance(value, (int, float)):
            parsed.append(float(value))
            continue
        for item in str(value).replace(",", " ").split():
            parsed.append(parse_rf_time(item))
    return parsed


def parse_mid_t_schedules(spec: str | None) -> list[list[float]]:
    if not spec:
        return []
    schedules = []
    for item in spec.split(";"):
        if not item.strip():
            schedules.append([])
            continue
        schedules.append(parse_rf_time_values(item))
    return schedules


def build_distilled_t_steps(num_steps: int, sigma_max: float, mid_t: list[float], device) -> torch.Tensor:
    """Build distilled timestep schedule.  ``mid_t`` values are in RF space."""
    mid_t = mid_t[: num_steps - 1]
    sigma_max_rf = sigma_max / (sigma_max + 1)
    return torch.tensor([sigma_max_rf, *mid_t, 0], dtype=torch.float64, device=device)


def build_ode_t_steps(num_steps: int, sigma_max: float, shift: float, device) -> torch.Tensor:
    s = sigma_max / (sigma_max + 1)
    s = s / (shift - (shift - 1) * s)
    t = torch.linspace(s, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    return shift * t / (1 + (shift - 1) * t)


def build_distilled_schedules_per_chunk(
    step_counts: list[int],
    steps_per_chunk: list[int] | None,
    sigma_max: float,
    default_mid_t: list[float],
    mid_t_schedules: list[list[float]],
    device,
) -> list[torch.Tensor]:
    if mid_t_schedules and not steps_per_chunk:
        raise ValueError("--mid_t_schedules requires --steps_per_chunk")
    if mid_t_schedules and len(mid_t_schedules) != len(steps_per_chunk):
        raise ValueError(
            f"--mid_t_schedules must have one schedule per --steps_per_chunk entry, got {len(mid_t_schedules)} and {len(steps_per_chunk)}"
        )

    schedules = []
    for idx, steps in enumerate(step_counts):
        if mid_t_schedules:
            schedule_idx = min(idx, len(mid_t_schedules) - 1)
            mid_t = mid_t_schedules[schedule_idx]
        else:
            mid_t = default_mid_t
        if len(mid_t) < steps - 1:
            raise ValueError(f"Need {steps - 1} mid_t values for {steps} steps, got {mid_t}")
        schedules.append(build_distilled_t_steps(steps, sigma_max, mid_t, device))
    return schedules


def expand_steps_per_chunk(steps_per_chunk: list[int] | None, num_blocks: int, default_steps: int) -> list[int]:
    if not steps_per_chunk:
        return [default_steps] * num_blocks
    step_counts = [int(step) for step in steps_per_chunk[:num_blocks]]
    if len(step_counts) < num_blocks:
        step_counts.extend([step_counts[-1]] * (num_blocks - len(step_counts)))
    if any(step < 1 for step in step_counts):
        raise ValueError(f"steps_per_chunk must contain positive integers, got {steps_per_chunk}")
    return step_counts


# ---------------------------------------------------------------------------
# Sampling: bidirectional
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_bidirectional(
    net,
    init_noise: torch.Tensor,
    t_steps: torch.Tensor,
    condition: dict,
    uncondition: dict | None,
    guidance: float,
    ode: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    B, C, T, H, W = init_noise.shape
    x = init_noise.to(torch.float64) * t_steps[0]
    ones_B_1 = torch.ones(B, 1, device=x.device, dtype=x.dtype)
    use_cfg = uncondition is not None and guidance > 1.0

    if not ode:
        step_noises = [torch.randn(*init_noise.shape, dtype=torch.float32, device=x.device, generator=generator) for _ in range(len(t_steps) - 1)]

    for step_idx, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        timesteps_B_1 = (t_cur * ones_B_1 * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)

        v_cond = net(
            x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
            timesteps_B_T=timesteps_B_1,
            **condition,
        ).to(torch.float64)

        if use_cfg:
            v_uncond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=timesteps_B_1,
                **uncondition,
            ).to(torch.float64)
            v_pred = v_uncond + guidance * (v_cond - v_uncond)
        else:
            v_pred = v_cond

        if ode:
            x = x - (t_cur - t_next) * v_pred
        else:
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * step_noises[step_idx]

    return x.float()


# ---------------------------------------------------------------------------
# Sampling: causal
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_causal(
    net,
    init_noise: torch.Tensor,
    t_steps: torch.Tensor,
    t_steps_per_chunk: list[torch.Tensor] | None,
    steps_per_chunk: list[int] | None,
    condition: dict,
    uncondition: dict | None,
    guidance: float,
    first_chunk_t: int,
    chunk_t: int,
    ode: bool,
    generator: torch.Generator | None = None,
    cp_group=None,
    context_from_last_step: bool = False,
    context_from_last_step_start_chunk: int = 0,
) -> torch.Tensor:
    if context_from_last_step_start_chunk < 0:
        raise ValueError(f"context_from_last_step_start_chunk must be non-negative, got {context_from_last_step_start_chunk}")
    from rcm.utils.blockmask import AttnMaskSpec, BlockPattern
    from rcm.utils.context_parallel import broadcast
    from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode

    B, C, T, H, W = init_noise.shape
    frame_tokens = H * W // net.get_spatial_patch_size()
    assert (T - first_chunk_t) % chunk_t == 0
    num_blocks = 1 + (T - first_chunk_t) // chunk_t
    bp = BlockPattern(frame_tokens=frame_tokens, first_chunk_frames=first_chunk_t, chunk_frames=chunk_t)

    ones_B_1 = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
    use_cfg = uncondition is not None and guidance > 1.0

    kv_cond = net.allocate_kv_caches(max_len=T * frame_tokens)
    kv_uncond = net.allocate_kv_caches(max_len=T * frame_tokens) if use_cfg else None

    if not ode:
        default_steps = len(t_steps) - 1
        step_counts = expand_steps_per_chunk(steps_per_chunk, num_blocks, default_steps)
        max_steps = max(step_counts)
        step_noises = []
        for _ in range(max_steps):
            n = torch.randn_like(init_noise, dtype=torch.float32, generator=generator)
            step_noises.append(broadcast(n, cp_group) if cp_group is not None else n)
    else:
        step_counts = [len(t_steps) - 1] * num_blocks
        step_noises = None

    x_blocks = []
    for i in range(num_blocks):
        frame_start = bp.blocks_to_frames(i)
        block_size = bp.block_size(i)
        attn_meta = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
        t_steps_block = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
        num_steps = len(t_steps_block) - 1
        use_last_step_context = context_from_last_step and i >= context_from_last_step_start_chunk

        x = init_noise[:, :, frame_start : frame_start + block_size].to(torch.float64) * t_steps_block[0]

        for step_idx, (t_cur, t_next) in enumerate(zip(t_steps_block[:-1], t_steps_block[1:])):
            t_cur_B_block = repeat(t_cur * ones_B_1, "b 1 -> b t", t=block_size)
            is_last = step_idx == num_steps - 1
            kv_mode = KVCacheMode.APPEND if (use_last_step_context and is_last) else KVCacheMode.READONLY
            inf_cond = CausalInferenceState(mode=kv_mode, kv_caches=kv_cond, pattern=bp, block_cursor=i)

            v_cond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=(t_cur_B_block * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS),
                **condition,
                inference_state=inf_cond,
                attn_meta=attn_meta,
            ).float()

            if use_cfg:
                inf_uncond = CausalInferenceState(mode=kv_mode, kv_caches=kv_uncond, pattern=bp, block_cursor=i)
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                    timesteps_B_T=(t_cur_B_block * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS),
                    **uncondition,
                    inference_state=inf_uncond,
                    attn_meta=attn_meta,
                ).float()
                v_pred = v_uncond + guidance * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            if ode:
                x = x - (t_cur - t_next) * v_pred.to(torch.float64)
            else:
                noise = step_noises[step_idx][:, :, frame_start : frame_start + block_size]
                x = (1 - t_next) * (x - t_cur * v_pred.to(torch.float64)) + t_next * noise

        if not use_last_step_context:
            zero_t = (0 * t_cur_B_block).to(**TENSOR_KWARGS)
            inf_append_cond = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_cond, pattern=bp, block_cursor=i)
            net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **condition, inference_state=inf_append_cond, attn_meta=attn_meta)
            if use_cfg:
                inf_append_uncond = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_uncond, pattern=bp, block_cursor=i)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **uncondition, inference_state=inf_append_uncond, attn_meta=attn_meta)

        x_blocks.append(x)

    del kv_cond, kv_uncond
    torch.cuda.empty_cache()
    return torch.cat(x_blocks, dim=2).float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(prompt: str) -> str:
    # VBench standard mode reconstructs filenames as "{prompt}-{idx}.mp4".
    # Do not truncate the prompt, or standard-dimension evaluation will miss files.
    name = prompt.replace("/", "-").replace("\n", " ")
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Distributed VBench video sampling", formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--arch", choices=["bidirectional", "causal"], required=True)
    p.add_argument("--distilled", action="store_true", help="Few-step distilled sampling (default: multi-step ODE / diffusion)")
    p.add_argument("--prompt_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None, help="Output directory (default: auto-generated from run config)")
    p.add_argument("--num_samples", type=int, default=5, help="Number of videos per prompt")
    p.add_argument("--no_augmented_prompt", action="store_true", help="Disable augmented prompts; use original short prompts instead")

    p.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B")
    p.add_argument("--dit_path", type=str, required=True)
    p.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth")
    p.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")

    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--resolution", type=str, default="480p")
    p.add_argument("--aspect_ratio", type=str, default="16:9")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num_steps", type=int, default=4, help="Sampling steps (1-4 for distilled, e.g. 50 for diffusion ODE)")
    p.add_argument("--sigma_max", type=float, default=1600)
    p.add_argument("--mid_t", type=str, nargs="*", default=None, help="Intermediate RF timesteps for distilled mode")
    p.add_argument("--steps_per_chunk", type=int, nargs="*", default=None, help="Causal distilled steps per chunk; repeats last value, e.g. 4 2")
    p.add_argument("--mid_t_schedules", type=str, default="", help="Per-entry schedules aligned with --steps_per_chunk, e.g. '0.9,0.75,0.5;5/6'")

    p.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale (non-distilled)")
    p.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--timestep_shift", type=float, default=5.0)

    # Causal-specific
    p.add_argument("--first_chunk_t", type=int, default=1)
    p.add_argument("--chunk_t", type=int, default=1)
    p.add_argument("--context_from_last_step", action="store_true", help="Cache KVs from last denoising step instead of an extra t=0 pass")
    p.add_argument("--context_from_last_step_start_chunk", type=int, default=0, help="First causal chunk index to use --context_from_last_step")

    # Context parallelism
    p.add_argument("--cp_size", type=int, default=1, help="Context parallelism size (default: 1, no CP)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_distilled = args.distilled
    is_causal = args.arch == "causal"
    device = torch.device("cuda")

    assert args.cp_size >= 1
    assert args.cp_size == 1 or is_causal, "--cp_size > 1 requires --arch causal"
    assert world_size % args.cp_size == 0, f"world_size ({world_size}) must be divisible by cp_size ({args.cp_size})"

    # ---- Set up context parallelism ----
    cp_group = None
    if args.cp_size > 1:
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(context_parallel_size=args.cp_size)
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = parallel_state.get_context_parallel_rank()
    else:
        dp_rank, dp_size = rank, world_size
        cp_rank = 0

    is_cp_leader = cp_rank == 0

    if args.output_dir is None:
        ckpt_stem = Path(args.dit_path).stem
        args.output_dir = (
            str(Path(args.prompt_json).parent / f"videos_{args.arch}_distilled_{args.num_steps}steps_{ckpt_stem}_seed{args.seed}")
            if is_distilled
            else str(
                Path(args.prompt_json).parent
                / f"videos_{args.arch}_cfg{args.guidance_scale}_shift{args.timestep_shift}_{args.num_steps}steps_{ckpt_stem}_seed{args.seed}"
            )
        )

    log.info(f"[Rank {rank}/{world_size}] arch={args.arch}, distilled={is_distilled}, cp_size={args.cp_size}, dp_rank={dp_rank}/{dp_size}")

    # ---- Load & partition prompts by data-parallel rank ----
    all_prompts = load_prompts(args.prompt_json, use_augmented=not args.no_augmented_prompt)
    my_prompts = partition_prompts(all_prompts, dp_rank, dp_size)
    log.info(f"[Rank {rank}] {len(my_prompts)}/{len(all_prompts)} prompts assigned (dp_rank={dp_rank})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-compute text embeddings ----
    log.info(f"[Rank {rank}] Encoding {len(my_prompts)} prompts ...")
    text_embs: list[torch.Tensor] = []
    for p in tqdm(my_prompts, desc=f"[R{rank}] Text enc", disable=(rank != 0)):
        emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=p["gen_prompt"])
        text_embs.append(emb.to(dtype=torch.bfloat16).cpu())

    neg_emb = None
    if not is_distilled:
        neg_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt)
        neg_emb = neg_emb.to(dtype=torch.bfloat16).cpu()

    clear_umt5_memory()
    log.info(f"[Rank {rank}] Text encoding done, freed encoder memory")

    # ---- Build & load DiT ----
    net = build_dit(args.arch, args.model_size)
    load_dit_weights(net, args.dit_path, args.arch)
    log.success(f"[Rank {rank}] Loaded DiT from {args.dit_path}")
    net.to(**TENSOR_KWARGS).cpu()
    torch.cuda.empty_cache()

    # ---- Enable context parallelism on the model ----
    if cp_group is not None and cp_group.size() > 1:
        net.enable_context_parallel(cp_group)
        log.info(f"[Rank {rank}] Context parallelism enabled (cp_size={args.cp_size})")

    # ---- Tokenizer & latent shape ----
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    # ---- Build timestep schedule ----
    ode = not is_distilled
    guidance = args.guidance_scale if ode else 1.0

    if is_distilled:
        mid_t = parse_rf_time_values(args.mid_t) if args.mid_t is not None else _DEFAULT_MID_T
        t_steps = build_distilled_t_steps(args.num_steps, args.sigma_max, mid_t, device)
        if args.steps_per_chunk and not is_causal:
            raise ValueError("--steps_per_chunk requires --arch causal")
        if args.context_from_last_step and not is_causal:
            raise ValueError("--context_from_last_step requires --arch causal")
        if is_causal:
            num_blocks = 1 + (state_shape[1] - args.first_chunk_t) // args.chunk_t
            step_counts = expand_steps_per_chunk(args.steps_per_chunk, num_blocks, args.num_steps)
            t_steps_per_chunk = build_distilled_schedules_per_chunk(
                step_counts, args.steps_per_chunk, args.sigma_max, mid_t, parse_mid_t_schedules(args.mid_t_schedules), device
            )
        else:
            step_counts = None
            t_steps_per_chunk = None
    else:
        if args.steps_per_chunk:
            raise ValueError("--steps_per_chunk is only supported with --distilled")
        t_steps = build_ode_t_steps(args.num_steps, args.sigma_max, args.timestep_shift, device)
        step_counts = None
        t_steps_per_chunk = None

    log.info(f"Mode: {'distilled' if is_distilled else 'ODE'}, steps={len(t_steps)-1}, guidance={guidance}")
    if step_counts is not None and args.steps_per_chunk:
        log.info(f"Per-chunk distilled steps: {step_counts}")

    # ---- Sampling loop ----
    net.cuda()
    eval_info_list: list[dict] = []
    pbar = tqdm(total=len(my_prompts), desc=f"[R{rank}] Sampling", disable=(rank != 0))

    for p_idx, prompt_info in enumerate(my_prompts):
        orig_prompt = prompt_info["prompt"]
        fname_base = sanitize_filename(orig_prompt)

        video_paths = [str(output_dir / f"{fname_base}-{i}.mp4") for i in range(args.num_samples)]
        all_exist = all(os.path.exists(p) for p in video_paths)

        if not all_exist:
            cond_emb = text_embs[p_idx].to(**TENSOR_KWARGS)
            condition = {"crossattn_emb": cond_emb}

            uncondition = None
            if neg_emb is not None:
                uncondition = {"crossattn_emb": neg_emb.to(**TENSOR_KWARGS)}

            generator = torch.Generator(device=device)

            for i in range(args.num_samples):
                if os.path.exists(video_paths[i]):
                    continue
                generator.manual_seed(args.seed + i)
                init_noise = torch.randn(1, *state_shape, dtype=torch.float32, device=device, generator=generator)

                # Sync noise across CP group so all ranks see identical input
                if cp_group is not None:
                    from rcm.utils.context_parallel import broadcast

                    init_noise = broadcast(init_noise, cp_group)

                if is_causal:
                    samples = sample_causal(
                        net,
                        init_noise,
                        t_steps,
                        t_steps_per_chunk,
                        args.steps_per_chunk,
                        condition,
                        uncondition,
                        guidance,
                        args.first_chunk_t,
                        args.chunk_t,
                        ode=ode,
                        generator=generator,
                        cp_group=cp_group,
                        context_from_last_step=args.context_from_last_step,
                        context_from_last_step_start_chunk=args.context_from_last_step_start_chunk,
                    )
                else:
                    samples = sample_bidirectional(
                        net,
                        init_noise,
                        t_steps,
                        condition,
                        uncondition,
                        guidance,
                        ode=ode,
                        generator=generator,
                    )

                # Only the CP leader saves videos (all ranks have identical output)
                if is_cp_leader:
                    video = tokenizer.decode(samples)  # [1, C, T, H, W]
                    video = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
                    save_video(video[0], video_paths[i], fps=args.fps)
                    del video

                del samples, init_noise
                torch.cuda.empty_cache()

        if is_cp_leader:
            entry = {
                "prompt_en": orig_prompt,
                "dimension": prompt_info["dimensions"],
                "video_list": video_paths,
            }
            if "auxiliary_info" in prompt_info:
                entry["auxiliary_info"] = prompt_info["auxiliary_info"]
            eval_info_list.append(entry)

        pbar.update(1)

    pbar.close()
    net.cpu()
    torch.cuda.empty_cache()

    if not is_cp_leader:
        distributed.barrier()
        return

    # ---- Save per-dp-rank eval info ----
    rank_path = output_dir / f"vbench_eval_info_rank{dp_rank}.json"
    with open(rank_path, "w") as f:
        json.dump(eval_info_list, f, indent=2, ensure_ascii=False)

    distributed.barrier()

    # ---- dp_rank 0: merge eval info ----
    if dp_rank == 0:
        merged = []
        for r in range(dp_size):
            rp = output_dir / f"vbench_eval_info_rank{r}.json"
            if rp.exists():
                with open(rp) as f:
                    merged.extend(json.load(f))
                rp.unlink()
        merged_path = output_dir / "vbench_eval_info.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        log.success(f"Done! {len(merged)} prompts, eval info: {merged_path}")


if __name__ == "__main__":
    main()
