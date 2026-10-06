"""COCO-10k evaluation of one SDXL generator checkpoint (single GPU).

Protocol as in DMD2 (main/sdxl/test_folder_sdxl.py upstream): 10k COCO-2014 prompts, per-batch noise from
torch.Generator().manual_seed(batch_index), batch size 4, seed 10, fp32; FID and patch FID against the COCO-10k
reference images at 512px, CLIP score with ViT-g/14. Optional PickScore, HPSv2 / HPSv2.1 and ImageReward from local
weight files. Results are written to --result_path (JSON).

    4-step generator:  --num_step 4 --conditioning_timestep 999
    1-step generator:  --num_step 1 --conditioning_timestep 399
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from main.coco_eval.coco_evaluator import compute_clip_score, evaluate_model
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import SDTextDataset


def fold_spectral_norm(state_dict):
    """Fold frozen-gain spectral-norm parametrizations (--generator_spectral_norm) into plain conv weights, computing
    exactly what the first forward after loading would: v' = normalize(W^T u), u' = normalize(W v'),
    sigma = u'^T W v', weight = W * gain / sigma."""
    marker = ".parametrizations.weight.original"
    prefixes = [k[: -len(marker)] for k in state_dict if k.endswith(marker)]
    if not prefixes:
        return state_dict
    folded = {k: v for k, v in state_dict.items() if ".parametrizations.weight." not in k}
    for p in prefixes:
        w = state_dict[f"{p}.parametrizations.weight.original"].float()
        u = state_dict[f"{p}.parametrizations.weight.0.u"].float()
        gain = state_dict[f"{p}.parametrizations.weight.0.gain"].float()
        w2 = w.reshape(w.shape[0], -1)
        v = torch.nn.functional.normalize(w2.t() @ u, dim=0)
        u = torch.nn.functional.normalize(w2 @ v, dim=0)
        sigma = torch.sum(u * (w2 @ v)).clamp(min=1e-12)
        folded[f"{p}.weight"] = w * (gain / sigma)
    return folded


def build_condition_input(resolution, device):
    add_time_ids = [resolution, resolution, 0, 0, resolution, resolution]  # original size, crop top-left, target size
    return torch.tensor([add_time_ids], device=device, dtype=torch.float32)


def get_x0_from_noise(sample, model_output, timestep, alphas_cumprod):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    return (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)


@torch.no_grad()
def sample(noise, unet_added_conditions, model, vae, noise_scheduler, prompt_embed, device="cuda", num_step=1,
           conditioning_timestep=999):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    if num_step == 1:
        all_timesteps = [conditioning_timestep]
        step_interval = 0
    elif num_step == 4:
        all_timesteps = [999, 749, 499, 249]
        step_interval = 250
    else:
        raise NotImplementedError()

    DTYPE = prompt_embed.dtype
    for constant in all_timesteps:
        current_timesteps = torch.ones(len(prompt_embed), device=device, dtype=torch.long) * constant
        eval_images = model(noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions).sample
        eval_images = get_x0_from_noise(noise, eval_images, current_timesteps, alphas_cumprod).float()

        next_timestep = current_timesteps - step_interval
        noise = noise_scheduler.add_noise(eval_images, torch.randn_like(eval_images), next_timestep).to(DTYPE)

    eval_images = vae.decode(eval_images / vae.config.scaling_factor, return_dict=False)[0]
    eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return eval_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="generator weights, e.g. .../pytorch_model_ema.bin")
    parser.add_argument("--model_id", type=str, required=True, help="local stable-diffusion-xl-base-1.0 folder")
    parser.add_argument("--anno_path", type=str, required=True, help="coco10k/all_prompts.pkl")
    parser.add_argument("--ref_dir", type=str, required=True, help="coco10k/subset (reference images)")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--grid_path", type=str, help="optional 3x3 sample grid (PNG)")
    parser.add_argument("--num_step", type=int, default=4, choices=[1, 4])
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--image_resolution", type=int, default=1024)
    parser.add_argument("--eval_res", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4, help="changes the noise; keep 4 to reproduce")
    parser.add_argument("--total_eval_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--no_clip_score", action="store_true")
    parser.add_argument("--pickscore_dir", type=str, help="local yuvalkirstain/PickScore_v1")
    parser.add_argument("--pickscore_processor_dir", type=str, help="local laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--hps_v2_ckpt", type=str, help="local HPS_v2_compressed.pt")
    parser.add_argument("--hps_v2_1_ckpt", type=str, help="local HPS_v2.1_compressed.pt")
    parser.add_argument("--image_reward_path", type=str, help="local ImageReward.pt")
    parser.add_argument("--image_reward_med_config", type=str, help="local med_config.json")
    parser.add_argument("--revision", type=str)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    start_time = time.time()

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")
    assert accelerator.num_processes == 1, "run as a single-GPU process"
    device = accelerator.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    text_encoder = SDXLTextEncoder(args, accelerator)
    tokenizer_one = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False)
    # as in DMD2's evaluator, the second tokenizer is also loaded from "tokenizer" (kept for reproducibility)
    tokenizer_two = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False)
    dataset = SDTextDataset(anno_path=args.anno_path, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, is_sdxl=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, num_workers=8)
    base_add_time_ids = build_condition_input(args.image_resolution, device)

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device).float()
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    generator = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()
    generator.requires_grad_(False)
    state_dict = fold_spectral_norm(torch.load(args.checkpoint_path, map_location="cpu"))
    print(generator.load_state_dict(state_dict, strict=True))
    del state_dict
    generator = generator.to(device)

    set_seed(args.seed + accelerator.process_index)

    all_images, all_captions = [], []
    for i, data in tqdm(enumerate(dataloader), total=len(dataset) // args.eval_batch_size):
        all_captions.append(data["key"])
        noise = torch.randn(
            len(data["key"]), 4, args.latent_resolution, args.latent_resolution,
            dtype=torch.float32, generator=torch.Generator().manual_seed(i),
        ).to(device)
        prompt_embed, pooled_prompt_embed = text_encoder(data)
        unet_added_conditions = {
            "time_ids": base_add_time_ids.repeat(noise.shape[0], 1),
            "text_embeds": pooled_prompt_embed,
        }
        eval_images = sample(
            noise, unet_added_conditions, generator, vae, scheduler, prompt_embed, device=device,
            num_step=args.num_step, conditioning_timestep=args.conditioning_timestep,
        )
        all_images.append(eval_images.cpu().numpy())

    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples]
    all_captions = [caption for sublist in all_captions for caption in sublist]
    torch.cuda.empty_cache()

    fid = evaluate_model(args, device, all_images, patch_fid=False)
    patch_fid = evaluate_model(args, device, all_images, patch_fid=True)
    print(f"{args.checkpoint_path}: fid {fid} patch fid {patch_fid}")
    result = {
        "checkpoint": args.checkpoint_path,
        "fid": float(fid),
        "patch_fid": float(patch_fid),
        "num_images": int(len(all_images)),
        "protocol": {
            "num_step": args.num_step,
            "conditioning_timestep": args.conditioning_timestep,
            "eval_res": args.eval_res,
            "eval_batch_size": args.eval_batch_size,
            "total_eval_samples": args.total_eval_samples,
            "seed": args.seed,
        },
    }

    if not args.no_clip_score:
        result["clip_score"] = float(compute_clip_score(
            images=all_images, captions=all_captions, clip_model="ViT-G/14", device=device, how_many=args.total_eval_samples,
        ))
        print(f"clip score {result['clip_score']}")

    from main.sdxl.extra_metrics import compute_hps_v2, compute_image_reward, compute_pick_score
    if args.pickscore_dir:
        result["pick_score"] = compute_pick_score(all_images, all_captions, args.pickscore_dir, args.pickscore_processor_dir, device)
        print(f"pick score {result['pick_score']}")
    if args.hps_v2_ckpt:
        result["hps_v2"] = compute_hps_v2(all_images, all_captions, args.hps_v2_ckpt, device)
        print(f"hps v2 {result['hps_v2']}")
    if args.hps_v2_1_ckpt:
        result["hps_v2_1"] = compute_hps_v2(all_images, all_captions, args.hps_v2_1_ckpt, device)
        print(f"hps v2.1 {result['hps_v2_1']}")
    if args.image_reward_path:
        result["image_reward"] = compute_image_reward(all_images, all_captions, args.image_reward_path, args.image_reward_med_config, device)
        print(f"image reward {result['image_reward']}")
    result["elapsed_sec"] = round(time.time() - start_time, 1)

    if args.grid_path:
        res = args.image_resolution
        grid = np.zeros((3 * res, 3 * res, 3), dtype=np.uint8)
        for k, tile in enumerate(all_images[:9]):
            r, c = divmod(k, 3)
            grid[r * res:(r + 1) * res, c * res:(c + 1) * res] = tile
        os.makedirs(os.path.dirname(os.path.abspath(args.grid_path)), exist_ok=True)
        Image.fromarray(grid).save(args.grid_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.result_path}")


if __name__ == "__main__":
    main()
