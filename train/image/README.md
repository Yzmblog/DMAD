# DMAD on ImageNet-64 and SDXL

Code for the ImageNet-64x64 (EDM teacher) and SDXL experiments of DMAD. It is built on
[DMD2](https://github.com/tianweiy/DMD2) (and EDM), and keeps DMD2's layout, data and evaluation protocols.

DMAD replaces DMD2's fake score and single GAN classifier with a two-head critic:

* the **real head** separates real data (T) from generated samples (G);
* the **teacher head** separates teacher samples (Q) from generated samples (G).

The critic is trained with BCE on both heads. The generator minimizes `-d_teacher(G)` and `-d_real(G)`, both
linear losses. With `--teacher_gap_route`, the teacher term is re-weighted per noise band by the online gap
`E[d_real(T)] - E[d_real(Q)]`: bands where the teacher samples look less real than the data get less teacher
weight. The critic backbone is the encoder of a copy of the teacher UNet, with frozen-gain spectral norm.
Critic inputs are noised, and the T/G/Q samples of each index share one noise level. Generator encoders use the
same spectral norm. The ImageNet code also supports a frozen-backbone critic, and a critic built on frozen
pretrained feature networks with R1 (`pg_critic.py`).

Main files:

| | ImageNet-64 | SDXL |
| --- | --- | --- |
| critic | [main/edm/edm_guidance.py](main/edm/edm_guidance.py), [main/edm/pg_critic.py](main/edm/pg_critic.py) | [main/sd_guidance.py](main/sd_guidance.py) |
| generator + critic wrapper | [main/edm/edm_unified_model.py](main/edm/edm_unified_model.py) | [main/sd_unified_model.py](main/sd_unified_model.py) |
| training | [main/edm/train_edm.py](main/edm/train_edm.py) | [main/train_sd.py](main/train_sd.py) |
| teacher samples (Q) | [main/edm/generate_teacher_samples.py](main/edm/generate_teacher_samples.py) | [main/sdxl/generate_teacher_latents.py](main/sdxl/generate_teacher_latents.py) |
| evaluation | [main/edm/test_folder_edm.py](main/edm/test_folder_edm.py) | [main/sdxl/eval_sdxl.py](main/sdxl/eval_sdxl.py) |

## Environment Setup

```bash
conda create -n dmad python=3.8 -y
conda activate dmad
pip install -r requirements.txt
python setup.py develop
```

All results were produced with the versions pinned in `requirements.txt` (PyTorch 2.1.2, CUDA 12.1). ImageNet trains
in BF16 (`--use_fp16` selects BF16, as in DMD2).

## ImageNet-64x64

### Model Zoo

| Setting | Script | FID (50k) | Iterations | Checkpoint |
| --- | --- | --- | --- | --- |
| teacher-UNet critic + gap routing | [dmad_imagenet_gaproute.sh](experiments/imagenet/dmad_imagenet_gaproute.sh) | 1.24 | 568k | [link](https://huggingface.co/ZhengmingYu/DMAD/tree/main/imagenet64/dmad_imagenet_gaproute) |
| pretrained-feature critic | [dmad_imagenet_pgcritic.sh](experiments/imagenet/dmad_imagenet_pgcritic.sh) | 1.04 | 108k | [link](https://huggingface.co/ZhengmingYu/DMAD/tree/main/imagenet64/dmad_imagenet_pgcritic) |
| frozen teacher-UNet critic + gap routing | [dmad_imagenet_frozencritic.sh](experiments/imagenet/dmad_imagenet_frozencritic.sh) | 1.20 | 221k | [link](https://huggingface.co/ZhengmingYu/DMAD/tree/main/imagenet64/dmad_imagenet_frozencritic) |

FID is computed on the EMA generator (`pytorch_model_ema.bin`), one step, 50k samples.

### Data

```bash
export CHECKPOINT_PATH=""   # your data / model folder
export OUTPUT_PATH=""       # training outputs
export WANDB_ENTITY="" WANDB_PROJECT=""
mkdir -p $CHECKPOINT_PATH

# EDM teacher, Inception detector, FID reference statistics, ImageNet-64 LMDB (as in DMD2)
bash scripts/download_imagenet.sh $CHECKPOINT_PATH

# teacher samples (Q): 1M EDM samples, stochastic Heun sampler with 256 steps (S_churn 40, S_min 0.05, S_max 50,
# S_noise 1.003), seed 10 + index, class = index mod 1000; generated with 8 GPUs:
torchrun --nproc_per_node 8 main/edm/generate_teacher_samples.py --model_path $CHECKPOINT_PATH/edm-imagenet-64x64-cond-adm.pkl \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl \
    --output_dir $CHECKPOINT_PATH --run_name edm_teacher_samples --total_eval_samples 1000000
```

### Training and Evaluation

Training uses one node with 8 GPUs (48 images per GPU). FID runs as a separate single-GPU process that watches the
checkpoint cache, as in DMD2.

```bash
# training
bash experiments/imagenet/dmad_imagenet_gaproute.sh

# on another GPU: evaluate every checkpoint the run writes (primary EMA -> "fid", extra EMAs -> "fid_ema_K")
bash experiments/imagenet/eval_fid.sh dmad_imagenet_gaproute
```

`dmad_imagenet_gaproute.sh` also keeps EMAs with 8000 and 16000 kimg half-lives (`--generator_ema_kimg_list`). Each
is saved in every checkpoint as `pytorch_model_ema_{K}.bin` and scored by the FID process.

To evaluate a released checkpoint (each release folder holds one `checkpoint_model_<iteration>/pytorch_model_ema.bin`):

```bash
hf download ZhengmingYu/DMAD --include "imagenet64/dmad_imagenet_gaproute/*" --local-dir ckpt
python main/edm/test_folder_edm.py --folder ckpt/imagenet64/dmad_imagenet_gaproute --run_once \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl --wandb_mode disabled
```

## SDXL

### Model Zoo

| Setting | Script | Steps | FID | Patch FID | CLIP | Iterations | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-step, backward simulation | [dmad_sdxl_4step.sh](experiments/sdxl/dmad_sdxl_4step.sh) | 4 | 14.47 | 19.88 | 0.328 | 17k | [link](https://huggingface.co/ZhengmingYu/DMAD/blob/main/sdxl/dmad_sdxl_4step_unet_fp16.bin) |
| 1-step, ODE init | [dmad_sdxl_1step.sh](experiments/sdxl/dmad_sdxl_1step.sh) | 1 | 16.29 | 24.17 | 0.330 | 49k | [link](https://huggingface.co/ZhengmingYu/DMAD/blob/main/sdxl/dmad_sdxl_1step_unet_fp16.bin) |
| 1-step, ODE init, frozen critic backbone | [dmad_sdxl_1step_frozencritic.sh](experiments/sdxl/dmad_sdxl_1step_frozencritic.sh) | 1 | 16.11 | 32.42 | 0.338 | 17.5k | [link](https://huggingface.co/ZhengmingYu/DMAD/blob/main/sdxl/dmad_sdxl_1step_frozencritic_unet_fp16.bin) |

COCO-2014 10k prompts, EMA generator; FID and patch FID against the COCO-10k reference images, CLIP score with
ViT-g/14. [eval_sdxl.py](main/sdxl/eval_sdxl.py) also reports PickScore, HPSv2/v2.1 and ImageReward.

### Inference

The released UNets have the spectral norm folded in and load into the standard SDXL pipeline:

```python
import torch
from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(torch.load(hf_hub_download("ZhengmingYu/DMAD", "sdxl/dmad_sdxl_4step_unet_fp16.bin"), map_location="cuda"))
pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# 4-step model
image = pipe(prompt="a photo of a cat", num_inference_steps=4, guidance_scale=0, timesteps=[999, 749, 499, 249]).images[0]
# 1-step models (sdxl/dmad_sdxl_1step_unet_fp16.bin / sdxl/dmad_sdxl_1step_frozencritic_unet_fp16.bin)
# image = pipe(prompt="a photo of a cat", num_inference_steps=1, guidance_scale=0, timesteps=[399]).images[0]
```

### Data

```bash
export CHECKPOINT_PATH="" OUTPUT_PATH="" WANDB_ENTITY="" WANDB_PROJECT=""
# SDXL base, DMD2's LAION prompts and real latents (T), COCO-10k, DMD2's ODE-pretrained one-step generator,
# and our teacher latents (Q)
bash scripts/download_sdxl.sh $CHECKPOINT_PATH
```

The teacher latents are SDXL-base samples for the captions of the real-latent LMDB: 50-step Euler, CFG 8. To
regenerate them, see [generate_teacher_latents.py](main/sdxl/generate_teacher_latents.py).

### Training and Evaluation

All SDXL runs use one node with 8 GPUs (FSDP, 8 images per GPU).

```bash
bash experiments/sdxl/dmad_sdxl_4step.sh
bash experiments/sdxl/dmad_sdxl_1step_frozencritic.sh
bash experiments/sdxl/dmad_sdxl_1step.sh

# evaluate one checkpoint on one GPU (set METRIC_PATH to also compute PickScore / HPSv2 / ImageReward)
bash experiments/sdxl/eval_sdxl.sh <checkpoint folder>/pytorch_model_ema.bin 4step result.json
```

`METRIC_PATH` holds `PickScore_v1/` (yuvalkirstain/PickScore_v1), `CLIP-ViT-H-14-laion2B-s32B-b79K/`
(its processor files), `HPS_v2_compressed.pt`, `HPS_v2.1_compressed.pt` (xswu/HPSv2), and `ImageReward.pt` with
`med_config.json` (THUDM/ImageReward).

## License

This part of the repository is derived from DMD2 and EDM and is released under the same
[CC BY-NC-SA 4.0](LICENSE.md) license.

## Acknowledgements

This code builds on [DMD2](https://github.com/tianweiy/DMD2), [EDM](https://github.com/NVlabs/edm),
[clean-fid](https://github.com/GaParmar/clean-fid) and [GigaGAN](https://github.com/mingukkang/GigaGAN)'s evaluation code.
