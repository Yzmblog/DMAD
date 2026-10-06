## DMAD on MiniMax-H3 (text-to-audio-video)

This directory distills [MiniMax-H3](https://huggingface.co/MiniMaxAI) (33B, joint video + audio generation) into a
4-step student with DMAD. The student and the critic are LoRAs on the H3 transformer. The critic has two heads on its
last-block features: one separates real clips from generated samples, the other separates teacher samples from
generated samples. The code is built on [LightX2V](https://github.com/ModelTC/LightX2V)'s training framework.

### Model Zoo

| Config | Critic | Steps | Resolution | Iters | Released student |
| ------ | ------ | ----- | ---------- | ----- | ---------------- |
| [dmad_minimax_h3](configs/dmad_minimax_h3.yaml) | LoRA (the paper) | 4 | 768x1344, 124 frames (5.2 s) + stereo audio | 800 | [`lora_critic`](https://huggingface.co/ZhengmingYu/DMAD) |
| [dmad_minimax_h3_full_critic](configs/dmad_minimax_h3_full_critic.yaml) | full, blocks 25-49 | 4 | same | 1600 | [`full_critic`](https://huggingface.co/ZhengmingYu/DMAD) |

`lora_critic` (the checkpoint of the paper) is the EMA (sigma_rel 0.10) of the student LoRA at iteration 800 of the
first run (`checkpoint-000000800/ema_lora/pytorch_lora_weights.safetensors`). `full_critic` is the live student LoRA
at iteration 1600 of the second run (`checkpoint-000001600/pytorch_lora_weights.safetensors`), which differs only in
the critic: instead of a LoRA, `fake.train_type: full` trains the critic's transformer blocks 25-49 (16.2B parameters)
and `critic_freeze_blocks: 25` freezes its token refiner, embedders and blocks 0-24. Training took about 6.7 hours
per 100 iterations with the LoRA critic and 5.5 hours with the full critic on 8 x H200.

### Environment Setup

```bash
conda create -n dmad_h3 python=3.12 -y && conda activate dmad_h3
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Download the MiniMax-H3 base model (diffusers layout, with `transformer/`, `vae/`, `audio_vae/`, `text_encoder/`,
`tokenizer/` and `processor/`) and set `H3_MODEL_DIR` to it.

### Inference

Inference lives at the repository root: [`inference.py`](../../inference.py) (the sampler of the paper, with the
`--low-vram` path for 24 GB GPUs) and [`run_diffusers_pipeline.py`](../../run_diffusers_pipeline.py); see the
[main README](../../README.md). A training checkpoint's `ema_lora/pytorch_lora_weights.safetensors` (or the live
`pytorch_lora_weights.safetensors`) is passed as `--lora`.

### Data Preparation

Training uses [UltraVideo](https://huggingface.co/datasets/APRIL-AIGC/UltraVideo) (`short.csv` and the
`clips_short_1920` zips) for both the prompts and the real clips. [scripts/prepare_data.sh](scripts/prepare_data.sh)
builds everything:

```bash
export H3_MODEL_DIR=/path/to/MiniMax-H3 ULTRAVIDEO_DIR=/path/to/UltraVideo DATA_ROOT=/path/to/dmad_h3_data
bash scripts/prepare_data.sh
```

| Step | Script | Output |
| ---- | ------ | ------ |
| 1 | `data_process/build_prompt_list.py` | caption list and caption-to-clip map |
| 2 | `data_process/encode_real_videos.py` | VAE latents of the real clips (24,005 clips; 17,912 at 124 frames) |
| 3 | `data_process/build_pool_indices.py` | the 17,902 captions used in training |
| 4 | `data_process/encode_prompts.py` | text-encoder cache of those captions |
| 5 | `data_process/gen_teacher_data.py` | one 31-step teacher sample per caption |

The preprocessed data is on Hugging Face: [ZhengmingYu/DMAD-H3-data](https://huggingface.co/datasets/ZhengmingYu/DMAD-H3-data)
(gated by the UltraVideo license terms, about 300 GB). Its folders are the four data paths of the config:
`prompt_cache/` (`prompt_cache_dir`), `real_latents/` (`real_latent_dir`), `teacher_latents/` (`teacher_latent_dir`) and
`ultravideo_prompts/ultravideo_index_to_clipid.jsonl` (`real_caption_map`).

```bash
hf download ZhengmingYu/DMAD-H3-data --repo-type dataset --local-dir /path/to/dmad_h3_data
```

### Training

Fill in the data paths in [configs/dmad_minimax_h3.yaml](configs/dmad_minimax_h3.yaml), then:

```bash
bash scripts/train.sh configs/dmad_minimax_h3.yaml
```

Checkpoints are written every 100 iterations to `training.output_dir/checkpoint-XXXXXXXXX/`:
`pytorch_lora_weights.safetensors` (live student LoRA), `ema_lora/` and `ema2_lora/` (EMA students, sigma_rel ~0.10 and
~0.05), `fake_lora/` (critic LoRA), `critic_state.pt` (critic heads, EMA state) and `dist_state/` (for resuming).
Each checkpoint also writes samples of the live and EMA students to `training.output_dir/samples/`.
Set `WANDB_MODE=online` to log to Weights & Biases.

### License

The code is released under the Apache 2.0 license (see [LICENSE](LICENSE)), inherited from LightX2V. The MiniMax-H3
weights and UltraVideo are subject to their own licenses.
