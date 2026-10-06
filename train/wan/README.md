# DMAD on Wan2.1 T2V (480p)

This folder is [rCM](https://github.com/NVlabs/rcm) with DMAD added. Only the parts DMAD needs are kept. The rCM
distillation model is kept, but its DMD term is replaced by a two-head GAN critic, and the sCM loss is switched off.

* **Critic.** The fake-score network, a copy of the teacher, is followed by two per-token MLP heads on the token output
  of its last block. Their outputs are mean-pooled over tokens:
  * real head: real videos (T) vs generated videos (G);
  * teacher head: teacher samples (Q) vs generated videos (G).

  G, Q and T of one sample are noised at the same TrigFlow time, drawn from p_D. The critic is trained with softplus BCE
  on both heads.
* **Generator loss.** `-d_teacher(G) - d_real(G)`. The teacher term is weighted per noise band (the ten CDF deciles of
  p_D) by `sigmoid((median gap - gap_band) / tau)`, with `gap = E[d_real(T)] - E[d_real(Q)]` tracked online.
* **Spectral norm.** Frozen-gain spectral norm is applied to the critic's block and head Linears, and to the first half
  of the generator's blocks.
* **Updates.** Generator and critic updates alternate 1:1. The EMA is rCM's power EMA.

Code: [rcm/models/t2v_model_distill_dmad.py](rcm/models/t2v_model_distill_dmad.py) (model),
[rcm/configs/experiments/dmad/wan2pt1_t2v.py](rcm/configs/experiments/dmad/wan2pt1_t2v.py) (experiments),
[rcm/datasets/webdataset_dmad.py](rcm/datasets/webdataset_dmad.py) (data), [experiments/dmad/](experiments/dmad/) (scripts).

## Models

| Model | Experiment | GPUs | Iterations | VBench Total / Quality / Semantic | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Wan2.1-1.3B, 4 steps | `wan2pt1_1pt3B_res480p_t2v_dmad` | 8 | 17k | 84.70 / 85.98 / 79.55 | [link](https://huggingface.co/ZhengmingYu/DMAD/blob/main/wan2.1/dmad_wan2pt1_1pt3B.pth) |
| Wan2.1-14B, 4 steps | `wan2pt1_14B_res480p_t2v_dmad` | 64 | 19.5k | 85.15 / 85.96 / 81.90 | [link](https://huggingface.co/ZhengmingYu/DMAD/blob/main/wan2.1/dmad_wan2pt1_14B.pth) |

Both models use the EMA generator.

VBench protocol:
* 944 augmented prompts, 5 videos each, 81 frames, seed 0.
* Sampling schedules: 1.3B uses `--sigma_max 320 --mid_t 0.93378 0.76 0.11`; 14B uses `--sigma_max 320 --mid_t 0.84 0.11 0.065`.

## Installation

```bash
conda create -n dmad_wan python=3.12 -y
conda activate dmad_wan
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

transformer_engine, flash-attn and MagiAttention are optional: without them, AdamW with fp32 master weights and
PyTorch attention are used.

Download the Wan2.1 teacher checkpoints (`.pth`), the VAE, the umT5 text encoder and the negative-prompt embedding to
`assets/checkpoints`, then convert the teachers to DCP for training:

```bash
# make sure git lfs is installed
git clone https://huggingface.co/worstcoder/Wan assets/checkpoints
for m in 1.3B 14B; do
    python -m torch.distributed.checkpoint.format_utils torch_to_dcp \
        assets/checkpoints/Wan2.1-T2V-$m.pth assets/checkpoints/Wan2.1-T2V-$m.dcp
done
```

All scripts read the checkpoints from `assets/checkpoints`; set `CKPT=/other/folder` to change it.

## Inference

```bash
hf download ZhengmingYu/DMAD --include "wan2.1/*" --local-dir ckpt   # dmad_wan2pt1_1pt3B.pth (2.8 GB), dmad_wan2pt1_14B.pth (29 GB)
# one video per prompt of a JSON list [{"prompt": ...}, ...]
bash experiments/dmad/sample.sh 1.3B ckpt/wan2.1/dmad_wan2pt1_1pt3B.pth my_prompts.json outputs/samples_1pt3B
bash experiments/dmad/sample.sh 14B  ckpt/wan2.1/dmad_wan2pt1_14B.pth  my_prompts.json outputs/samples_14B
```

`sample.sh` uses 8 GPUs; set `NPROC` to change it. A training checkpoint converts to this `.pth` format (EMA
generator, spectral norm folded in) with
`python scripts/dcp_to_pth_dmad.py --dcp_checkpoint_dir <run>/checkpoints/iter_XXXXXXXXX/model --save_path model.pth`.

## Evaluation

```bash
# all 944 VBench prompts x 5 videos
NUM_SAMPLES=5 bash experiments/dmad/sample.sh 1.3B ckpt/wan2.1/dmad_wan2pt1_1pt3B.pth evaluation/vbench_text2video/prompts.json outputs/vbench_1pt3B
```

Then score `outputs/vbench_1pt3B` as described in
[evaluation/vbench_text2video/README.md](evaluation/vbench_text2video/README.md).

## Data

Each training row holds:
* a Wan2.1-14B teacher sample for an UltraVideo caption: 100-step Euler, CFG 5, shift 3, 81 frames, 480p;
* its T5 embedding and prompt;
* the latent of the UltraVideo clip that caption describes (`real_latent.pt`).

There are 42,158 rows in total.

```bash
# download the prepared shards
huggingface-cli download HF_PLACEHOLDER --repo-type dataset --local-dir $DATA/shards

# or build them: UltraVideo_short.csv + clips_short_960 from huggingface.co/datasets/APRIL-AIGC/UltraVideo
ULTRAVIDEO=/path/to/UltraVideo DATA=/path/to/data bash experiments/dmad/prepare_data.sh
```

## Training

```bash
DATA=/path/to/data bash experiments/dmad/train_1pt3B.sh                    # 1 node x 8 GPUs
DATA=/path/to/data NODE_RANK=<i> MASTER_ADDR=<host> bash experiments/dmad/train_14B.sh   # on each of 8 nodes
```

Both scripts launch with `torchrun`. Outputs go to `$IMAGINAIRE_OUTPUT_ROOT` (default `checkpoints/`). Set
`WANDB_API_KEY` and `WANDB_ENTITY` to log to wandb, or `WANDB_MODE=disabled` to turn it off.

## License and Acknowledgement

This folder is derived from [rCM](https://github.com/NVlabs/rcm) and is released under the same
[Apache 2.0](LICENSE.txt) license. rCM builds on [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

```bibtex
@article{zheng2025rcm,
  title={Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency},
  author={Zheng, Kaiwen and Wang, Yuji and Ma, Qianli and Chen, Huayu and Zhang, Jintao and Balaji, Yogesh and Chen, Jianfei and Liu, Ming-Yu and Zhu, Jun and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2510.08431},
  year={2025}
}
```
