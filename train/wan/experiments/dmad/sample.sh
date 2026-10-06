#!/usr/bin/env bash
# 4-step sampling with a DMAD Wan checkpoint, using the schedules of the reported VBench scores (8 GPUs).
# Usage: bash experiments/dmad/sample.sh <1.3B|14B> <model .pth> <prompt json> <output dir>
#   Convert a training checkpoint first: python scripts/dcp_to_pth_dmad.py --dcp_checkpoint_dir <iter>/model --save_path <.pth>
#   Prompt json: a list of {"prompt": ...} (optionally "augmented_prompt"); VBench: evaluation/vbench_text2video/prompts.json
#   with NUM_SAMPLES=5.
set -euo pipefail
SIZE=${1:?1.3B or 14B}; DIT=${2:?model .pth}; PROMPTS=${3:?prompt json}; OUT=${4:?output dir}
CKPT=${CKPT:-assets/checkpoints}   # Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth
case $SIZE in
    1.3B) SCHEDULE=(--sigma_max 320 --mid_t 0.93378 0.76 0.11) ;;
    14B)  SCHEDULE=(--sigma_max 320 --mid_t 0.84 0.11 0.065) ;;
    *) echo "unknown size $SIZE" >&2; exit 1 ;;
esac
export PYTHONPATH=.
torchrun --standalone --nproc_per_node=${NPROC:-8} evaluation/vbench_text2video/sample_videos.py --arch bidirectional \
    --distilled --num_steps 4 --model_size $SIZE --dit_path $DIT --prompt_json $PROMPTS --num_samples ${NUM_SAMPLES:-1} \
    --seed 0 --output_dir $OUT --vae_path $CKPT/Wan2.1_VAE.pth \
    --text_encoder_path $CKPT/models_t5_umt5-xxl-enc-bf16.pth "${SCHEDULE[@]}"
