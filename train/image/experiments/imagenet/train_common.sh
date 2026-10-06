#!/usr/bin/env bash
# Shared DMAD ImageNet-64 launcher (one node, 8 GPUs). The per-setting scripts set CRITIC_ARGS / LOSS_ARGS and call this.
#   CHECKPOINT_PATH : folder with edm-imagenet-64x64-cond-adm.pkl, imagenet-64x64_lmdb and the teacher-sample LMDB
#   OUTPUT_PATH     : training outputs;   WANDB_ENTITY / WANDB_PROJECT : wandb
set -euo pipefail
: "${CHECKPOINT_PATH:?}" "${OUTPUT_PATH:?}" "${RUN_NAME:?}"
TEACHER_LMDB=${TEACHER_LMDB:-$CHECKPOINT_PATH/edm_teacher_samples_lmdb}
torchrun --nproc_per_node ${NPROC_PER_NODE:-8} main/edm/train_edm.py \
    --model_id $CHECKPOINT_PATH/edm-imagenet-64x64-cond-adm.pkl \
    --real_image_path $CHECKPOINT_PATH/imagenet-64x64_lmdb \
    --teacher_image_path $TEACHER_LMDB \
    --output_path $OUTPUT_PATH/$RUN_NAME \
    --cache_dir $OUTPUT_PATH/$RUN_NAME/cache --delete_ckpts --max_checkpoint 20 \
    --batch_size 48 --seed 10 --use_fp16 --num_workers 3 \
    --generator_lr 2e-6 --adam_beta1 0 --adam_beta2 0.99 --warmup_step 500 \
    --real_loss_weight 3e-3 --generator_spectral_norm \
    --generator_ema_kimg 1000 --generator_ema_rampup 0.05 \
    --log_iters 500 --wandb_iters 100 \
    --wandb_entity "${WANDB_ENTITY:-}" --wandb_project "${WANDB_PROJECT:-dmad-imagenet}" --wandb_name $RUN_NAME \
    "$@"
