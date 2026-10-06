#!/usr/bin/env bash
# Shared DMAD SDXL launcher (one node, 8 GPUs, FSDP). The per-setting scripts add their arguments and call this.
#   CHECKPOINT_PATH : folder with stable-diffusion-xl-base-1.0, captions_laion_score6.25.pkl,
#                     sdxl_vae_latents_laion_500k_lmdb and sdxl_teacher_latents_lmdb (scripts/download_sdxl.sh)
#   OUTPUT_PATH     : training outputs;   WANDB_ENTITY / WANDB_PROJECT : wandb
#   USE_ORIG_PARAMS : 1 for FSDP use_orig_params (required with --critic_freeze_backbone)
set -euo pipefail
: "${CHECKPOINT_PATH:?}" "${OUTPUT_PATH:?}" "${RUN_NAME:?}"
FSDP_DIR=$OUTPUT_PATH/$RUN_NAME/fsdp_config
python main/sdxl/create_sdxl_fsdp_configs.py --folder $FSDP_DIR --master_ip 127.0.0.1 --num_machines 1 \
    --sharding_strategy 4 --use_orig_params ${USE_ORIG_PARAMS:-0}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file $FSDP_DIR/config_rank0.yaml --main_process_port ${MASTER_PORT:-2345} main/train_sd.py \
    --model_id $CHECKPOINT_PATH/stable-diffusion-xl-base-1.0 \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb \
    --teacher_latent_path $CHECKPOINT_PATH/sdxl_teacher_latents_lmdb \
    --output_path $OUTPUT_PATH/$RUN_NAME/ckpt --cache_dir $OUTPUT_PATH/$RUN_NAME/cache --log_path $OUTPUT_PATH/$RUN_NAME/log \
    --batch_size 8 --seed 10 --use_fp16 --gradient_checkpointing \
    --generator_lr 5e-7 --adam_beta1 0 --adam_beta2 0.99 --max_grad_norm 10 \
    --teacher_loss_weight 5e-3 --real_loss_weight 5e-3 --critic_loss_weight 1e-2 \
    --critic_max_timestep 1000 --critic_spectral_norm --generator_spectral_norm --teacher_gap_route --gap_tau 0.5 \
    --generator_ema_kimg 1000 --generator_ema_rampup 0.05 \
    --log_iters 500 --max_checkpoint 20 --wandb_iters 100 \
    --wandb_entity "${WANDB_ENTITY:-}" --wandb_project "${WANDB_PROJECT:-dmad-sdxl}" --wandb_name $RUN_NAME \
    "$@"
