#!/usr/bin/env bash
# DMAD SDXL 1-step with a frozen critic backbone (only the two heads train, 10x critic learning rate), initialized
# from DMD2's ODE-pretrained one-step model (released checkpoint: iteration 17500, EMA).
: "${CHECKPOINT_PATH:?}"
USE_ORIG_PARAMS=1 RUN_NAME=${RUN_NAME:-dmad_sdxl_1step_frozencritic} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 26001 --guidance_lr 5e-6 --warmup_step 500 --conditioning_timestep 399 --critic_freeze_backbone \
    --generator_ckpt_path $CHECKPOINT_PATH/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399_checkpoint_model_002000.bin "$@"
