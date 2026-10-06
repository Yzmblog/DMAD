#!/usr/bin/env bash
# DMAD ImageNet-64, teacher-UNet critic with a frozen backbone (heads train, critic lr 2e-5), spectral norm + gap
# routing (FID 1.20 at 221k iterations).
RUN_NAME=${RUN_NAME:-dmad_imagenet_frozencritic} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 600001 --guidance_lr 2e-5 --teacher_loss_weight 1e-4 \
    --critic_freeze_backbone --critic_spectral_norm --critic_max_timestep 500 --teacher_gap_route --gap_tau 0.5 "$@"
