#!/usr/bin/env bash
# DMAD ImageNet-64, frozen pretrained-feature critic (VGG16-BN + EfficientNet-lite0) on clean images, R1 = 1
# (FID 1.04 at 108k iterations). The torchvision / timm backbone weights are downloaded on first use.
RUN_NAME=${RUN_NAME:-dmad_imagenet_pgcritic} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 600001 --guidance_lr 2e-6 --teacher_loss_weight 1e-4 \
    --pretrained_critic vgg16_bn,tf_efficientnet_lite0 --critic_clean --r1_gamma 1.0 --r1_batch_size 16 "$@"
