# Usage: bash scripts/download_sdxl.sh <CHECKPOINT_PATH>
# SDXL base model, DMD2's LAION training data and COCO-10k evaluation set, DMD2's ODE-pretrained one-step model
# (initialization of the 1-step runs), and our teacher-sample latents.
CHECKPOINT_PATH=$1
HF=https://huggingface.co/tianweiy/DMD2/resolve/main

# SDXL base (fp32 weights; the training code loads it from this local folder)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir $CHECKPOINT_PATH/stable-diffusion-xl-base-1.0 \
    --include "model_index.json" "scheduler/*" "tokenizer/*" "tokenizer_2/*" "text_encoder/config.json" "text_encoder/model.safetensors" \
    "text_encoder_2/config.json" "text_encoder_2/model.safetensors" "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
    "vae/config.json" "vae/diffusion_pytorch_model.safetensors"

# training prompts
wget "$HF/data/laion/captions_laion_score6.25.pkl?download=true" -O $CHECKPOINT_PATH/captions_laion_score6.25.pkl

# real latents (T), packed into an LMDB
mkdir -p $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k
for INDEX in {0..59}; do
    INDEX_PADDED=$(printf "%03d" $INDEX)
    wget "$HF/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt?download=true" \
        -O "$CHECKPOINT_PATH/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt"
done
python main/data/create_lmdb_iterative.py --data_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k/ \
    --lmdb_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb

# teacher-sample latents (Q). TODO: HF link. To generate them instead, see main/sdxl/generate_teacher_latents.py.
# wget "<HF_PLACEHOLDER>/sdxl_teacher_latents_lmdb.zip" -O $CHECKPOINT_PATH/sdxl_teacher_latents_lmdb.zip
# unzip $CHECKPOINT_PATH/sdxl_teacher_latents_lmdb.zip -d $CHECKPOINT_PATH

# COCO-10k evaluation prompts and reference images
wget "$HF/data/coco/coco10k.zip?download=true" -O $CHECKPOINT_PATH/coco10k.zip
unzip $CHECKPOINT_PATH/coco10k.zip -d $CHECKPOINT_PATH

# DMD2's ODE-pretrained one-step generator (initialization of the 1-step runs)
wget "$HF/model/sdxl/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399_checkpoint_model_002000.bin?download=true" \
    -O $CHECKPOINT_PATH/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399_checkpoint_model_002000.bin
