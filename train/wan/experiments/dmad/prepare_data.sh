#!/usr/bin/env bash
# DMAD Wan training data (run from the repo root): UltraVideo prompts -> Wan2.1-14B teacher samples -> + real latents.
#   ULTRAVIDEO : folder with UltraVideo_short.csv and the extracted clips_short_960/ (huggingface.co/datasets/APRIL-AIGC/UltraVideo)
#   DATA       : output folder;  CKPT : assets/checkpoints (Wan2.1-T2V-14B.pth, Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth)
#   NUM_MACHINES / MACHINE_ID : split the teacher sampling over several 8-GPU machines (run step 2 on each)
# Each step resumes where it stopped.
set -euo pipefail
: "${ULTRAVIDEO:?}" "${DATA:?}"
CKPT=${CKPT:-assets/checkpoints}
export PYTHONPATH=.

# 1. de-duplicated prompts (42,158) and the prompt -> clip map
python rcm/datasets/build_prompt_list.py --csv $ULTRAVIDEO/UltraVideo_short.csv --output_dir $DATA

# 2. teacher samples: Wan2.1-14B, 100-step Euler, CFG 5, shift 3, 81 frames at 480p; seed = 0 + prompt index
torchrun --standalone --nproc_per_node=8 rcm/datasets/build_synthetic_dataset.py \
    --prompt_file $DATA/prompts.txt --machine_id ${MACHINE_ID:-0} --num_machines ${NUM_MACHINES:-1} \
    --output_dir $DATA/teacher_shards --samples_per_shard 8 --batch_size 1 --model_size 14B --num_steps 100 \
    --sigma_max 5000 --sampler Euler --guidance_scale 5.0 --timestep_shift 3.0 --dit_path $CKPT/Wan2.1-T2V-14B.pth \
    --vae_path $CKPT/Wan2.1_VAE.pth --text_encoder_path $CKPT/models_t5_umt5-xxl-enc-bf16.pth \
    --num_frames 81 --resolution 480p --aspect_ratio 16:9 --seed 0 --exit_after_done

# 3. real latents of the caption-paired UltraVideo clips, one process per GPU
ls $DATA/teacher_shards/shard_*.tar | sort > $DATA/teacher_shards.txt
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python rcm/datasets/build_real_latents.py \
        --teacher_shards $(awk -v i=$i 'NR % 8 == i' $DATA/teacher_shards.txt) --map_file $DATA/index_to_clipid.jsonl \
        --video_root $ULTRAVIDEO/clips_short_960 --output_dir $DATA/shards --vae_path $CKPT/Wan2.1_VAE.pth &
done
wait
