# ComfyUI-DMAD

Custom nodes that run the DMAD 4-step students of MiniMax-H3 in ComfyUI the way they were trained. ComfyUI's stock
`lcm` sampler with the `simple` scheduler (after `ModelSamplingMiniMaxH3`) implements the same rule and sigma grid and
gives bit-identical videos, so these nodes are a convenience; either works.

* **DMAD Sampler (re-noise)** — a `SAMPLER` for `SamplerCustom` / `SamplerCustomAdvanced`. At every step the model
  predicts the clean sample x0 and the next input is x0 re-noised with *fresh* noise at the next sigma
  (x' = (1 − σ') x0 + σ' ε), the DMD-style multistep rule the students were trained with (the same update as `lcm`). ODE samplers
  (euler, ...) step from x_t instead, which is not the students' operating point.
* **DMAD Sigmas** — the students' sigma grid: shifted linear from 1 to 0, shift 12 (video), `steps` model evaluations.

The ComfyUI-layout LoRAs, [`minimax_h3/dmad_minimax_h3_4step_{lora_critic,full_critic}_comfyui.safetensors`](https://huggingface.co/ZhengmingYu/DMAD/tree/main/minimax_h3)
on the Hugging Face repo, are exact conversions of the released LoRAs (rank 128; q/k/v fused block-diagonally into
`qkv_proj`, SwiGLU halves reordered; no merging or rank reduction), made with
[`tools/convert_lora_comfyui.py`](../../tools/convert_lora_comfyui.py).

## Install

Copy or symlink this folder into `ComfyUI/custom_nodes/`:

```bash
cp -r comfyui/ComfyUI-DMAD /path/to/ComfyUI/custom_nodes/
hf download ZhengmingYu/DMAD --include "minimax_h3/*_comfyui.safetensors" --local-dir /path/to/ComfyUI/models/loras   # -> models/loras/minimax_h3/
```

The base model, text encoder and VAEs are the
[Comfy-Org MiniMax-H3 repackage](https://huggingface.co/Comfy-Org/MiniMax-H3) (ComfyUI loads the `fl2va` or `ref2va`
partition; the students were trained on the text-to-audio-video transformer, see the note below).

## Workflow (text-to-audio-video)

```
UNETLoader (minimax_h3_fl2va_*.safetensors)
  -> LoraLoaderModelOnly (dmad_minimax_h3_4step_lora_critic_comfyui.safetensors, strength 1.0)
  -> ModelSamplingMiniMaxH3 (shift_video 12.0, shift_audio 2.0)
  -> BasicGuider (conditioning from MiniMax H3 Image to Video / CLIPTextEncode with the minimax CLIP)
SamplerCustomAdvanced: noise = RandomNoise, guider as above, sampler = DMAD Sampler, sigmas = DMAD Sigmas (steps 4, shift 12)
                       (or sampler = KSamplerSelect lcm, sigmas = BasicScheduler simple, 4 steps: identical result),
                       latent_image = Empty MiniMax H3 AV Latent (1344 x 768, 124 frames)
  -> VAEDecode (video VAE) + VAEDecodeAudio (audio VAE) -> Create Video (24 fps) -> Save Video
```

Settings that matter: LoRA strength **1.0** (the LoRA is the distilled student, not a style), **cfg 1.0** (no
classifier-free guidance; H3 is guidance-distilled), `shift_audio` **2.0** (ComfyUI's H3 default is 3.0; the students
were trained with 2.0), **4 steps** with the DMAD Sigmas node. The students are trained in continuous time, so the
same sampler also supports more steps (`steps` in DMAD Sigmas; sampling time scales linearly): on prompts with fast
motion, 8 and 12 steps render the fast movements cleaner than 4-step.

## Re-noise sampling vs an ODE sampler

Measured in ComfyUI 0.38 (Comfy-Org `fl2va_bf16`, `full_critic` LoRA, 7 high-motion prompts, seed 42), mean
Laplacian variance of the frames as a sharpness proxy, DMAD sampler (= `lcm`) / euler at the same sigmas:

| steps | 2 | 4 | 6 | 8 | 10 | 12 | 25 |
|---|---|---|---|---|---|---|---|
| sharpness ratio | 1.02 | 1.09 | 1.09 | 1.18 | 1.13 | 1.15 | 1.12 |
| prompts where DMAD is sharper | 4/7 | 6/7 | 5/7 | 5/7 | 4/7 | 5/7 | 6/7 |

Visually, euler smears the fastest-moving limbs (a swordsman mid-strike, boxing gloves) where the re-noise sampler
keeps them crisp; the composition is the same with the same seed.

## Note on the base partition

ComfyUI ships the `fl2va` / `ref2va` partitions of MiniMax-H3 and uses them for text-to-video as well; the DMAD
students were trained on the `transformer/` (text-to-audio-video) partition of the original release. The LoRA loads
on all partitions (same architecture); results on `fl2va` / `ref2va` are what you get in ComfyUI, not the exact videos
of `inference.py`.
