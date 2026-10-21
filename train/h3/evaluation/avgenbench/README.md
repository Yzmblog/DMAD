# AVGen-Bench Evaluation


## 1. Generation settings

| | |
|---|---|
| steps | 4 (`inference.py` default; not `--euler`) |
| time shift | 12 (video), 2 (audio); no classifier-free guidance |
| size | 1344x768, 124 frames at 24 fps (5.2 s), stereo audio |
| seed | 42 for every prompt (`--seed 42`, the default) |

## 2. Prompts

[AVGen-Bench](https://github.com/microsoft/AVGen-Bench) has 235 prompts in 11 categories. We rewrote every prompt into the structured format of the MiniMax-H3 prompt-writing guide (`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` of the MiniMax-H3 repository): an `integrated_multimodal_description:`
shot list with timestamps within the 5 s, speaker IDs and the dialogue verbatim in `<d>[English] ...</d>`, double-quoted on-screen text, then `overall_soundscape:` and `non_diegetic_music:`.

`prompts.jsonl` holds them in the benchmark's order, one line per prompt:

```
{"index": 0, "category": "ads", "content": <the benchmark's title>, "prompt": <our structured rewrite>, "prompt_original": <the benchmark's prompt>}
```

`inference.py` takes the `prompt` field of each line, newlines included, and encodes it verbatim.

## 3. Generate and lay out the videos

From the repository root:

```bash
python inference.py --model-dir $H3_MODEL_DIR --lora ckpt/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
    --prompt-file train/h3/evaluation/avgenbench/prompts.jsonl --seed 42 --output-dir outputs/avgenbench_235_lora_critic

python inference.py --model-dir $H3_MODEL_DIR --lora ckpt/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors \
    --prompt-file train/h3/evaluation/avgenbench/prompts.jsonl --seed 42 --output-dir outputs/avgenbench_235_full_critic

# <i:04d>.mp4 -> AVGen-Bench's <videos-dir>/<category>/<safe_filename(content)>.mp4, and the prompt lists the judge reads
python train/h3/evaluation/avgenbench/layout.py train/h3/evaluation/avgenbench/prompts.jsonl outputs/avgenbench_235_lora_critic \
    /path/to/AVGen-Bench/generated_videos/dmad_lora_critic --prompts-dir /path/to/AVGen-Bench/prompts_h3pe

python train/h3/evaluation/avgenbench/layout.py train/h3/evaluation/avgenbench/prompts.jsonl outputs/avgenbench_235_full_critic \
    /path/to/AVGen-Bench/generated_videos/dmad_full_critic --prompts-dir /path/to/AVGen-Bench/prompts_h3pe
```

## 4. Evaluation

We ran the official AVGen-Bench modules with the arguments of `run_full_evaluation.sh` (the speech module with
`--compute_type int8`) with the following adaptation:

1. **The judge reads the h3pe prompt**, i.e. the text the video was generated from: `--prompts-dir` is the
   `prompts_h3pe` directory written above (the official prompt lists with the `prompt` field replaced,
   `content` unchanged). Every method in our comparison was generated from the same rewritten prompts and judged
   against them.
2. **Qwen3-Omni-Thinking instead of Gemini as the judge.** The five judge-based dimensions (Hi-Phy, Holistic, Text,
   Speech, Music) call Gemini in the official code. We served
   [`Qwen/Qwen3-Omni-30B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) with vLLM 0.29.0 as an OpenAI-compatible server and pointed
   the judge modules at it; the official prompts of the modules are unchanged.
   * server: `vllm serve ... --dtype bfloat16 --max-model-len 65536 --reasoning-parser qwen3 --media-io-kwargs '{"video": {"fps": 2}}'`
   * request: `temperature 0.6, top_p 0.95, max_tokens 16384` (thinking on); one video per request, sent as the
     native mp4 (`video_url` data URL) plus its audio track, downmixed to a mono 16 kHz wav, as an `input_audio`
     part, preceded by a one-line text note that the video follows, then its complete audio track, and that both are
     to be judged; the modules' own prompts are unchanged
   * a reply that returns the output template's placeholder (score 0 with confidence 0) is treated as a judge
     failure and left out of the mean, like a parse failure

The six other modules (Q-Align, Audiobox Aesthetics, Synchformer, SyncNet, InsightFace, VideoPhy-2) are the official
ones. The official total is `0.2 * mean(Visual, Audio) + 0.2 * mean(AV sync, Lip sync) + 0.6 * mean(7 fine-grained dims)`.
