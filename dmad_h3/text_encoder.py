"""Prompt encoding with the MiniMax-H3 conditioner (Qwen3-VL, hidden layer 50): the prompt verbatim, no chat template."""

import json
from pathlib import Path

import torch

TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def read_prompts(path):
    """A UTF-8 .txt file (one prompt per line) or .jsonl (one {"prompt": ...} object per line)."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        if str(path).endswith(".jsonl"):
            prompts = [json.loads(line)["prompt"] for line in handle if line.strip()]
        else:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise RuntimeError(f"No non-empty prompts found in {path}.")
    return prompts


def load_conditioner(model_dir, device, dtype=torch.bfloat16):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    root = Path(model_dir).expanduser().resolve()
    for sub in ("text_encoder", "tokenizer", "processor"):
        if not (root / sub).exists():
            raise FileNotFoundError(f"Missing MiniMax-H3 component: {root / sub}")
    load_kwargs = {"dtype": dtype, "local_files_only": True}
    if str(device) != "cpu":
        load_kwargs["device_map"] = {"": str(device)}
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(root / "text_encoder"), **load_kwargs)
    text_encoder.requires_grad_(False).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(root / "processor"), local_files_only=True)
    return text_encoder, tokenizer, processor


@torch.inference_mode()
def encode_prompt(text_encoder, tokenizer, processor, prompt, device, output_dtype=torch.bfloat16):
    """-> {"prompt_embeds": [1, tokens, 5120] on CPU, "text_token_tags": [tokens] long}."""
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("The tokenizer produced no tokens for a non-empty prompt.")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_hidden_states=True,
    )
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER].to(dtype=output_dtype).cpu()
    text_token_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    return {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}
