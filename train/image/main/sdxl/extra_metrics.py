"""PickScore / HPSv2 / ImageReward scores with weights loaded from local paths.

Each follows the metric's official scoring code:
  - PickScore: yuvalkirstain/PickScore_v1 example (logit-scaled cosine similarity, mean).
  - HPSv2: tgxs002/HPSv2 img_score.py (open_clip ViT-H-14 + HPS checkpoint, cosine similarity under autocast, mean);
    works for both the v2 and v2.1 checkpoints.
  - ImageReward: the loop of main/coco_eval/coco_evaluator.compute_image_reward with local weights.
images: uint8 NHWC array; captions: list of strings.
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def compute_pick_score(images, captions, model_dir, processor_dir, device, batch_size=32):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_dir)
    model = AutoModel.from_pretrained(model_dir).eval().to(device)

    scores = []
    for start in tqdm(range(0, len(images), batch_size), desc="PickScore"):
        pils = [Image.fromarray(im) for im in images[start:start + batch_size]]
        texts = captions[start:start + batch_size]

        image_inputs = processor(images=pils, return_tensors="pt").to(device)
        text_inputs = processor(
            text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)

        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        batch_scores = model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        scores.append(batch_scores.cpu())

    del model
    torch.cuda.empty_cache()
    return float(torch.cat(scores).mean())


@torch.no_grad()
def compute_hps_v2(images, captions, ckpt_path, device, batch_size=32):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=None, precision="amp", device=device
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device).eval()

    scores = []
    for start in tqdm(range(0, len(images), batch_size), desc=f"HPSv2({ckpt_path.split('/')[-1]})"):
        pils = [Image.fromarray(im) for im in images[start:start + batch_size]]
        texts = captions[start:start + batch_size]

        image_batch = torch.stack([preprocess(p) for p in pils]).to(device)
        text_batch = tokenizer(texts).to(device)

        with torch.cuda.amp.autocast():
            image_features = model.encode_image(image_batch, normalize=True)
            text_features = model.encode_text(text_batch, normalize=True)
            hps_score = (image_features * text_features).sum(dim=-1)
        scores.append(hps_score.float().cpu())

    del model
    torch.cuda.empty_cache()
    return float(torch.cat(scores).mean())


def compute_image_reward(images, captions, model_path, med_config, device):
    import ImageReward as RM

    model = RM.load(model_path, device=device, med_config=med_config)
    rewards = []
    for image, prompt in tqdm(zip(images, captions), total=len(images), desc="ImageReward"):
        rewards.append(model.score(prompt, Image.fromarray(image)))
    del model
    torch.cuda.empty_cache()
    return float(np.mean(np.array(rewards)))
