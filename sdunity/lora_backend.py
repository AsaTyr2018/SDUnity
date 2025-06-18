import os
import json
from typing import Dict, List, Generator

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


class ImagePromptDataset(Dataset):
    """Simple dataset returning image tensors and prompts."""

    def __init__(self, img_dir: str, tags: Dict[str, List[str]], size: int = 512):
        self.img_dir = img_dir
        self.tags = {k: v for k, v in tags.items() if os.path.isfile(os.path.join(img_dir, k))}
        self.files = list(self.tags.keys())
        self.size = size

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int):  # type: ignore[override]
        fname = self.files[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, (self.size, self.size), method=Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) * 2.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        prompt = ", ".join(self.tags[fname])
        return tensor, prompt


def _load_tags(data_dir: str) -> Dict[str, List[str]]:
    proj_meta = os.path.join(os.path.dirname(data_dir), "project.json")
    if os.path.isfile(proj_meta):
        with open(proj_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {k: v for k, v in meta.get("tags", {}).items() if isinstance(v, list)}
    return {}


def train_lora(
    instance_dir: str,
    pretrained_model: str,
    output_dir: str,
    steps: int = 1000,
    learning_rate: float = 1e-4,
) -> Generator[str, None, None]:
    """Run a minimal LoRA training loop using Diffusers and PEFT."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, safety_checker=None
    )
    pipe.to(device)

    tags = _load_tags(instance_dir)
    dataset = ImagePromptDataset(instance_dir, tags)
    if len(dataset) == 0:
        yield "No training images found"
        return
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    unet = get_peft_model(pipe.unet, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_v", "to_k", "to_out"]))
    text_enc = get_peft_model(pipe.text_encoder, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    unet.train()
    text_enc.train()

    optim = torch.optim.AdamW(list(unet.parameters()) + list(text_enc.parameters()), lr=learning_rate)
    yield f"Training for {steps} steps"

    for step in range(int(steps)):
        try:
            img, prompt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            img, prompt = next(loader_iter)

        img = img.to(device)
        ids = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        latents = pipe.vae.encode(img).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        noisy_latents = latents + noise
        encoder_hidden_states = text_enc(ids)[0]
        noise_pred = unet(noisy_latents, torch.tensor([0]).to(device), encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        optim.step()
        optim.zero_grad()

        if (step + 1) % 10 == 0 or step == 0:
            yield f"Step {step+1}/{steps} - loss {loss.item():.4f}"

    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    text_enc.save_pretrained(os.path.join(output_dir, "text_encoder"))
    yield f"Training complete. LoRA saved to {output_dir}"
