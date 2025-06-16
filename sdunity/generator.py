import random
import inspect
import io
import imageio
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import gradio as gr

from . import presets, models, gallery


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    model: str,
    lora,
    nsfw_filter: bool,
    images_per_batch: int,
    batch_count: int,
    preset: str,
    smooth_preview: bool,
    progress=gr.Progress(),
) -> Tuple[Image.Image, int, bytes]:
    """Generate one or more images using the selected diffusion model."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    images_per_batch = max(1, int(images_per_batch))
    batch_count = max(1, int(batch_count))

    if preset:
        enhancement = presets.PRESETS.get(preset)
        if enhancement:
            prompt = f"{prompt}, {enhancement}"

    pipe = models.get_pipeline(model, progress=progress)

    if not hasattr(pipe, "_original_safety_checker"):
        pipe._original_safety_checker = getattr(pipe, "safety_checker", None)
    pipe.safety_checker = pipe._original_safety_checker if nsfw_filter else None

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    images = []
    preview_frames = [] if smooth_preview else None

    def _decode_preview_latents(latents):
        if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
            return pipe.image_processor.postprocess(latents, output_type="pil")
        if hasattr(pipe, "vae_image_processor") and hasattr(pipe.vae_image_processor, "postprocess"):
            return pipe.vae_image_processor.postprocess(latents, output_type="pil")
        if hasattr(pipe, "decode_latents"):
            imgs = pipe.decode_latents(latents)
            return pipe.numpy_to_pil(imgs)
        lat = latents / 0.18215
        imgs = pipe.vae.decode(lat).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((img * 255).astype("uint8")) for img in imgs]

    def _save_preview_old(step, t, latents):
        if preview_frames is None:
            return
        imgs = _decode_preview_latents(latents)
        preview_frames.append(imgs[0])

    def _save_preview_new(_pipe, step, t, kwargs):
        latents = kwargs.get("latents")
        if latents is not None:
            _save_preview_old(step, t, latents)
        return {}

    for _ in range(int(batch_count)):
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            generator=generator,
            num_images_per_prompt=int(images_per_batch),
        )

        if smooth_preview:
            sig = inspect.signature(pipe.__call__)
            if "callback_on_step_end" in sig.parameters:
                call_kwargs["callback_on_step_end"] = _save_preview_new
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            else:
                call_kwargs["callback"] = _save_preview_old
                call_kwargs["callback_steps"] = 1

        result = pipe(**call_kwargs)
        for img in result.images:
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": int(seed),
                "steps": int(steps),
                "width": int(width),
                "height": int(height),
                "model": model,
                "lora": lora,
            }
            gallery.save_generation(img, metadata)
        images.extend(result.images)

    last_img = images[-1] if images else None

    preview_data = None
    if smooth_preview and preview_frames:
        buffer = io.BytesIO()
        imageio.mimsave(buffer, preview_frames, format="GIF", duration=0.2)
        buffer.seek(0)
        preview_data = buffer

    return last_img, seed, preview_data
