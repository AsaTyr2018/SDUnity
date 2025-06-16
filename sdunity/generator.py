import random
import inspect
import threading
from queue import Queue
from PIL import Image
import numpy as np
import torch
import gradio as gr

from . import presets, models, gallery


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed: int,
    random_seed: bool,
    steps: int,
    width: int,
    height: int,
    model_type: str,
    model: str,
    lora,
    nsfw_filter: bool,
    images_per_batch: int,
    batch_count: int,
    preset: str,
    smooth_preview: bool,
    progress=gr.Progress(),
) -> tuple[Image.Image | None, int]:
    """Generate one or more images using the selected diffusion model.

    Parameters
    ----------
    random_seed : bool
        If True, ignore the provided seed and generate a new random seed
        for this generation run.
    """
    if random_seed or seed is None:
        seed = random.randint(0, 2**32 - 1)

    images_per_batch = max(1, int(images_per_batch))
    batch_count = max(1, int(batch_count))

    if preset:
        enhancement = presets.PRESETS.get(preset)
        if enhancement:
            prompt = f"{prompt}, {enhancement}"

    pipe = models.get_pipeline(model, progress=progress, category=model_type)

    if not hasattr(pipe, "_original_safety_checker"):
        pipe._original_safety_checker = getattr(pipe, "safety_checker", None)
    pipe.safety_checker = pipe._original_safety_checker if nsfw_filter else None

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    images = []
    preview_queue = Queue() if smooth_preview else None
    _STOP = object()

    def _decode_preview_latents(latents):
        if hasattr(pipe, "image_processor") and hasattr(
            pipe.image_processor, "postprocess"
        ):
            return pipe.image_processor.postprocess(latents, output_type="pil")
        if hasattr(pipe, "vae_image_processor") and hasattr(
            pipe.vae_image_processor, "postprocess"
        ):
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
        if preview_queue is None:
            return
        imgs = _decode_preview_latents(latents)
        preview_queue.put(imgs[0])

    def _save_preview_new(_pipe, step, t, kwargs):
        latents = kwargs.get("latents")
        if latents is not None:
            _save_preview_old(step, t, latents)
        return {}

    def _run_generation():
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
                    "model_type": model_type,
                    "model": model,
                    "lora": lora,
                }
                gallery.save_generation(img, metadata)
            images.extend(result.images)

        if preview_queue is not None:
            preview_queue.put(_STOP)

    if smooth_preview:
        thread = threading.Thread(target=_run_generation)
        thread.start()
        while True:
            frame = preview_queue.get()
            if frame is _STOP:
                break
            yield frame, seed
        thread.join()
    else:
        _run_generation()

    last_img = images[-1] if images else None

    yield last_img, seed
