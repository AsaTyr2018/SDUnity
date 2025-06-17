import random
import os
import inspect
import threading
from queue import Queue
from PIL import Image
import torch
import gradio as gr
from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)

from . import presets, models, gallery
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _apply_clip_skip(pipe, skip: int) -> None:
    """Patch the pipeline's text encoder to apply clip skip."""
    if skip <= 1:
        if hasattr(pipe, "_clip_skip_patched") and pipe._clip_skip_patched:
            pipe.text_encoder.forward = pipe._orig_text_encoder_forward
            pipe._clip_skip_patched = False
        return

    if not hasattr(pipe, "_orig_text_encoder_forward"):
        pipe._orig_text_encoder_forward = pipe.text_encoder.forward

    def _forward(*args, **kwargs):
        kwargs["output_hidden_states"] = True
        out = pipe._orig_text_encoder_forward(*args, **kwargs)
        hidden = out.hidden_states[-skip - 1]
        hidden = pipe.text_encoder.text_model.final_layer_norm(hidden)
        data = {**out.__dict__, "last_hidden_state": hidden}
        return BaseModelOutputWithPooling(**data)

    pipe.text_encoder.forward = _forward
    pipe._clip_skip_patched = True


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed: int,
    random_seed: bool,
    steps: int,
    width: int,
    height: int,
    guidance_scale: float,
    clip_skip: int,
    model_type: str,
    model: str,
    lora,
    nsfw_filter: bool,
    images_per_batch: int,
    batch_count: int,
    preset: str,
    smooth_preview: bool,
    scheduler: str,
    precision: str,
    tile: bool,
    lora_weight: float,
    denoising_strength: float,
    highres_fix: bool,
    progress=gr.Progress(),
) -> tuple[Image.Image | None, int, list]:
    """Generate one or more images using the selected diffusion model.

    Parameters
    ----------
    random_seed : bool
        If ``True``, a new random seed is used for each batch. Otherwise the
        provided ``seed`` is incremented per batch to keep results
        reproducible.
    guidance_scale : float
        Classifier-free guidance scale.
    clip_skip : int
        Number of final CLIP layers to skip when encoding text.
    scheduler : str
        Sampling scheduler to use.
    precision : str
        ``fp16`` or ``fp32`` precision.
    tile : bool
        Enable tiling for seamless textures.
    lora_weight : float
        Weight for loaded LoRA modules.
    denoising_strength : float
        Denoising strength for future image-to-image support.
    highres_fix : bool
        Apply a high resolution pass when enabled.
    """
    if random_seed or seed is None:
        seed = random.randint(0, 2**32 - 1)

    base_seed = int(seed)

    images_per_batch = max(1, int(images_per_batch))
    batch_count = max(1, int(batch_count))

    if preset:
        enhancement = presets.PRESETS.get(preset)
        if enhancement:
            prompt = f"{prompt}, {enhancement}"

    pipe = models.get_pipeline(model, progress=progress, category=model_type)

    if scheduler:
        sched_map = {
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
        }
        sched_cls = sched_map.get(scheduler)
        if sched_cls:
            pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)

    dtype = torch.float16 if precision == "fp16" else torch.float32
    try:
        pipe.to(dtype=dtype)
    except Exception:
        pass

    if tile and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(pipe, "disable_vae_tiling"):
        pipe.disable_vae_tiling()

    if lora:
        if not isinstance(lora, list):
            lora = [lora]
        for name in lora:
            path = models.LORA_LOOKUP.get(name)
            if path and hasattr(pipe, "load_lora_weights"):
                try:
                    pipe.load_lora_weights(path, weight=float(lora_weight))
                except Exception:
                    pass

    if not hasattr(pipe, "_original_safety_checker"):
        pipe._original_safety_checker = getattr(pipe, "safety_checker", None)
    pipe.safety_checker = pipe._original_safety_checker if nsfw_filter else None

    _apply_clip_skip(pipe, int(clip_skip))


    images = []
    new_paths = []
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
        for batch_idx in range(int(batch_count)):
            if random_seed:
                batch_seed = random.randint(0, 2**32 - 1)
            else:
                batch_seed = base_seed + batch_idx

            seeds = [batch_seed + i for i in range(int(images_per_batch))]
            if images_per_batch == 1:
                gens = torch.Generator(device=pipe.device).manual_seed(seeds[0])
            else:
                gens = [
                    torch.Generator(device=pipe.device).manual_seed(s)
                    for s in seeds
                ]

            call_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                generator=gens,
                num_images_per_prompt=int(images_per_batch),
                guidance_scale=float(guidance_scale),
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
            for idx, img in enumerate(result.images):
                meta_seed = seeds[idx]
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": int(meta_seed),
                    "steps": int(steps),
                    "width": int(width),
                    "height": int(height),
                    "guidance_scale": float(guidance_scale),
                    "clip_skip": int(clip_skip),
                    "scheduler": scheduler,
                    "precision": precision,
                    "tile": bool(tile),
                    "lora_weight": float(lora_weight),
                    "denoising_strength": float(denoising_strength),
                    "highres_fix": bool(highres_fix),
                    "model_type": model_type,
                    "model": model,
                    "lora": lora,
                }
                path = gallery.save_generation(img, metadata)
                new_paths.append(path)
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
            yield frame, base_seed, gr.update()
        thread.join()
    else:
        _run_generation()

    last_img = images[-1] if images else None
    gallery_items = [(p, os.path.basename(p)) for p in new_paths]

    yield last_img, base_seed, gallery_items, new_paths
