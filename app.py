import os
import random
import json
from urllib import request
from datetime import datetime
import requests
import inspect

import gradio as gr
from PIL import Image
import numpy as np
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

PRESETS_FILE = "presets.txt"


def load_presets(filepath=PRESETS_FILE):
    """Load prompt enhancement presets from a pipe-separated file."""
    presets = {}
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                display = f"{parts[0]} | {parts[1]}"
                presets[display] = parts[2]
    return presets


PRESETS = load_presets()

MODELS_DIR = "models"
LORA_DIR = "loras"
MODEL_REGISTRY_PATH = os.path.join("config", "model_registry.json")
GENERATIONS_DIR = "generations"
os.makedirs(GENERATIONS_DIR, exist_ok=True)


def load_model_registry(path=MODEL_REGISTRY_PATH):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_REGISTRY = load_model_registry()


def build_model_lookup():
    lookup = {}
    for category, models in MODEL_REGISTRY.items():
        for _name, info in models.items():
            filename = os.path.basename(info.get("url", ""))
            if filename:
                lookup[filename] = os.path.join(MODELS_DIR, category, filename)
    if os.path.isdir(MODELS_DIR):
        for root, _dirs, files in os.walk(MODELS_DIR):
            for f in files:
                if f.lower().endswith((".ckpt", ".safetensors", ".bin")):
                    lookup[f] = os.path.join(root, f)
    return lookup


MODEL_LOOKUP = build_model_lookup()

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def _find_model_info(model_name):
    """Return (category, info) tuple for the given filename from the registry."""
    for category, models in MODEL_REGISTRY.items():
        for _name, info in models.items():
            if os.path.basename(info.get("url", "")) == model_name:
                return category, info
    return None, None


def download_model_file(model_name, progress=None):
    """Download the model file defined in the registry if missing."""
    category, info = _find_model_info(model_name)
    if not info:
        raise ValueError(f"Unknown model: {model_name}")

    url = info.get("url")
    if not url:
        raise ValueError(f"No download URL for model: {model_name}")

    dest_dir = os.path.join(MODELS_DIR, category)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(url))

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest

    print(f"Downloading {model_name} from {url}")
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        if progress:
            progress(0, desc=f"Downloading {model_name}", total=total)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    progress(downloaded, desc=f"Downloading {model_name}", total=total)
        if progress:
            progress(total, desc="Download complete", total=total)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    return dest
# Cache for loaded pipelines
PIPELINES = {}


def list_categories():
    """Return available model categories discovered locally or in the registry."""
    cats = set(MODEL_REGISTRY.keys())
    if os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, fname)
            if os.path.isdir(path):
                cats.add(fname)
            elif fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                cats.add("Uncategorized")
    return sorted(cats)


def list_models(category=None):
    """Return model file names for the given category."""
    names = []
    if category in MODEL_REGISTRY:
        for _name, info in MODEL_REGISTRY[category].items():
            filename = os.path.basename(info.get("url", ""))
            if filename:
                names.append(filename)
    cat_dir = os.path.join(MODELS_DIR, category) if category else None
    if cat_dir and os.path.isdir(cat_dir):
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                if fname not in names:
                    names.append(fname)
    if category == "Uncategorized" and os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, fname)
            if os.path.isfile(path) and fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                names.append(fname)
    return sorted(names)


def list_loras():
    if not os.path.isdir(LORA_DIR):
        return []
    return [f for f in os.listdir(LORA_DIR) if f.lower().endswith(".safetensors")]


def refresh_lists(selected_category=None):
    """Return updated choices for category, model and LoRA dropdowns."""
    global MODEL_LOOKUP
    MODEL_LOOKUP = build_model_lookup()
    categories = list_categories()
    if selected_category not in categories:
        selected_category = categories[0] if categories else None
    models = list_models(selected_category)
    return (
        gr.update(choices=categories, value=selected_category),
        gr.update(choices=models, value=models[0] if models else None),
        gr.update(choices=list_loras()),
    )


gallery_images = []

def save_generation(image, metadata):
    """Save generated image and metadata to the generations directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date_dir = os.path.join(GENERATIONS_DIR, timestamp.split("_")[0])
    os.makedirs(date_dir, exist_ok=True)
    filename = f"{timestamp}_{metadata.get('seed', '0')}.png"
    img_path = os.path.join(date_dir, filename)
    image.save(img_path)
    meta_path = img_path.replace(".png", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return img_path

def load_metadata(img_path):
    """Load metadata JSON for the given image path."""
    meta_path = img_path.replace(".png", ".json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_pipeline(model_name, progress=None):
    """Load and cache the diffusion pipeline for the given model."""
    if model_name not in PIPELINES:
        repo = MODEL_LOOKUP.get(model_name)
        if not repo or not os.path.isfile(repo):
            # Attempt to download the model file if it doesn't exist
            try:
                repo = download_model_file(model_name, progress=progress)
                # rebuild lookup so subsequent calls see the new path
                MODEL_LOOKUP[model_name] = repo
            except Exception as e:
                raise ValueError(f"Unknown model: {model_name}") from e

        if "xl" in model_name.lower() or "xl" in repo.lower():
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    repo, torch_dtype=torch.float16, variant="fp16"
                )
            except ImportError as e:
                raise RuntimeError(
                    "StableDiffusionXLPipeline requires the 'transformers' library. "
                    "Install it with `pip install transformers`."
                ) from e
        else:
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    repo, torch_dtype=torch.float16
                )
            except ImportError as e:
                raise RuntimeError(
                    "StableDiffusionPipeline requires the 'transformers' library. "
                    "Install it with `pip install transformers`."
                ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        PIPELINES[model_name] = pipe
    return PIPELINES[model_name]


def generate_image(
    prompt,
    negative_prompt,
    seed,
    steps,
    width,
    height,
    model,
    lora,
    nsfw_filter,
    images_per_batch,
    batch_count,
    preset,
    smooth_preview,
    progress=gr.Progress(),
):
    """Generate one or more images using the selected diffusion model."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if preset:
        enhancement = PRESETS.get(preset)
        if enhancement:
            prompt = f"{prompt}, {enhancement}"

    pipe = get_pipeline(model, progress=progress)

    # Toggle safety checker based on nsfw_filter flag
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
            save_generation(img, metadata)
        images.extend(result.images)

    if images:
        gallery_images.extend(images)
        last_img = images[-1]
    else:
        last_img = None

    preview_data = None
    if smooth_preview and preview_frames:
        import io, imageio

        buffer = io.BytesIO()
        imageio.mimsave(buffer, preview_frames, format="GIF", duration=0.2)
        buffer.seek(0)
        preview_data = buffer

    return last_img, seed, preview_data

theme = gr.themes.Monochrome(primary_hue="slate").set(
    body_background_fill="#111111",
    body_background_fill_dark="#111111",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_background_fill="#1e1e1e",
    block_background_fill_dark="#1e1e1e",
    block_border_color="#333333",
    block_border_color_dark="#333333",
    input_background_fill="#222222",
    input_background_fill_dark="#222222",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# SDUnity")

    with gr.Tabs():
        with gr.TabItem("Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", lines=2)
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                    preset = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="Preset",
                        value=None,
                    )
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    with gr.Accordion("Model", open=True):
                        model_category = gr.Radio(
                            choices=list_categories(),
                            value=list_categories()[0] if list_categories() else None,
                            label="Model Type",
                        )
                        model = gr.Dropdown(
                            choices=list_models(list_categories()[0] if list_categories() else None),
                            label="Model",
                        )
                        lora = gr.Dropdown(choices=list_loras(), label="LoRA", multiselect=True)
                        refresh = gr.Button("Refresh")
                    with gr.Accordion("Generation Settings", open=False):
                        seed = gr.Number(label="Seed", value=None, precision=0)
                        steps = gr.Slider(1, 50, value=20, label="Steps")
                        width = gr.Slider(64, 1024, value=256, step=64, label="Width")
                        height = gr.Slider(64, 1024, value=256, step=64, label="Height")
                        nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)
                        smooth_preview_chk = gr.Checkbox(label="Smooth Preview", value=False)
                        images_per_batch = gr.Number(label="Images per Batch", value=1, precision=0)
                        batch_count = gr.Number(label="Batch Count", value=1, precision=0)

            with gr.Row():
                output = gr.Image(label="Result")
                preview = gr.Image(label="Preview", visible=False)

        with gr.TabItem("Model Manager"):
            gr.Markdown("WIP")

        with gr.TabItem("Gallery"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_tree = gr.FileExplorer(
                        root_dir=GENERATIONS_DIR,
                        glob="**/*.png",
                        file_count="single",
                        label="Generations",
                        every=5,
                    )
                with gr.Column(scale=2):
                    selected_image = gr.Image(label="Image")
                    metadata = gr.JSON(label="Metadata")

            def _load_selection(path):
                if not path:
                    return None, None
                img = Image.open(path)
                meta = load_metadata(path)
                return img, meta

            file_tree.change(
                _load_selection,
                inputs=file_tree,
                outputs=[selected_image, metadata],
            )

    generate_btn.click(
        generate_image,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            steps,
            width,
            height,
            model,
            lora,
            nsfw_filter,
            smooth_preview_chk,
            images_per_batch,
            batch_count,
            preset,
        ],
        outputs=[output, seed, preview],
    )
    refresh.click(
        refresh_lists,
        inputs=model_category,
        outputs=[model_category, model, lora],
    )
    model_category.change(
        lambda c: gr.update(choices=list_models(c), value=list_models(c)[0] if list_models(c) else None),
        inputs=model_category,
        outputs=model,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
