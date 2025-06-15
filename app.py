import os
import random
import json

import gradio as gr
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline

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

def get_pipeline(model_name):
    """Load and cache the diffusion pipeline for the given model."""
    if model_name not in PIPELINES:
        repo = MODEL_LOOKUP.get(model_name)
        if repo is None:
            raise ValueError(f"Unknown model: {model_name}")

        if "xl" in model_name.lower() or "xl" in repo.lower():
            try:
                pipe = DiffusionPipeline.from_single_file(
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
):
    """Generate one or more images using the selected diffusion model."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if preset:
        enhancement = PRESETS.get(preset)
        if enhancement:
            prompt = f"{prompt}, {enhancement}"

    pipe = get_pipeline(model)

    # Toggle safety checker based on nsfw_filter flag
    if not hasattr(pipe, "_original_safety_checker"):
        pipe._original_safety_checker = getattr(pipe, "safety_checker", None)
    pipe.safety_checker = pipe._original_safety_checker if nsfw_filter else None

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    images = []
    for _ in range(int(batch_count)):
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            generator=generator,
            num_images_per_prompt=int(images_per_batch),
        )
        images.extend(result.images)

    if images:
        gallery_images.extend(images)
        last_img = images[-1]
    else:
        last_img = None

    return last_img, seed, gallery_images

with gr.Blocks() as demo:
    gr.Markdown("# SDUnity - Prototype")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")

            with gr.Row():
                seed = gr.Number(label="Seed", value=None, precision=0)
                steps = gr.Slider(1, 50, value=20, label="Steps")
                width = gr.Slider(64, 1024, value=256, step=64, label="Width")
                height = gr.Slider(64, 1024, value=256, step=64, label="Height")

            with gr.Row():
                model_category = gr.Radio(
                    choices=list_categories(),
                    value=list_categories()[0] if list_categories() else None,
                    label="Model Type",
                )
                model = gr.Dropdown(
                    choices=list_models(list_categories()[0] if list_categories() else None),
                    label="Model",
                )
            with gr.Row():
                lora = gr.Dropdown(choices=list_loras(), label="LoRA", multiselect=True)
                refresh = gr.Button("Refresh")

            generate_btn = gr.Button("Generate")

        with gr.Column(scale=1):
            # Box was removed in newer versions of Gradio; Group provides a
            # simple container without padding/margin.
            with gr.Group():
                gr.Markdown("### Additional Settings")
                nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)
                images_per_batch = gr.Number(
                    label="Images per Batch", value=1, precision=0
                )
            batch_count = gr.Number(label="Batch Count", value=1, precision=0)

        with gr.Group():
            gr.Markdown("### Presets")
            preset = gr.Dropdown(
                choices=list(PRESETS.keys()),
                label="Preset",
                value=None,
            )

    with gr.Row():
        output = gr.Image(label="Result")
        gallery = gr.Gallery(label="Gallery")

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
            images_per_batch,
            batch_count,
            preset,
        ],
        outputs=[output, seed, gallery],
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
