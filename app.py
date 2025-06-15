import os
import random

import gradio as gr
from PIL import Image
import numpy as np

MODELS_DIR = "models"
LORA_DIR = "loras"


def list_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".ckpt")]


def list_loras():
    if not os.path.isdir(LORA_DIR):
        return []
    return [f for f in os.listdir(LORA_DIR) if f.lower().endswith(".safetensors")]


def refresh_lists():
    """Return updated choices for model and LoRA dropdowns."""
    return gr.update(choices=list_models()), gr.update(choices=list_loras())


gallery_images = []


def generate_image(prompt, negative_prompt, seed, steps, width, height, model, lora):
    """Placeholder generator that returns a random image."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    gallery_images.append(img)
    return img, seed, gallery_images

with gr.Blocks() as demo:
    gr.Markdown("# SDUnity - Prototype")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt")

    with gr.Row():
        seed = gr.Number(label="Seed", value=None, precision=0)
        steps = gr.Slider(1, 50, value=20, label="Steps")
        width = gr.Slider(64, 1024, value=256, step=64, label="Width")
        height = gr.Slider(64, 1024, value=256, step=64, label="Height")

    with gr.Row():
        model = gr.Dropdown(choices=list_models(), label="Model")
        lora = gr.Dropdown(choices=list_loras(), label="LoRA", multiselect=True)
        refresh = gr.Button("Refresh")

    generate_btn = gr.Button("Generate")
    with gr.Row():
        output = gr.Image(label="Result")
        gallery = gr.Gallery(label="Gallery")

    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, seed, steps, width, height, model, lora],
        outputs=[output, seed, gallery],
    )
    refresh.click(refresh_lists, outputs=[model, lora])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
