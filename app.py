import gradio as gr
from PIL import Image

from sdunity import presets, models, generator, gallery, config

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

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
                        choices=list(presets.PRESETS.keys()),
                        label="Preset",
                        value=None,
                    )
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    with gr.Accordion("Model", open=True):
                        categories = models.list_categories()
                        model_category = gr.Radio(
                            choices=categories,
                            value=categories[0] if categories else None,
                            label="Model Type",
                        )
                        model = gr.Dropdown(
                            choices=models.list_models(categories[0] if categories else None),
                            label="Model",
                        )
                        lora = gr.Dropdown(choices=models.list_loras(), label="LoRA", multiselect=True)
                        refresh = gr.Button("Refresh")
                    with gr.Accordion("Generation Settings", open=False):
                        seed = gr.Number(label="Seed", value=None, precision=0)
                        steps = gr.Slider(1, 50, value=20, label="Steps")
                        width = gr.Slider(64, 1024, value=256, step=64, label="Width")
                        height = gr.Slider(64, 1024, value=256, step=64, label="Height")
                        nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)
                        smooth_preview_chk = gr.Checkbox(label="Smooth Preview", value=False)
                        images_per_batch = gr.Number(label="Images per Batch", value=1, precision=0, minimum=1)
                        batch_count = gr.Number(label="Batch Count", value=1, precision=0, minimum=1)

            with gr.Row():
                output = gr.Image(label="Result")
                preview = gr.Image(label="Preview", visible=False)

        with gr.TabItem("Model Manager"):
            with gr.Tabs():
                with gr.TabItem("Models"):
                    gr.FileExplorer(
                        root_dir=config.MODELS_DIR,
                        glob="**/*",
                        file_count="multiple",
                        label="Model Files",
                    )
                with gr.TabItem("LoRAs"):
                    gr.FileExplorer(
                        root_dir=config.LORA_DIR,
                        glob="**/*.safetensors",
                        file_count="multiple",
                        label="LoRA Files",
                    )

        with gr.TabItem("Gallery"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_tree = gr.FileExplorer(
                        root_dir=config.GENERATIONS_DIR,
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
                meta = gallery.load_metadata(path)
                return img, meta

            file_tree.change(
                _load_selection,
                inputs=file_tree,
                outputs=[selected_image, metadata],
            )

    generate_btn.click(
        generator.generate_image,
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
            smooth_preview_chk,
        ],
        outputs=[output, seed, preview],
    )
    refresh.click(
        models.refresh_lists,
        inputs=model_category,
        outputs=[model_category, model, lora],
    )
    model_category.change(
        lambda c: gr.update(choices=models.list_models(c), value=models.list_models(c)[0] if models.list_models(c) else None),
        inputs=model_category,
        outputs=model,
    )

if __name__ == "__main__":
    # Launch Gradio using settings from config for easier customization
    demo.launch(**config.GRADIO_LAUNCH_CONFIG)
