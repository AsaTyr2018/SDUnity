import os
import sys
import gradio as gr
from PIL import Image

from sdunity import presets, models, generator, gallery, config, civitai, bootcamp

MODEL_DIR_MAP = {"sd15": "SD15", "sdxl": "SDXL", "ponyxl": "PonyXL"}

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

css = """
#preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain;
}
#download_popup {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}
.download-popup-content {
    background-color: #1e1e1e;
    padding: 1em;
    border-radius: 8px;
    width: 300px;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
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
                    with gr.Accordion("Generation Settings", open=False):
                        seed = gr.Number(label="Seed", value=None, precision=0)
                        random_seed_chk = gr.Checkbox(label="Random Seed", value=False)
                        steps = gr.Slider(1, 50, value=20, label="Steps")
                        width = gr.Slider(64, 1024, value=256, step=64, label="Width")
                        height = gr.Slider(64, 1024, value=256, step=64, label="Height")
                        nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)
                        smooth_preview_chk = gr.Checkbox(
                            label="Smooth Preview", value=False
                        )
                        images_per_batch = gr.Number(
                            label="Images per Batch", value=1, precision=0, minimum=1
                        )
                        batch_count = gr.Number(
                            label="Batch Count", value=1, precision=0, minimum=1
                        )

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Model", open=True):
                        categories = models.list_categories()
                        model_category = gr.Radio(
                            choices=categories,
                            value=categories[0] if categories else None,
                            label="Model Type",
                        )
                        model = gr.Dropdown(
                            choices=models.list_models(
                                categories[0] if categories else None
                            ),
                            label="Model",
                        )
                        lora = gr.Dropdown(
                            choices=models.list_loras(), label="LoRA", multiselect=True
                        )
                        refresh = gr.Button("Refresh")

            with gr.Row():
                output = gr.Image(
                    label="Result",
                    visible=True,
                    width=768,
                    height=768,
                    elem_id="preview",
                )

        with gr.TabItem("Model Manager"):
            with gr.Tabs():
                with gr.TabItem("Models"):
                    model_browser = gr.FileExplorer(
                        root_dir=config.MODELS_DIR,
                        glob="**/*.safetensors",
                        file_count="multiple",
                        label="Model Files",
                    )
                    remove_model_btn = gr.Button("Remove Selected")
                    remove_status = gr.Markdown()
                with gr.TabItem("LoRAs"):
                    gr.FileExplorer(
                        root_dir=config.LORA_DIR,
                        glob="**/*.safetensors",
                        file_count="multiple",
                        label="LoRA Files",
                    )

                with gr.TabItem("Civitai Browser"):
                    civitai_query = gr.Textbox(label="Search")
                    civitai_type = gr.Radio(
                        choices=["sd15", "sdxl", "ponyxl"],
                        value="sd15",
                        label="Model Type",
                    )
                    civitai_sort = gr.Dropdown(
                        choices=["Most Downloaded", "Highest Rated", "Newest"],
                        value="Most Downloaded",
                        label="Sort",
                    )
                    civitai_search = gr.Button("Search")
                    civitai_results = gr.Dropdown(label="Results")
                    civitai_preview = gr.Image(label="Preview")
                    civitai_download = gr.Button("Download")
                    civitai_status = gr.Markdown(visible=False)
                    civitai_state = gr.State([])
                    with gr.Group(elem_id="download_popup", visible=False) as download_popup:
                        with gr.Column(elem_classes="download-popup-content"):
                            popup_status = gr.Markdown()
                            popup_close = gr.Button("Close")

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

            def _civitai_search(q, t, s):
                results = civitai.search_models(q, t, s)
                names = [r["name"] for r in results]
                img = results[0].get("image") if results else None
                return (
                    gr.update(choices=names, value=names[0] if names else None),
                    results,
                    img,
                    gr.update(value="", visible=False),
                )

            def _civitai_preview(name, state):
                if not name:
                    return None
                for r in state:
                    if r["name"] == name:
                        return r.get("image")
                return None

            def _open_download_popup(name, t, state):
                for r in state:
                    if r["name"] == name:
                        dest = os.path.join(config.MODELS_DIR, MODEL_DIR_MAP.get(t, t))
                        filename = os.path.basename(r["downloadUrl"])
                        msg = f"Downloading {filename} to {dest}..."
                        return gr.update(visible=True), gr.update(value=msg)
                return gr.update(visible=False), gr.update(value="Model not found")

            def _close_download_popup():
                return gr.update(visible=False)

            def _civitai_download(name, t, state, progress=gr.Progress()):
                for r in state:
                    if r["name"] == name:
                        dest = os.path.join(config.MODELS_DIR, MODEL_DIR_MAP.get(t, t))
                        try:
                            path = civitai.download_model(r["downloadUrl"], dest, progress=progress)
                        except Exception as e:
                            print("Civitai download failed:", e)
                            return gr.update(value=f"Download failed: {e}")
                        return gr.update(value=f"Saved to {os.path.basename(path)}")
                return gr.update(value="Model not found")

            def _remove_models(paths, current_cat):
                if not paths:
                    cat, mdl, lora_list = models.refresh_lists(current_cat)
                    return cat, mdl, lora_list, "No files selected"
                removed = 0
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    name = os.path.basename(p)
                    if models.remove_model_file(name):
                        removed += 1
                cat_upd, model_upd, lora_upd = models.refresh_lists(current_cat)
                return cat_upd, model_upd, lora_upd, f"Removed {removed} file(s)"

        with gr.TabItem("Bootcamp"):
            with gr.Row():
                with gr.Column(scale=1):
                    bc_instance = gr.Textbox(label="Instance Images", value="data/instance")
                    bc_model = gr.Textbox(label="Base Model", value="")
                    bc_output = gr.Textbox(label="Output Directory", value="loras/bootcamp")
                    bc_steps = gr.Number(label="Steps", value=1000, precision=0)
                    bc_lr = gr.Number(label="Learning Rate", value=1e-4)
                    bc_start = gr.Button("Start Bootcamp", variant="primary")
                bc_log = gr.Markdown()

        with gr.TabItem("Settings"):
            settings_inputs = []
            civitai_key = gr.Textbox(
                label="Civitai API Key",
                value=config.USER_CONFIG.get("civitai_api_key", ""),
            )
            settings_inputs.append(civitai_key)

            for k, default in config.GRADIO_LAUNCH_CONFIG.items():
                val = config.USER_CONFIG.get(k, default)
                if isinstance(default, bool):
                    comp = gr.Checkbox(label=k, value=bool(val))
                elif isinstance(default, int) or isinstance(default, float):
                    comp = gr.Number(label=k, value=val, precision=0)
                else:
                    comp = gr.Textbox(label=k, value="" if val is None else val)
                settings_inputs.append(comp)

            save_status = gr.Markdown("")
            with gr.Row():
                save_btn = gr.Button("Save")
                save_reload_btn = gr.Button("Save + Reload")

            def _save_settings(*vals):
                cfg = {"civitai_api_key": vals[0]}
                idx = 1
                for key, default in config.GRADIO_LAUNCH_CONFIG.items():
                    val = vals[idx]
                    idx += 1
                    if isinstance(default, bool):
                        cfg[key] = bool(val)
                    elif isinstance(default, int) or isinstance(default, float):
                        cfg[key] = None if val == "" else int(val)
                    else:
                        cfg[key] = val if val != "" else None
                config.USER_CONFIG.update(cfg)
                config.save_user_config()
                civitai.set_api_key(cfg.get("civitai_api_key", ""))
                for key in config.GRADIO_LAUNCH_CONFIG:
                    if key in cfg:
                        config.GRADIO_LAUNCH_CONFIG[key] = cfg[key]
                return "Settings saved"

            def _save_reload(*vals):
                _save_settings(*vals)
                os.execl(sys.executable, sys.executable, *sys.argv)

            save_btn.click(_save_settings, inputs=settings_inputs, outputs=save_status)
            save_reload_btn.click(
                _save_reload, inputs=settings_inputs, outputs=save_status
            )

    generate_btn.click(
        generator.generate_image,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            random_seed_chk,
            steps,
            width,
            height,
            model_category,
            model,
            lora,
            nsfw_filter,
            images_per_batch,
            batch_count,
            preset,
            smooth_preview_chk,
        ],
        outputs=[output, seed],
    )
    refresh.click(
        models.refresh_lists,
        inputs=model_category,
        outputs=[model_category, model, lora],
    )
    model_category.change(
        lambda c: gr.update(
            choices=models.list_models(c),
            value=models.list_models(c)[0] if models.list_models(c) else None,
        ),
        inputs=model_category,
        outputs=model,
    )

    bc_start.click(
        bootcamp.train_lora,
        inputs=[bc_instance, bc_model, bc_output, bc_steps, bc_lr],
        outputs=bc_log,
    )

    civitai_search.click(
        _civitai_search,
        inputs=[civitai_query, civitai_type, civitai_sort],
        outputs=[civitai_results, civitai_state, civitai_preview, civitai_status],
    )
    civitai_results.change(
        _civitai_preview,
        inputs=[civitai_results, civitai_state],
        outputs=civitai_preview,
    )
    civitai_download.click(
        _open_download_popup,
        inputs=[civitai_results, civitai_type, civitai_state],
        outputs=[download_popup, popup_status],
    ).then(
        _civitai_download,
        inputs=[civitai_results, civitai_type, civitai_state],
        outputs=popup_status,
    )
    popup_close.click(
        _close_download_popup,
        outputs=download_popup,
        js="() => { const el = document.getElementById('download_popup'); if (el) el.style.display = 'none'; }",
    )

    remove_model_btn.click(
        _remove_models,
        inputs=[model_browser, model_category],
        outputs=[model_category, model, lora, remove_status],
    )

if __name__ == "__main__":
    # Launch Gradio using settings from config for easier customization
    demo.queue()
    demo.launch(**config.GRADIO_LAUNCH_CONFIG)
