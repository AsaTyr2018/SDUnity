import os
import sys
import requests
import gradio as gr
from PIL import Image

from sdunity import presets, models, generator, gallery, config, civitai, bootcamp, tags

MAX_THUMBNAILS = 50

# Keep a master list of gallery image paths so that select callbacks
# can access the full set without relying on Gradio to pass state.
GALLERY_PATHS: list[str] = []

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
#gengallery {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
}
#gengallery .gallery-item {
    flex: 0 0 auto;
    width: 254px !important;
    height: 254px !important;
    margin-right: 4px;
}
#gengallery img {
    object-fit: contain;
    width: 254px !important;
    height: 254px !important;
}

#images {
    display: grid;
    grid-template-columns: repeat(auto-fill, 128px);
    gap: 4px;
}
#images img {
    object-fit: contain;
    width: 128px !important;
    height: 128px !important;
}

#imagesgallery .thumbnail-item.thumbnail-lg {
    width: 128px !important;
    height: 128px !important;
    flex: 0 0 128px !important;
}

#imagesgallery .thumbnail-item.thumbnail-lg img {
    width: 128px !important;
    height: 128px !important;
    object-fit: cover;
}

#imagesgallery .gallery-item {
    width: 128px !important;
    height: 128px !important;
}
#imagesgallery img {
    object-fit: contain;
    width: 128px !important;
    height: 128px !important;
}
"""
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("# SDUnity")

    with gr.Tabs():
        with gr.TabItem("Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", lines=2)
                    tag_suggestions = gr.Dropdown(
                        label="Tag Suggestions",
                        choices=[],
                        visible=False,
                    )
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                    preset = gr.Dropdown(
                        choices=list(presets.PRESETS.keys()),
                        label="Preset",
                        value=None,
                    )
                    auto_enhance_chk = gr.Checkbox(
                        label="Auto Enhance Prompt", value=False
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
                            choices=models.list_models(
                                categories[0] if categories else None
                            ),
                            label="Model",
                        )
                        lora = gr.Dropdown(
                            choices=models.list_loras(), label="LoRA", multiselect=True
                        )
                        refresh = gr.Button("Refresh")
                        load_btn = gr.Button("Load to VRAM", variant="primary")
                        unload_btn = gr.Button("Unload from VRAM")
                        load_status = gr.Markdown("")
                        loaded_model_state = gr.State("")

            with gr.Row():
                output = gr.Image(
                    label="Result",
                    visible=True,
                    width=384,
                    height=384,
                    elem_id="preview",
                )
            with gr.Row():
                gen_gallery = gr.Gallery(
                    label="Current Batch",
                    show_label=True,
                    columns=[5],
                    elem_id="gengallery",
                )
                gen_gallery_state = gr.State([])

        with gr.TabItem("Generation Settings"):
            with gr.Row():
                with gr.Column():
                    seed = gr.Number(
                        label="Seed",
                        value=None,
                        precision=0,
                        info="Random seed for reproducible results",
                    )
                    random_seed_chk = gr.Checkbox(
                        label="Random Seed",
                        value=False,
                        info="Use a new seed for each batch",
                    )
                    steps = gr.Slider(
                        1, 50, value=20, label="Steps", info="Number of denoising steps"
                    )
                    guidance_scale = gr.Slider(
                        1,
                        20,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely the image follows the prompt",
                    )
                    clip_skip = gr.Slider(
                        1,
                        4,
                        value=1,
                        step=1,
                        label="Clip Skip",
                        info="Skip final CLIP layers",
                    )
                    scheduler = gr.Dropdown(
                        label="Sampler",
                        choices=["Euler", "Euler a", "DDIM", "DPM++ 2M Karras"],
                        value="Euler",
                    )
                    precision_dd = gr.Dropdown(
                        label="Precision",
                        choices=["fp16", "fp32"],
                        value="fp16",
                    )
                    tile_chk = gr.Checkbox(label="Tile Output", value=False)
                    lora_weight = gr.Slider(
                        0.0,
                        1.0,
                        step=0.05,
                        value=1.0,
                        label="LoRA Weight",
                    )
                    denoising_strength = gr.Slider(
                        0.0,
                        1.0,
                        step=0.05,
                        value=0.75,
                        label="Denoising Strength",
                    )
                    highres_fix_chk = gr.Checkbox(label="High Res Fix", value=False)
                    nsfw_filter = gr.Checkbox(
                        label="NSFW Filter", value=True, info="Enable safety checker"
                    )
                with gr.Column():
                    width = gr.Slider(
                        64,
                        1024,
                        value=256,
                        step=64,
                        label="Width",
                        info="Output image width",
                    )
                    height = gr.Slider(
                        64,
                        1024,
                        value=256,
                        step=64,
                        label="Height",
                        info="Output image height",
                    )
                    smooth_preview_chk = gr.Checkbox(
                        label="Smooth Preview",
                        value=False,
                        info="Show real-time preview while generating",
                    )
                    images_per_batch = gr.Number(
                        label="Images per Batch",
                        value=1,
                        precision=0,
                        minimum=1,
                        info="How many images to generate at once",
                    )
                    batch_count = gr.Number(
                        label="Batch Count",
                        value=1,
                        precision=0,
                        minimum=1,
                        info="Number of batches to run",
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
                    model_move_target = gr.Dropdown(
                        choices=models.list_categories(), label="Move To"
                    )
                    move_model_btn = gr.Button("Move Selected")
                    remove_model_btn = gr.Button("Remove Selected")
                    remove_status = gr.Markdown()
                with gr.TabItem("LoRAs"):
                    lora_browser = gr.FileExplorer(
                        root_dir=config.LORA_DIR,
                        glob="**/*.safetensors",
                        file_count="multiple",
                        label="LoRA Files",
                    )
                    lora_move_target = gr.Textbox(label="Move To (Category)")
                    move_lora_btn = gr.Button("Move Selected")
                    remove_lora_btn = gr.Button("Remove Selected")
                    lora_status = gr.Markdown()

                with gr.TabItem("Civitai Browser"):
                    with gr.Row():
                        with gr.Column(scale=1):
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
                            civitai_link = gr.Textbox(label="Direct Link")
                            civitai_link_dl = gr.Button("Download Link")
                        with gr.Column(scale=1):
                            civitai_results = gr.Dropdown(label="Results")
                            civitai_versions = gr.Dropdown(label="Version")
                            civitai_preview = gr.Image(
                                label="Preview", height=256, width=256
                            )
                            civitai_progress = gr.Textbox(
                                label="Download Progress",
                                value="",
                                interactive=False,
                                visible=False,
                            )
                            civitai_download = gr.Button("Download")
                            civitai_status = gr.Markdown(visible=False)
                            civitai_state = gr.State([])
                        with gr.Column(scale=1):
                            civitai_meta = gr.Markdown("", label="Model Info")

        with gr.TabItem("Gallery"):
            with gr.Row():
                with gr.Column(scale=2, elem_id="images"):
                    gr.Markdown("**Images**")
                    gallery_grid = [
                        gr.Image(
                            show_label=False,
                            interactive=True,
                            sources=[],
                            visible=False,
                        )
                        for _ in range(MAX_THUMBNAILS)
                    ]

                    refresh_gallery = gr.Button("Refresh")
                with gr.Column(scale=1):
                    selected_image = gr.Image(label="Image", elem_id="big_image")
                    metadata = gr.JSON(label="Metadata")
                    delete_btn = gr.Button("Delete Selected")
                    delete_status = gr.Markdown()
                    gallery_state = gr.State([])
                    selected_path = gr.State("")

            def _refresh_gallery():
                global GALLERY_PATHS
                paths = gallery.list_images()
                GALLERY_PATHS = paths
                updates = []
                for i in range(MAX_THUMBNAILS):
                    if i < len(paths):
                        updates.append(gr.update(value=paths[i], visible=True))
                    else:
                        updates.append(gr.update(value=None, visible=False))
                return updates + [paths[:MAX_THUMBNAILS], None, None, ""]

            def _select_image(index: int):
                global GALLERY_PATHS
                if index is None or index >= len(GALLERY_PATHS):
                    return None, None, ""
                path = GALLERY_PATHS[index]
                img = Image.open(path)
                meta = gallery.load_metadata(path)
                return img, meta, path

            def _delete_image(path):
                msg = "No image selected"
                if path and gallery.delete_image(path):
                    msg = "Image deleted"
                res = _refresh_gallery()
                return res[:-1] + [msg, ""]

            refresh_gallery.click(
                _refresh_gallery,
                outputs=gallery_grid
                + [gallery_state, selected_image, metadata, delete_status],
            )
            demo.load(
                _refresh_gallery,
                outputs=gallery_grid
                + [gallery_state, selected_image, metadata, delete_status],
            )
            for idx, img in enumerate(gallery_grid):
                img.select(
                    lambda i=idx: _select_image(i),
                    outputs=[selected_image, metadata, selected_path],
                )
            delete_btn.click(
                _delete_image,
                inputs=selected_path,
                outputs=gallery_grid
                + [
                    gallery_state,
                    selected_image,
                    metadata,
                    delete_status,
                    selected_path,
                ],
            )

            def _civitai_search(q, t, s):
                results = civitai.search_models(q, t, s)
                names = [r["name"] for r in results]
                if results:
                    vers = [v["name"] for v in results[0]["versions"]]
                    img = results[0]["versions"][0].get("image")
                    meta = civitai.format_metadata(
                        results[0], results[0]["versions"][0]
                    )
                else:
                    vers = []
                    img = None
                    meta = ""
                return (
                    gr.update(choices=names, value=names[0] if names else None),
                    gr.update(choices=vers, value=vers[0] if vers else None),
                    results,
                    img,
                    meta,
                    gr.update(value="", visible=False),
                )

            def _civitai_preview(name, state):
                if not name:
                    return None, "", gr.update(choices=[], value=None)
                for r in state:
                    if r["name"] == name:
                        vers = [v["name"] for v in r["versions"]]
                        ver = r["versions"][0]
                        img = ver.get("image")
                        meta = civitai.format_metadata(r, ver)
                        return (
                            img,
                            meta,
                            gr.update(choices=vers, value=vers[0] if vers else None),
                        )
                return None, "", gr.update(choices=[], value=None)

            def _civitai_version_change(model_name, version_name, state):
                for r in state:
                    if r["name"] == model_name:
                        for ver in r["versions"]:
                            if ver["name"] == version_name:
                                img = ver.get("image")
                                meta = civitai.format_metadata(r, ver)
                                return img, meta
                return None, ""

            def _civitai_download(name, ver_name, t, state, progress=gr.Progress()):
                for r in state:
                    if r["name"] == name:
                        for ver in r["versions"]:
                            if ver["name"] == ver_name:
                                dest_dir = os.path.join(
                                    config.MODELS_DIR, MODEL_DIR_MAP.get(t, t)
                                )
                                os.makedirs(dest_dir, exist_ok=True)
                                url = ver["downloadUrl"]
                                filename = os.path.basename(url)
                                msg = f"Downloading {filename} to {dest_dir}..."
                                yield gr.update(value=msg, visible=True), gr.update(
                                    value="0%", visible=True
                                )
                                try:
                                    resp = requests.get(
                                        url,
                                        stream=True,
                                        timeout=60,
                                        headers=civitai._headers(),
                                    )
                                    resp.raise_for_status()
                                except Exception as e:  # pragma: no cover - network
                                    print("Civitai download failed:", e)
                                    yield gr.update(
                                        value=f"Download failed: {e}"
                                    ), gr.update(value="Failed")
                                    return

                                filename = civitai._extract_filename(resp, url)
                                dest = os.path.join(dest_dir, filename)
                                total = int(resp.headers.get("content-length", 0))
                                downloaded = 0
                                progress((0, total), desc=f"Downloading {filename}")
                                yield gr.update(
                                    value="Download running... 0%"
                                ), gr.update(value="0%")
                                last_percent = 0
                                with open(dest, "wb") as f:
                                    for chunk in resp.iter_content(chunk_size=8192):
                                        if not chunk:
                                            continue
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total:
                                            percent = int(downloaded / total * 100)
                                            progress(
                                                (downloaded, total),
                                                desc=f"Downloading {filename}",
                                            )
                                            if percent - last_percent >= 5:
                                                last_percent = percent
                                                yield gr.update(
                                                    value=f"Download running... {percent}%"
                                                ), gr.update(value=f"{percent}%")
                                progress((total, total), desc="Download complete")
                                yield gr.update(
                                    value=f"Saved to {os.path.basename(dest)}"
                                ), gr.update(value="Done")
                                return
                yield gr.update(value="Model not found"), gr.update(value="")

            def _civitai_link_download(link, t, progress=gr.Progress()):
                if not link:
                    yield gr.update(value="No link provided"), gr.update(value="")
                    return
                dest_dir = os.path.join(config.MODELS_DIR, MODEL_DIR_MAP.get(t, t))
                os.makedirs(dest_dir, exist_ok=True)
                yield gr.update(
                    value=f"Downloading to {dest_dir}...", visible=True
                ), gr.update(value="0%", visible=True)
                try:
                    path = civitai.download_by_link(link, dest_dir, progress=progress)
                except Exception as e:  # pragma: no cover - network
                    print("Civitai link download failed:", e)
                    yield gr.update(value=f"Download failed: {e}"), gr.update(
                        value="Failed"
                    )
                    return
                yield gr.update(value=f"Saved to {os.path.basename(path)}"), gr.update(
                    value="Done"
                )

            def _remove_models(paths, current_cat):
                if not paths:
                    cat, mdl, lora_list = models.refresh_lists(current_cat)
                    return cat, mdl, lora_list, "No files selected", gr.update(choices=models.list_categories())
                removed = 0
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    name = os.path.basename(p)
                    if models.remove_model_file(name):
                        removed += 1
                cat_upd, model_upd, lora_upd = models.refresh_lists(current_cat)
                return (
                    cat_upd,
                    model_upd,
                    lora_upd,
                    f"Removed {removed} file(s)",
                    gr.update(choices=models.list_categories(), value=current_cat),
                )

            def _move_models(paths, dest_cat, current_cat):
                if not paths or not dest_cat:
                    cat, mdl, lora_list = models.refresh_lists(current_cat)
                    return (
                        cat,
                        mdl,
                        lora_list,
                        "No files selected",
                        gr.update(choices=models.list_categories()),
                    )
                moved = 0
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    name = os.path.basename(p)
                    try:
                        models.move_model_file(name, dest_cat, current_category=current_cat)
                        moved += 1
                    except FileNotFoundError:
                        pass
                cat_upd, model_upd, lora_upd = models.refresh_lists(current_cat)
                return (
                    cat_upd,
                    model_upd,
                    lora_upd,
                    f"Moved {moved} file(s)",
                    gr.update(choices=models.list_categories(), value=dest_cat),
                )

            def _remove_loras(paths):
                if not paths:
                    cat, mdl, lora_list = models.refresh_lists()
                    return cat, mdl, lora_list, "No files selected"
                removed = 0
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    name = os.path.basename(p)
                    if models.remove_lora_file(name):
                        removed += 1
                cat_upd, model_upd, lora_upd = models.refresh_lists()
                return cat_upd, model_upd, lora_upd, f"Removed {removed} file(s)"

            def _move_loras(paths, dest_cat):
                if not paths or not dest_cat:
                    cat, mdl, lora_list = models.refresh_lists()
                    return cat, mdl, lora_list, "No files selected"
                moved = 0
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    name = os.path.basename(p)
                    try:
                        models.move_lora_file(name, dest_cat)
                        moved += 1
                    except FileNotFoundError:
                        pass
                cat_upd, model_upd, lora_upd = models.refresh_lists()
                return cat_upd, model_upd, lora_upd, f"Moved {moved} file(s)"

            def _load_model(cat, name, loaded):
                if not name:
                    return loaded, "No model selected"
                if loaded and loaded != name:
                    models.unload_pipeline(loaded)
                models.load_pipeline(name, category=cat)
                return name, f"Loaded {name} to VRAM"

            def _unload_model(loaded):
                if not loaded:
                    return "", "No model loaded"
                models.unload_pipeline(loaded)
                return "", f"Unloaded {loaded}"

            def _prompt_autocomplete(text):
                opts = tags.suggestions_from_prompt(text)
                return gr.update(choices=opts, value=None, visible=bool(opts))

            def _apply_tag(text, choice):
                new_text = tags.apply_suggestion(text, choice)
                return new_text, gr.update(choices=[], value=None, visible=False)

        with gr.TabItem("Bootcamp"):
            with gr.Row():
                with gr.Column(scale=1):
                    bc_instance = gr.Textbox(
                        label="Instance Images",
                        value="data/instance",
                        info="Folder with training images",
                    )
                    bc_model = gr.Textbox(
                        label="Base Model",
                        value="",
                        info="Model checkpoint to fine-tune",
                    )
                    bc_output = gr.Textbox(
                        label="Output Directory",
                        value="loras/bootcamp",
                        info="Where to save the trained LoRA",
                    )
                    bc_steps = gr.Number(
                        label="Steps", value=1000, precision=0, info="Training steps"
                    )
                    bc_lr = gr.Number(
                        label="Learning Rate",
                        value=1e-4,
                        info="Optimizer learning rate",
                    )
                    bc_start = gr.Button("Start Bootcamp", variant="primary")
                bc_log = gr.Markdown()

        with gr.TabItem("Settings"):
            settings_inputs = []
            civitai_key = gr.Textbox(
                label="Civitai API Key",
                value=config.USER_CONFIG.get("civitai_api_key", ""),
                info="Optional API key for Civitai",
            )
            settings_inputs.append(civitai_key)

            with gr.Accordion("Server Options", open=False):
                for k, default in config.GRADIO_LAUNCH_CONFIG.items():
                    val = config.USER_CONFIG.get(k, default)
                    help_txt = {
                        "server_name": "Network interface to bind",
                        "server_port": "Port for the web UI",
                        "share": "Create a public link",
                        "inbrowser": "Open browser after launch",
                        "show_error": "Display errors in UI",
                        "debug": "Enable debug logs",
                        "max_threads": "Maximum thread workers",
                        "auth": "Username:password for login",
                        "auth_message": "Message on auth prompt",
                        "ssl_keyfile": "Path to SSL key",
                        "ssl_certfile": "Path to SSL certificate",
                        "quiet": "Reduce terminal output",
                        "show_api": "Expose REST API docs",
                    }.get(k, "")
                    if isinstance(default, bool):
                        comp = gr.Checkbox(label=k, value=bool(val), info=help_txt)
                    elif isinstance(default, int) or isinstance(default, float):
                        comp = gr.Number(label=k, value=val, precision=0, info=help_txt)
                    else:
                        comp = gr.Textbox(
                            label=k, value="" if val is None else val, info=help_txt
                        )
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
            guidance_scale,
            clip_skip,
            model_category,
            model,
            lora,
            nsfw_filter,
            images_per_batch,
            batch_count,
            preset,
            auto_enhance_chk,
            smooth_preview_chk,
            scheduler,
            precision_dd,
            tile_chk,
            lora_weight,
            denoising_strength,
            highres_fix_chk,
        ],
        outputs=[output, seed, gen_gallery, gen_gallery_state],
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

    prompt.input(_prompt_autocomplete, inputs=prompt, outputs=tag_suggestions)
    tag_suggestions.change(
        _apply_tag,
        inputs=[prompt, tag_suggestions],
        outputs=[prompt, tag_suggestions],
    )

    def _select_gen_image(evt: gr.SelectData, paths):
        if evt.index is None or evt.index >= len(paths):
            return None
        path = paths[evt.index]
        try:
            img = Image.open(path)
        except Exception:
            img = None
        return img

    gen_gallery.select(
        _select_gen_image,
        inputs=gen_gallery_state,
        outputs=output,
    )

    civitai_search.click(
        _civitai_search,
        inputs=[civitai_query, civitai_type, civitai_sort],
        outputs=[
            civitai_results,
            civitai_versions,
            civitai_state,
            civitai_preview,
            civitai_meta,
            civitai_status,
        ],
    )
    civitai_results.change(
        _civitai_preview,
        inputs=[civitai_results, civitai_state],
        outputs=[civitai_preview, civitai_meta, civitai_versions],
    )
    civitai_versions.change(
        _civitai_version_change,
        inputs=[civitai_results, civitai_versions, civitai_state],
        outputs=[civitai_preview, civitai_meta],
    )
    civitai_download.click(
        _civitai_download,
        inputs=[civitai_results, civitai_versions, civitai_type, civitai_state],
        outputs=[civitai_status, civitai_progress],
    )
    civitai_link_dl.click(
        _civitai_link_download,
        inputs=[civitai_link, civitai_type],
        outputs=[civitai_status, civitai_progress],
    )

    remove_model_btn.click(
        _remove_models,
        inputs=[model_browser, model_category],
        outputs=[model_category, model, lora, remove_status, model_move_target],
    )
    move_model_btn.click(
        _move_models,
        inputs=[model_browser, model_move_target, model_category],
        outputs=[model_category, model, lora, remove_status, model_move_target],
    )
    remove_lora_btn.click(
        _remove_loras,
        inputs=lora_browser,
        outputs=[model_category, model, lora, lora_status],
    )
    move_lora_btn.click(
        _move_loras,
        inputs=[lora_browser, lora_move_target],
        outputs=[model_category, model, lora, lora_status],
    )

    load_btn.click(
        _load_model,
        inputs=[model_category, model, loaded_model_state],
        outputs=[loaded_model_state, load_status],
    )
    unload_btn.click(
        _unload_model,
        inputs=loaded_model_state,
        outputs=[loaded_model_state, load_status],
    )

if __name__ == "__main__":
    # Launch Gradio using settings from config for easier customization
    demo.queue()
    demo.launch(**config.GRADIO_LAUNCH_CONFIG)
