import os
import sys
import requests
import gradio as gr
from PIL import Image

from sdunity import (
    presets,
    models,
    generator,
    gallery,
    config,
    civitai,
    bootcamp,
    tags,
    settings_presets,
)

MAX_THUMBNAILS = 50

# Keep a master list of gallery image paths so that select callbacks
# can access the full set without relying on Gradio to pass state.
GALLERY_PATHS: list[str] = []

MODEL_DIR_MAP = {"sd15": "SD15", "sdxl": "SDXL", "ponyxl": "PonyXL"}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

theme = gr.themes.Monochrome(
    primary_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#0d0d0d",
    body_background_fill_dark="#0d0d0d",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_background_fill="#1b1b1b",
    block_background_fill_dark="#1b1b1b",
    block_border_color="#262626",
    block_border_color_dark="#262626",
    input_background_fill="#222222",
    input_background_fill_dark="#222222",
    shadow_drop="0 4px 8px rgba(0,0,0,0.6)",
    shadow_drop_lg="0 6px 12px rgba(0,0,0,0.8)",
    block_shadow="var(--shadow-drop-lg)",
    block_shadow_dark="0 6px 12px rgba(0,0,0,0.9)",
)

css = """
# Global dark background with subtle gradient
body {
    background: radial-gradient(circle at 50% 0%, #111111, #000000) !important;
}

# Slight 3D effect for blocks
.gradio-container .gr-block, .gradio-container .gr-box {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.6), 0 4px 8px rgba(0, 0, 0, 0.4);
    border-radius: 8px;
}

#preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain;
}
#finished_gallery {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
}
#finished_gallery .gallery-item {
    flex: 0 0 auto;
    width: 254px !important;
    height: 254px !important;
    margin-right: 4px;
}
#finished_gallery img {
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
#prompt_wrapper {
    position: relative;
}
#tag_suggestions {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 10;
}

# Bootcamp grid
#bc_grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
    gap: 8px;
}
.bc_item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4px;
    border: 1px solid #333;
    border-radius: 4px;
}
.bc_item img {
    width: 128px !important;
    height: 128px !important;
    object-fit: cover;
    display: block;
    margin-bottom: 4px;
}
.bc_tags {
    text-align: center;
}
.bc_tag_none {
    color: #888;
    font-size: 0.8em;
}
.bc_tags button {
    margin: 0 2px 2px 0;
    padding: 2px 4px;
    font-size: 0.8em;
    background: #333;
    border: 1px solid #555;
    border-radius: 4px;
    cursor: pointer;
}
.bc_tags button.selected {
    background: #6b7280;
}
"""
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("# SDUnity")

    with gr.Tabs():
        with gr.TabItem("Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group(elem_id="prompt_wrapper"):
                        prompt = gr.Textbox(label="Prompt", lines=2, elem_id="prompt_box")
                        tag_suggestions = gr.Dropdown(
                            label="",
                            choices=[],
                            visible=False,
                            container=False,
                            elem_id="tag_suggestions",
                        )
                        gr.HTML(
                            """
                            <script>
                            document.addEventListener('DOMContentLoaded', () => {
                                function attach() {
                                    const prompt = document.querySelector('#prompt_box textarea, #prompt_box input');
                                    const sugg = document.querySelector('#tag_suggestions select');
                                    if (!prompt || !sugg) return false;
                                    prompt.addEventListener('keydown', (e) => {
                                        if (e.key === 'Tab' && sugg.options.length) {
                                            e.preventDefault();
                                            sugg.value = sugg.options[0].value;
                                            sugg.dispatchEvent(new Event('change', {bubbles: true}));
                                        }
                                    });
                                    return true;
                                }
                                if (!attach()) {
                                    const iv = setInterval(() => {
                                        if (attach()) clearInterval(iv);
                                    }, 500);
                                }
                            });
                            </script>
                            """,
                            elem_id="prompt_js",
                            visible=False,
                            container=False,
                        )
                with gr.Column(scale=3):
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    preset = gr.Dropdown(
                        choices=list(presets.PRESETS.keys()),
                        label="Preset",
                        value=None,
                    )
                    auto_enhance_chk = gr.Checkbox(label="Auto Enhance Prompt", value=False)

            with gr.Row():
                categories = models.list_categories()
                model_category = gr.Dropdown(
                    choices=categories,
                    value=categories[0] if categories else None,
                    label="Model Type",
                )
                model = gr.Dropdown(
                    choices=models.list_models(categories[0] if categories else None),
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

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Row():
                output = gr.Image(
                    label="Preview",
                    visible=False,
                    width=384,
                    height=384,
                    elem_id="preview",
                )
                finished_gallery = gr.Gallery(
                    label="Finished Images",
                    show_label=True,
                    columns=[5],
                    elem_id="finished_gallery",
                )
                finished_gallery_state = gr.State([])

        with gr.TabItem("Generation Settings"):
            with gr.Row():
                settings_preset_dd = gr.Dropdown(
                    choices=list(settings_presets.PRESETS.keys()),
                    label="Settings Preset",
                    value=None,
                )
                save_settings_preset_btn = gr.Button("Save Preset")
                remove_settings_preset_btn = gr.Button("Remove Preset")
            with gr.Row():
                preset_name_box = gr.Textbox(label="Preset Name", visible=False)
                preset_confirm_btn = gr.Button("Confirm", visible=False)
                preset_status = gr.Markdown("", visible=False)
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

            def _show_preset_prompt():
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value="", visible=True),
                )

            def _save_preset(
                name,
                seed_val,
                rand,
                step_val,
                width_val,
                height_val,
                guidance_val,
                clip_val,
                sched,
                prec,
                tile_val,
                lora_w,
                denoise,
                highres,
                nsfw,
                smooth,
                per_batch,
                batches,
                category,
                mdl,
                lora_sel,
            ):
                if not name:
                    return (
                        gr.update(),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(value="Name required", visible=True),
                    )
                if name in settings_presets.PRESETS:
                    return (
                        gr.update(),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(value="Name exists", visible=True),
                    )
                data = {
                    "seed": seed_val,
                    "random_seed": bool(rand),
                    "steps": int(step_val),
                    "width": int(width_val),
                    "height": int(height_val),
                    "guidance_scale": float(guidance_val),
                    "clip_skip": int(clip_val),
                    "scheduler": sched,
                    "precision": prec,
                    "tile": bool(tile_val),
                    "lora_weight": float(lora_w),
                    "denoising_strength": float(denoise),
                    "highres_fix": bool(highres),
                    "nsfw_filter": bool(nsfw),
                    "smooth_preview": bool(smooth),
                    "images_per_batch": int(per_batch),
                    "batch_count": int(batches),
                    "model_type": category,
                    "model": mdl,
                    "lora": lora_sel,
                }
                settings_presets.add_preset(name, data)
                return (
                    gr.update(choices=list(settings_presets.PRESETS.keys()), value=name),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(value="Preset saved", visible=True),
                )

            def _load_preset(name):
                data = settings_presets.PRESETS.get(name)
                if not data:
                    return [gr.update() for _ in range(20)]
                model_choices = models.list_models(data.get("model_type"))
                model_val = data.get("model")
                if model_val not in model_choices:
                    model_val = model_choices[0] if model_choices else None
                return [
                    data.get("seed"),
                    data.get("random_seed"),
                    data.get("steps"),
                    data.get("guidance_scale"),
                    data.get("clip_skip"),
                    data.get("scheduler"),
                    data.get("precision"),
                    data.get("tile"),
                    data.get("lora_weight"),
                    data.get("denoising_strength"),
                    data.get("highres_fix"),
                    data.get("nsfw_filter"),
                    data.get("width"),
                    data.get("height"),
                    data.get("smooth_preview"),
                    data.get("images_per_batch"),
                    data.get("batch_count"),
                    data.get("model_type"),
                    gr.update(choices=model_choices, value=model_val),
                    data.get("lora"),
                ]

            def _remove_preset(name):
                settings_presets.remove_preset(name)
                return gr.update(choices=list(settings_presets.PRESETS.keys()), value=None)

            def _prompt_autocomplete(text):
                opts = tags.suggestions_from_prompt(text)
                return gr.update(choices=opts, value=None, visible=bool(opts))

            def _apply_tag(text, choice):
                new_text = tags.apply_suggestion(text, choice)
                return new_text, gr.update(choices=[], value=None, visible=False)

        with gr.TabItem("Bootcamp"):
            bc_project = gr.State()
            bc_nav = gr.State(False)
            bc_step = gr.State(1)
            bc_progress = gr.Markdown("### Step 1 of 5", elem_id="bc_progress")
            gr.HTML(
                """
                <script>
                function goTab(id){
                  const el = document.getElementById(id);
                  if(el){ el.click(); }
                }
                </script>
                """,
                visible=False,
            )
            with gr.Tabs() as bc_tabs:
                with gr.TabItem("1. Setup", elem_id="bc_setup_tab"):
                    bc_type = gr.Radio(
                        ["Character", "Style", "Concept"],
                        label="LoRA Type",
                        value="Character",
                    )
                    bc_name = gr.Textbox(label="Project Name", value="")
                    bc_create = gr.Button("Create Project")
                    bc_next1 = gr.Button("Next", variant="primary")
                    bc_setup_out = gr.Markdown()
                with gr.TabItem("2. Upload", elem_id="bc_upload_tab"):
                    bc_upload_input = gr.File(
                        label="Upload Zip or Folder",
                        file_count="directory",
                        file_types=["image", ".zip"],
                    )
                    bc_upload = gr.Button("Upload")
                    bc_prev2 = gr.Button("Back")
                    bc_next2 = gr.Button("Next", variant="primary")
                    bc_file_count = gr.Number(label="Image Count", value=0, interactive=False)
                    bc_upload_msg = gr.Markdown()
                with gr.TabItem("3. Tagging", elem_id="bc_tag_tab"):
                    bc_autotag_open = gr.Button("Auto-Tagging", variant="secondary")
                    bc_download_ds = gr.Button("Download Dataset")
                    bc_reset_proj = gr.Button("Reset Project")
                    bc_download_out = gr.File(label="Dataset Zip", visible=False)
                    with gr.Column(visible=False) as bc_autotag_popup:
                        bc_max_tags = gr.Number(label="Max Tags", value=5, precision=0)
                        bc_thresh = gr.Number(label="Min Threshold", value=0.35)
                        bc_blacklist = gr.Textbox(label="Blacklist")
                        bc_prepend = gr.Textbox(label="Prepend Tags")
                        bc_append = gr.Textbox(label="Append Tags")
                        bc_run_autotag = gr.Button("Run Auto-Tagging")
                        bc_autotag_close = gr.Button("Close")
                        bc_autotag_msg = gr.Markdown()
                    bc_tags_grid = gr.HTML()
                    bc_tags_df = gr.Dataframe(headers=["Image", "Tags"], datatype=["str", "str"], row_count=0, visible=False)
                    bc_save_tags = gr.Button("Save Tags")
                    bc_prev3 = gr.Button("Back")
                    bc_next3 = gr.Button("Next", variant="primary")
                    bc_selected_tags = gr.Textbox(label="Selected Tags", interactive=False, elem_id="bc_selected_tags")
                    bc_tag_view = gr.Dataframe(headers=["Tag", "Count"], datatype=["str", "int"], interactive=False)
                with gr.TabItem("4. Training Parameters", elem_id="bc_params_tab"):
                    bc_model_select = gr.Radio(["SD 1.5", "SDXL", "Pony"], label="Model", value="SD 1.5")
                    bc_steps = gr.Number(label="Steps", value=1000, precision=0)
                    bc_lr = gr.Number(label="Learning Rate", value=1e-4)
                    with gr.Accordion("More Options", open=False):
                        bc_epochs = gr.Number(label="Epochs", value=10, precision=0)
                        bc_num_repeats = gr.Number(label="Num Repeats", value=1, precision=0)
                        bc_batch = gr.Number(label="Train Batch Size", value=4, precision=0)
                        bc_resolution = gr.Number(label="Resolution", value=1024, precision=0)
                        bc_lora_type_field = gr.Textbox(label="LoRA Type", value="Lora")
                        bc_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
                        bc_shuffle_tags = gr.Checkbox(label="Shuffle Tags", value=False)
                        bc_keep_tokens = gr.Number(label="Keep Tokens", value=0, precision=0)
                        bc_clip_skip_train = gr.Number(label="Clip Skip", value=1, precision=0)
                        bc_flip_aug = gr.Checkbox(label="Flip Augmentation", value=True)
                        bc_unet_lr = gr.Number(label="Unet LR", value=0.00050)
                        bc_text_lr = gr.Number(label="Text Encoder LR", value=0.00005)
                        bc_scheduler = gr.Textbox(label="LR Scheduler", value="cosine_with_restarts")
                        bc_scheduler_cycles = gr.Number(label="LR Scheduler Cycles", value=3, precision=0)
                        bc_min_snr = gr.Number(label="Min SNR Gamma", value=5)
                        bc_net_dim = gr.Number(label="Network Dim", value=32, precision=0)
                        bc_net_alpha = gr.Number(label="Network Alpha", value=16, precision=0)
                        bc_noise_offset = gr.Number(label="Noise Offset", value=0.10)
                        bc_optimizer = gr.Textbox(label="Optimizer", value="Adafactor")
                        bc_optimizer_args = gr.Textbox(
                            label="Optimizer Args",
                            value="scale_parameter=False, relative_step=False, warmup_init=False",
                        )
                    bc_prompt1 = gr.Textbox(label="Image #1", value="Automatically set")
                    bc_prompt2 = gr.Textbox(label="Image #2")
                    bc_prompt3 = gr.Textbox(label="Image #3")
                    bc_prev4 = gr.Button("Back")
                    bc_next4 = gr.Button("Next", variant="primary")
                with gr.TabItem("5. Review and Start Training", elem_id="bc_review_tab"):
                    bc_review = gr.Markdown()
                    bc_train = gr.Button("Start Training", variant="primary")
                    bc_prev5 = gr.Button("Back")
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
                    if k == "allowed_paths":
                        continue
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
                    if key == "allowed_paths":
                        continue
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

    def _generate_with_preview(*args):
        last = None
        for result in generator.generate_image(*args):
            last = result
            img, seed_val, gallery_items, new_paths = result
            yield gr.update(value=img, visible=True), seed_val, gallery_items, new_paths
        if last is not None:
            img, seed_val, gallery_items, new_paths = last
            yield gr.update(visible=False), seed_val, gallery_items, new_paths

    generate_btn.click(
        _generate_with_preview,
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
        outputs=[output, seed, finished_gallery, finished_gallery_state],
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


    def _create_project_ui(lora_type, name):
        if not name:
            return None, "Please provide a project name.", False, "### Step 1 of 5", 1
        proj = bootcamp.create_project(name, lora_type)
        return proj.name, f"Created project `{name}`", True, "### Step 2 of 5", 2

    def _upload_files_ui(proj_name, upload_files):
        if not proj_name or not upload_files:
            return 0, [], "", "No project or file uploaded", False, "### Step 2 of 5", 2
        if isinstance(upload_files, str):
            uploads = [upload_files]
        else:
            uploads = [f.name for f in upload_files] if hasattr(upload_files, "__iter__") else [upload_files.name]
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return 0, [], "", "Project not found", False, "### Step 2 of 5", 2
        count = bootcamp.import_uploads(proj, uploads)
        rows = [[img, ""] for img in proj.images]
        html = bootcamp.render_tag_grid(proj)
        return count, rows, html, f"Imported {count} images", True, "### Step 3 of 5", 3

    def _save_tags_ui(proj_name, rows):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return [], "", False, "### Step 3 of 5", 3
        for img, tag_str in rows:
            proj.tags[img] = [t.strip() for t in str(tag_str).split(",") if t.strip()]
        proj.save()
        data = [[k, v] for k, v in bootcamp.tag_summary(proj).items()]
        html = bootcamp.render_tag_grid(proj)
        return data, html, True, "### Step 4 of 5", 4

    def _run_autotag_ui(proj_name, prepend, append, blacklist, max_tags, thresh, progress=gr.Progress()):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return [], "", "Project not found"
        pre = [t.strip() for t in prepend.split(",") if t.strip()]
        app = [t.strip() for t in append.split(",") if t.strip()]
        bl = {t.strip() for t in blacklist.split(",") if t.strip()}
        bootcamp.auto_tag_dataset(
            proj,
            max_tags=int(max_tags),
            threshold=float(thresh),
            prepend=pre,
            append=app,
            blacklist=bl,
            progress=progress,
        )
        rows = [[img, ", ".join(proj.tags[img])] for img in proj.images]
        html = bootcamp.render_tag_grid(proj)
        return rows, html, "Auto tags generated"

    def _export_dataset_ui(proj_name):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return None
        out_path = os.path.join(config.BOOTCAMP_OUTPUT_DIR, f"{proj.name}.zip")
        bootcamp.export_dataset(proj, out_path)
        return out_path

    def _reset_project_ui(proj_name):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return [], ""
        bootcamp.reset_project(proj)
        return [], ""

    def _show_autotag_ui():
        return gr.update(visible=True)

    def _hide_autotag_ui():
        return gr.update(visible=False)

    def _review_ui(proj_name, model_type):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return "No project", 0, 0.0, 1
        params = bootcamp.suggest_params(proj, model_type)
        info = f"### {proj.name}\nType: {proj.lora_type}\nImages: {len(proj.images)}"
        return info, params["steps"], params["learning_rate"], params["num_repeats"]

    def _train_ui(proj_name, model_type, steps, lr):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            yield "Project not found"
            return
        yield from bootcamp.run_training(proj, model_type, steps, lr)

    def _next_from_setup(proj_name):
        if not proj_name:
            return "Please create a project first.", False, "### Step 1 of 5", 1
        return "", True, "### Step 2 of 5", 2

    def _next_from_upload(proj_name):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None or not proj.images:
            return "Please upload images first.", False, "### Step 2 of 5", 2
        return "", True, "### Step 3 of 5", 3

    def _next_from_tags(proj_name):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None or not proj.images:
            return "No dataset found.", False, "### Step 3 of 5", 3
        return "", True, "### Step 4 of 5", 4

    def _next_from_params(proj_name):
        proj = bootcamp.BootcampProject.load(proj_name)
        if proj is None:
            return "Project not found", False, "### Step 4 of 5", 4
        return "", True, "### Step 5 of 5", 5

    bc_create.click(
        _create_project_ui,
        inputs=[bc_type, bc_name],
        outputs=[bc_project, bc_setup_out, bc_nav, bc_progress, bc_step],
        js="(p,msg,nav,prog,step)=>{ if(nav){ goTab('bc_upload_tab'); } }",
    )
    bc_next1.click(
        _next_from_setup,
        inputs=bc_project,
        outputs=[bc_setup_out, bc_nav, bc_progress, bc_step],
        js="(msg,nav,prog,step)=>{ if(nav){ goTab('bc_upload_tab'); } }",
    )
    bc_upload.click(
        _upload_files_ui,
        inputs=[bc_project, bc_upload_input],
        outputs=[bc_file_count, bc_tags_df, bc_tags_grid, bc_upload_msg, bc_nav, bc_progress, bc_step],
        js="(count,rows,grid,msg,nav,prog,step)=>{ if(nav){ goTab('bc_tag_tab'); } }",
    )
    bc_prev2.click(
        lambda: ("### Step 1 of 5", 1),
        outputs=[bc_progress, bc_step],
        js="goTab('bc_setup_tab')",
    )
    bc_next2.click(
        _next_from_upload,
        inputs=bc_project,
        outputs=[bc_upload_msg, bc_nav, bc_progress, bc_step],
        js="(msg,nav,prog,step)=>{ if(nav){ goTab('bc_tag_tab'); } }",
    )
    bc_save_tags.click(
        _save_tags_ui,
        inputs=[bc_project, bc_tags_df],
        outputs=[bc_tag_view, bc_tags_grid, bc_nav, bc_progress, bc_step],
        js="(view,grid,nav,prog,step)=>{ if(nav){ goTab('bc_params_tab'); } }",
    )
    bc_prev3.click(
        lambda: ("### Step 2 of 5", 2),
        outputs=[bc_progress, bc_step],
        js="goTab('bc_upload_tab')",
    )
    bc_next3.click(
        _next_from_tags,
        inputs=bc_project,
        outputs=[bc_upload_msg, bc_nav, bc_progress, bc_step],
        js="(msg,nav,prog,step)=>{ if(nav){ goTab('bc_params_tab'); } }",
    )
    bc_download_ds.click(
        _export_dataset_ui,
        inputs=bc_project,
        outputs=bc_download_out,
    )
    bc_reset_proj.click(
        _reset_project_ui,
        inputs=bc_project,
        outputs=[bc_tags_df, bc_tags_grid],
    )
    bc_autotag_open.click(_show_autotag_ui, outputs=bc_autotag_popup)
    bc_autotag_close.click(_hide_autotag_ui, outputs=bc_autotag_popup)
    bc_run_autotag.click(
        _run_autotag_ui,
        inputs=[bc_project, bc_prepend, bc_append, bc_blacklist, bc_max_tags, bc_thresh],
        outputs=[bc_tags_df, bc_tags_grid, bc_autotag_msg],
    )
    bc_model_select.change(
        _review_ui,
        inputs=[bc_project, bc_model_select],
        outputs=[bc_review, bc_steps, bc_lr, bc_num_repeats],
    )
    bc_prev4.click(
        lambda: ("### Step 3 of 5", 3),
        outputs=[bc_progress, bc_step],
        js="goTab('bc_tag_tab')",
    )
    bc_next4.click(
        _next_from_params,
        inputs=bc_project,
        outputs=[bc_upload_msg, bc_nav, bc_progress, bc_step],
        js="(msg,nav,prog,step)=>{ if(nav){ goTab('bc_review_tab'); } }",
    )
    bc_train.click(
        _train_ui,
        inputs=[bc_project, bc_model_select, bc_steps, bc_lr],
        outputs=bc_log,
    )
    bc_prev5.click(
        lambda: ("### Step 4 of 5", 4),
        outputs=[bc_progress, bc_step],
        js="goTab('bc_params_tab')",
    )

    prompt.input(_prompt_autocomplete, inputs=prompt, outputs=tag_suggestions)
    tag_suggestions.change(
        _apply_tag,
        inputs=[prompt, tag_suggestions],
        outputs=[prompt, tag_suggestions],
    )

    save_settings_preset_btn.click(
        _show_preset_prompt,
        outputs=[preset_name_box, preset_confirm_btn, preset_status],
    )
    preset_confirm_btn.click(
        _save_preset,
        inputs=[
            preset_name_box,
            seed,
            random_seed_chk,
            steps,
            width,
            height,
            guidance_scale,
            clip_skip,
            scheduler,
            precision_dd,
            tile_chk,
            lora_weight,
            denoising_strength,
            highres_fix_chk,
            nsfw_filter,
            smooth_preview_chk,
            images_per_batch,
            batch_count,
            model_category,
            model,
            lora,
        ],
        outputs=[settings_preset_dd, preset_name_box, preset_confirm_btn, preset_status],
    )
    settings_preset_dd.change(
        _load_preset,
        inputs=settings_preset_dd,
        outputs=[
            seed,
            random_seed_chk,
            steps,
            guidance_scale,
            clip_skip,
            scheduler,
            precision_dd,
            tile_chk,
            lora_weight,
            denoising_strength,
            highres_fix_chk,
            nsfw_filter,
            width,
            height,
            smooth_preview_chk,
            images_per_batch,
            batch_count,
            model_category,
            model,
            lora,
        ],
    )
    remove_settings_preset_btn.click(
        _remove_preset,
        inputs=settings_preset_dd,
        outputs=settings_preset_dd,
    )

    def _select_gen_image(evt: gr.SelectData, paths):
        if evt.index is None or evt.index >= len(paths):
            return gr.update(visible=False)
        path = paths[evt.index]
        try:
            img = Image.open(path)
        except Exception:
            img = None
        return gr.update(value=img, visible=True)

    finished_gallery.select(
        _select_gen_image,
        inputs=finished_gallery_state,
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
