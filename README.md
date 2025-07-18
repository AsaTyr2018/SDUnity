# SDUnity (WORK IN PROGRESS!)

## Index
- [Overview](#overview)
- [Features](#features)
  - [Generator](#generator)
  - [Model Manager](#model-manager)
  - [Gallery](#gallery)
  - [Civitai Integration](#civitai-integration)
  - [Image Tagger](#image-tagger)
  - [Tag Suggestions](#tag-suggestions)
- [Setup](#setup)
- [Maintainer Script](#maintainer-script)
- [Command Line Tools](#command-line-tools)

## Overview
**SDUnity** is a modern self‑hosted web interface built with [Gradio](https://www.gradio.app/) for working with Stable Diffusion models. It unifies model management, LoRA handling and a gallery into one application.

The UI is split into tabs for generation, model management, a gallery and a settings page.

## Features
### Generator
- Prompt and negative prompt inputs
- Random or fixed seed support (enabled by default)
- Control over steps, width, height and guidance scale
- Clip skip and sampler selection with more sampler choices
- Available samplers: Euler, Euler a, DDIM, DPM++ 2M Karras, DPM++ 2M, PNDM, LMS, Heun, DPM++ SDE Karras
- Precision toggle and tiling option
- Multiple images per batch (parallel generation) and batch repetition
- Smooth step streaming preview (WIP)
- High‑res fix and denoising strength controls
- LoRA weight slider
- Wildcard prompts using `__name__` tokens that pull lines from files in `wildcards/`
- Optional GPT‑2 based prompt enhancer that augments your prompt (WIP)
- NSFW filter toggle
- Basic and advanced inference settings grouped for easier tweaking

### Model Manager
- Browse models under `models/` and LoRAs under `loras/`
- Categorise, move or delete model files
- Load or unload pipelines to VRAM
- Built‑in Civitai browser with search and metadata preview
- Download models via search results or direct link

### Gallery (WIP)
- Browse all saved images from `generations/`
- View metadata such as prompt, model, LoRA and seed
- Delete unwanted images


### Civitai Integration
- Search Civitai with dedicated tags for SD 1.5, SDXL and Pony models
- Inspect versions and previews before downloading

### Tag Suggestions
- Auto complete prompt tags using a local dataset, directly inside the prompt field.
- Suggestions include fuzzy matches powered by `rapidfuzz`.
- The maintainer script clones
  [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)
  and rebuilds the dataset via `scripts/import_tagcomplete.py`.
- Press **Tab** while typing to insert the top suggestion without leaving the prompt box.
### Image Tagger
- Automatically tag anime-style images using the WD14 tagger model


## Setup
Clone the repository and run the maintainer script to install SDUnity under `/opt/SDUnity` with its own virtual environment:

```bash
git clone https://github.com/AsaTyr2018/SDUnity.git
cd SDUnity
sudo ./maintainer.sh install
```

Start the interface with:

```bash
/opt/SDUnity/start.sh
```

The web UI is available on `http://localhost:7860/` by default. Launch options can be adjusted from the Settings tab or by editing `sdunity/config.py`.
Local overrides reside in `config/user_config.json`; this file is loaded on startup and currently sets `server_port` to `7861`.

Prompt styles live in the `styles/` directory as JSON files.
Model loader presets are stored in the `presets/` directory and
include default model/LoRA choices along with generation settings
such as guidance scale and aspect ratio.

## Maintainer Script
`maintainer.sh` also handles updates and removal. Run it with `sudo` followed by `install`, `update` or `uninstall`. It manages a virtual environment under `/opt/SDUnity/venv` and requires `git` and `python3`.

## Command Line Tools
Use `scripts/civitai_download.py` to fetch a model directly from the terminal:

```bash
python scripts/civitai_download.py <download_url> [destination] --api-key YOUR_KEY
```
