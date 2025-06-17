# SDUnity

**SDUnity** is a self‑hosted web interface built with [Gradio](https://www.gradio.app/) for working with Stable Diffusion models. The application brings model management, LoRA handling and an image gallery together under a single UI.

The interface uses a modern dark theme and is divided into five main tabs:

- **Generation** – create images from text prompts
- **Model Manager** – browse models and LoRAs or search Civitai
- **Gallery** – view previous generations
- **Bootcamp** – LoRA training (work in progress)
- **Settings** – configure persistent options

## Features

### Image Generator
- Prompt and Negative Prompt fields
- Seed input with optional randomisation
- Control over steps, width and height
- Smooth Step Streaming preview
- NSFW filter toggle
- Multiple images per batch and batch repetition
- Selectable prompt presets
- Choice of sampler and precision
- LoRA weight slider and tiling option
- Denoising strength control

### Model Management
- Choose from any files in `models/`
- LoRA dropdown lists files from `loras/` with multi‑select support
- Built‑in Civitai browser to search, inspect metadata and download model versions
- Paste a Civitai link to download directly
- Search uses dedicated tags for SD 1.5, SDXL and Pony models for more accurate results
- Move or delete model files from within the manager

### Web Gallery
- Browse all saved images from `generations/`
- Display metadata such as prompt, model, LoRA and seed

### Bootcamp (WIP)
- Simple interface to launch LoRA training using the diffusers DreamBooth script


### Command line Civitai download

To fetch a model directly from the terminal:

```bash
python scripts/civitai_download.py <download_url> [destination] --api-key YOUR_KEY
```

## Setup

Clone the repository and run the maintainer script which installs SDUnity under
`/opt/SDUnity` with its own virtual environment:

```bash
git clone https://github.com/AsaTyr2018/SDUnity.git
cd SDUnity
sudo ./maintainer.sh install
```

Start the web interface with:

```bash
/opt/SDUnity/start.sh
```

The interface runs on `http://localhost:7860/` by default. Gradio launch options can be adjusted from the Settings tab or by editing `sdunity/config.py` under `GRADIO_LAUNCH_CONFIG`.

Prompt presets live in `presets.txt` and can be selected from the dropdown in the Generation tab.

## Maintainer Script

The `maintainer.sh` script also handles updates and removal. Run it with
`sudo` followed by `install`, `update` or `uninstall`. It manages its own
virtual environment under `/opt/SDUnity/venv` and requires `git` and
`python3` to be available.
