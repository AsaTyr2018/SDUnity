# SDUnity

**SDUnity** is a self‑hosted web interface built with [Gradio](https://www.gradio.app/) for working with Stable Diffusion models. The application brings model management, LoRA handling and an image gallery together under a single UI.

The interface uses a modern dark theme and is divided into four main tabs:

- **Generation** – create images from text prompts
- **Model Manager** – browse models and LoRAs or search Civitai
- **Gallery** – view previous generations
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

### Model Management
- Choose between predefined models or any files in `models/`
- LoRA dropdown lists files from `loras/` with multi‑select support
- Built‑in Civitai browser to search and download checkpoints
- Move or delete model files from within the manager

### Web Gallery
- Browse all saved images from `generations/`
- Display metadata such as prompt, model, LoRA and seed

## Built‑in Models

Example checkpoints are defined in `config/model_registry.json`. Download them with:

```bash
python scripts/download_models.py
```

Any `.safetensors` or `.ckpt` files placed in `models/` are detected automatically.

### Command line Civitai download

To fetch a model directly from the terminal:

```bash
python scripts/civitai_download.py <download_url> [destination] --api-key YOUR_KEY
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) analyse an image folder:
   ```bash
   python scripts/analyze_data.py path/to/images
   ```
3. Create folders for models and LoRAs and download the predefined checkpoints:
   ```bash
   mkdir -p models loras
   python scripts/download_models.py
   ```
4. Launch the app:
   ```bash
   python app.py
   ```

The interface runs on `http://localhost:7860/` by default. Gradio launch options can be adjusted from the Settings tab or by editing `sdunity/config.py` under `GRADIO_LAUNCH_CONFIG`.

Prompt presets live in `presets.txt` and can be selected from the dropdown in the Generation tab.
