# SDUnity

SDUnity is a self-hosted web interface built with [Gradio](https://www.gradio.app/) for generating and managing images with Stable Diffusion models. It combines model selection, LoRA management and an image gallery into a single application.

## Features

### THE Generator
- Prompt and Negative Prompt inputs
- Real-time seed control
- Aspect ratio and sampling step sliders
 - Optional Smooth Preview mode showing intermediate steps as a GIF

### Model Selector
- Choose between predefined Hugging Face models or any files placed in `models/`
- Supports Stable Diffusion 1.5, SDXL and PonyXL

### LoRA Library
- Automatically lists LoRAs found in the `loras/` directory
- Optional metadata, tags and preview images
- Selected LoRAs are injected directly into the prompt

### WebGallery
- Browsable gallery of generated images
- Shows prompt, model, LoRA and timestamp
- Filter and sort by model, LoRA, date and tags

## Built-in Models

All predefined models are listed in `config/model_registry.json`. Run
`python scripts/download_models.py` to download them into the `models/` folder
sorted by category. Any additional `.safetensors` or `.ckpt` files placed in
`models/` are automatically detected by the app.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Optional: analyze your image folder to view statistics:
   ```bash
   python scripts/analyze_data.py path/to/images
   ```
3. Ensure folders for models and LoRAs exist and download the predefined models:
   ```bash
   mkdir -p models loras
   python scripts/download_models.py
   ```
4. Launch the app:
   ```bash
   python app.py
   ```

The interface will be available at `http://localhost:7860/` by default. Use the controls to generate images and browse them in the gallery.

Presets for prompt enhancements are stored in `presets.txt` and can be selected from the dropdown in the UI.

