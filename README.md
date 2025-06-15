# SDUnity

This project is a prototype interface for Stable Diffusion built with Gradio.
It features a demo generator UI with prompt and negative prompt fields as well as
controls for seed, steps and resolution. The app can list models and LoRA files
from the `models/` and `loras/` folders and displays generated images in a
gallery.

## Available Models

Built-in presets are provided for several Hugging Face models. Any model files
placed in the `models/` folder (e.g. `.safetensors` or `.ckpt`) are also offered
in the selector.

| Modelltyp | Modell (Hugging Face) | Besonderheiten |
| --------- | --------------------- | -------------- |
| **SD 1.5** | runwayml/stable-diffusion-v1-5 | Offizielles Basis-Modell |
| | hakurei/waifu-diffusion | Anime-Fokus |
| | SG161222/Realistic_Vision_V2.0 | Realistischer Look |
| **SDXL** | stabilityai/stable-diffusion-xl-base-1.0 | SDXL Basis |
| | RunDiffusion/Juggernaut-XL | Vielseitig & beliebt |
| **PonyXL** | stablediffusionapi/pony-diffusion-v6-xl | Ponys & Anthro |
| | glides/ponyxl | Gute Basis für Pferde |

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Analyze your image folder:
   ```bash
   python scripts/analyze_data.py path/to/images
   ```
3. Create folders for your models and LoRAs if they don't exist:
   ```bash
   mkdir -p models loras
   ```
4. Run the demo app:
   ```bash
   python app.py
   ```
