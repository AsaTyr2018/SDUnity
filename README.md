# SDUnity

This project is a prototype interface for Stable Diffusion built with Gradio.
It features a demo generator UI with prompt and negative prompt fields as well as
controls for seed, steps and resolution. The app can list models and LoRA files
from the `models/` and `loras/` folders and displays generated images in a
gallery.

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
