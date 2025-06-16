import os
import json
from datetime import datetime
from PIL import Image

from . import config


def save_generation(image: Image.Image, metadata: dict) -> str:
    """Save generated image and metadata to the generations directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date_dir = os.path.join(config.GENERATIONS_DIR, timestamp.split("_")[0])
    os.makedirs(date_dir, exist_ok=True)
    filename = f"{timestamp}_{metadata.get('seed', '0')}.png"
    img_path = os.path.join(date_dir, filename)
    image.save(img_path)
    meta_path = img_path.replace(".png", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return img_path


def load_metadata(img_path: str) -> dict:
    meta_path = img_path.replace(".png", ".json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
