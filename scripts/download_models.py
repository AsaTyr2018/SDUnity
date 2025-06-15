import argparse
import json
import os
from urllib import request

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join("config", "model_registry.json")


def load_registry(path=REGISTRY_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_file(url, dest):
    """Download a file from url to dest with basic error handling."""
    try:
        request.urlretrieve(url, dest)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_all():
    registry = load_registry()
    for category, models in registry.items():
        cat_dir = os.path.join(MODELS_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        for name, info in models.items():
            url = info.get("url")
            if not url:
                continue
            filename = os.path.basename(url)
            dest = os.path.join(cat_dir, filename)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print(f"{filename} already exists, skipping")
                continue
            print(f"Downloading {filename} to {dest}")
            try:
                download_file(url, dest)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models defined in the registry")
    _ = parser.parse_args()
    download_all()
