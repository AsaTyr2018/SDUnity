import os

# Directory paths
MODELS_DIR = "models"
LORA_DIR = "loras"
MODEL_REGISTRY_PATH = os.path.join("config", "model_registry.json")
GENERATIONS_DIR = "generations"

# Ensure generations directory exists
os.makedirs(GENERATIONS_DIR, exist_ok=True)
