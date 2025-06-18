import os
import json

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
LORA_DIR = os.path.join(BASE_DIR, "loras")
GENERATIONS_DIR = os.path.join(BASE_DIR, "generations")
WILDCARDS_DIR = os.path.join(BASE_DIR, "wildcards")


# Ensure output directories exist
os.makedirs(GENERATIONS_DIR, exist_ok=True)
os.makedirs(WILDCARDS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Gradio Launch Configuration
# ---------------------------------------------------------------------------
# Central location for default `Blocks.launch` arguments. Adjust values here to
# change how the Gradio server starts. See the Gradio docs for explanation of
# each option.
ALLOWED_PATHS = [GENERATIONS_DIR]

GRADIO_LAUNCH_CONFIG = {
    # Networking
    "server_name": "0.0.0.0",  # listen on all interfaces
    "server_port": None,       # default port (7860)
    "share": False,            # set True to create a public link

    # Interface behaviour
    "inbrowser": False,        # open browser automatically on launch
    "show_error": False,       # display errors in the UI
    "debug": False,            # enable debug logs
    "max_threads": 40,         # maximum thread workers

    # Security and authentication
    "auth": None,              # tuple of (user, pass) or callable
    "auth_message": None,      # message shown on auth prompt
    "ssl_keyfile": None,       # path to SSL key
    "ssl_certfile": None,      # path to SSL certificate

    # Miscellaneous
    "quiet": False,            # reduce terminal output
    "show_api": True,          # expose REST API docs
    "allowed_paths": ALLOWED_PATHS,  # directories accessible via /file=
}

# ---------------------------------------------------------------------------
# User Configuration
# ---------------------------------------------------------------------------
# Settings in this section are persisted to ``config/user_config.json`` so that
# changes made in the UI survive restarts.

USER_CONFIG_PATH = os.path.join(BASE_DIR, "config", "user_config.json")

# Default values for the user configuration. We include all Gradio launch
# options so they can be customized from the UI along with the Civitai API key.
DEFAULT_USER_CONFIG = {"civitai_api_key": "", **GRADIO_LAUNCH_CONFIG}


def load_user_config() -> dict:
    """Load the persistent user configuration from disk."""
    if os.path.isfile(USER_CONFIG_PATH):
        with open(USER_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    cfg = DEFAULT_USER_CONFIG.copy()
    cfg.update(data)
    return cfg


USER_CONFIG = load_user_config()


def save_user_config(cfg: dict | None = None) -> None:
    """Persist the provided user configuration to disk."""
    if cfg is None:
        cfg = USER_CONFIG
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# Update the launch config based on the loaded user settings
for key in GRADIO_LAUNCH_CONFIG:
    if key in USER_CONFIG:
        GRADIO_LAUNCH_CONFIG[key] = USER_CONFIG[key]
