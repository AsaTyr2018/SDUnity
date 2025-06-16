import os

# Directory paths
MODELS_DIR = "models"
LORA_DIR = "loras"
MODEL_REGISTRY_PATH = os.path.join("config", "model_registry.json")
GENERATIONS_DIR = "generations"

# Ensure generations directory exists
os.makedirs(GENERATIONS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Gradio Launch Configuration
# ---------------------------------------------------------------------------
# Central location for default `Blocks.launch` arguments. Adjust values here to
# change how the Gradio server starts. See the Gradio docs for explanation of
# each option.
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
}
