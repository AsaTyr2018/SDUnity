import argparse
import os
from sdunity import civitai, config


def cli_progress(x: int, desc: str = "", total: int = 0) -> None:
    """Simple console progress printer."""
    if total:
        pct = x / total * 100
        print(f"\r{desc} {pct:.1f}% ({x}/{total})", end="", flush=True)
    else:
        print(f"\r{desc} {x} bytes", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Civitai model")
    parser.add_argument("url", help="Civitai download URL")
    parser.add_argument("dest", nargs="?", default=config.MODELS_DIR, help="Destination directory")
    parser.add_argument("--api-key", dest="api_key", help="Civitai API key", default=None)
    args = parser.parse_args()

    if args.api_key:
        civitai.set_api_key(args.api_key)

    os.makedirs(args.dest, exist_ok=True)
    path = civitai.download_model(args.url, args.dest, progress=cli_progress)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
