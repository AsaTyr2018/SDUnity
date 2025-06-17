import argparse
import importlib.util
import os
from pathlib import Path

_TAG_PATH = Path(__file__).resolve().parent.parent / "sdunity" / "tags.py"
spec = importlib.util.spec_from_file_location("sdunity_tags", _TAG_PATH)
tags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tags)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge tags from a1111-sd-webui-tagcomplete into the local dataset"
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default=os.environ.get("TAGCOMPLETE_REPO", "data/tagcomplete"),
        help=(
            "Path to the cloned a1111-sd-webui-tagcomplete repository. "
            "Defaults to $TAGCOMPLETE_REPO or 'data/tagcomplete'."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/all_tags.csv",
        help="Destination CSV file"
    )
    args = parser.parse_args()

    tags.update_dataset_from_tagcomplete(args.repo, args.output)
    print(f"Updated tag dataset written to {args.output}")


if __name__ == "__main__":
    main()
