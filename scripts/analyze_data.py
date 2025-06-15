import os
from PIL import Image

def analyze_image_folder(folder):
    stats = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(folder, fname)
        with Image.open(path) as img:
            stats.append(img.size)
    if not stats:
        print("No images found")
        return
    widths, heights = zip(*stats)
    print(f"Total images: {len(stats)}")
    print(f"Average width: {sum(widths)/len(widths):.1f}")
    print(f"Average height: {sum(heights)/len(heights):.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze image folder")
    parser.add_argument(
        "folder",
        nargs="?",
        default="data",
        help="Folder with images",
    )
    args = parser.parse_args()
    analyze_image_folder(args.folder)
