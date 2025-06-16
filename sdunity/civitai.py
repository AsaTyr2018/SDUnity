import os
import re
import requests

from . import config

BASE_MODEL_MAP = {
    "sd15": "SD 1.5",
    "sdxl": "SDXL 1.0",
    "ponyxl": "Pony",
}

API_URL = "https://civitai.com/api/v1/models"

# API key loaded from the user configuration
API_KEY = config.USER_CONFIG.get("civitai_api_key", "")


def set_api_key(key: str) -> None:
    """Update the in-memory API key."""
    global API_KEY
    API_KEY = key


def _headers() -> dict:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def search_models(
    query: str = "",
    model_type: str = "sd15",
    sort: str = "Most Downloaded",
    limit: int = 70,
):
    """Search models on Civitai and return metadata and versions.

    If ``query`` is provided, the API's ``query`` parameter is used to
    filter results by name.
    """

    params = {
        "types": "Checkpoint",
        "limit": limit,
        "sort": sort,
    }
    if query:
        params["query"] = query

    resp = requests.get(API_URL, params=params, timeout=30, headers=_headers())
    resp.raise_for_status()
    data = resp.json()

    base = BASE_MODEL_MAP.get(model_type, "SD 1.5")
    items = []

    for item in data.get("items", []):
        versions = []
        for ver in item.get("modelVersions") or []:
            if ver.get("baseModel") != base:
                continue
            url = ver.get("downloadUrl")
            if not url:
                continue
            img = None
            images = ver.get("images") or []
            if images:
                img = images[0].get("url")
            versions.append(
                {
                    "id": ver.get("id"),
                    "name": ver.get("name"),
                    "downloadUrl": url,
                    "image": img,
                    "trainedWords": ver.get("trainedWords"),
                    "description": ver.get("description"),
                    "stats": ver.get("stats", {}),
                }
            )

        if not versions:
            continue

        preview = versions[0].get("image")

        items.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "description": item.get("description"),
                "stats": item.get("stats", {}),
                "versions": versions,
                "preview": preview,
            }
        )

    return items


def _extract_filename(resp: requests.Response, url: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', cd)
    if match:
        return match.group(1)
    return os.path.basename(url.split("?")[0])


def download_model(download_url: str, dest_dir: str, progress=None) -> str:
    """Download model file to dest_dir and return filepath."""
    os.makedirs(dest_dir, exist_ok=True)
    resp = requests.get(download_url, stream=True, timeout=60, headers=_headers())
    resp.raise_for_status()
    filename = _extract_filename(resp, download_url)
    dest = os.path.join(dest_dir, filename)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    if progress is not None:
        progress((0, total), desc=f"Downloading {filename}")
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if progress is not None and total:
                progress((downloaded, total), desc=f"Downloading {filename}")
    if progress is not None:
        progress((total, total), desc="Download complete")
    return dest


def resolve_download_link(link: str) -> str:
    """Return a direct download URL from a Civitai link or page."""
    if "api/download/models" in link:
        return link

    match = re.search(r"modelVersionId=(\d+)", link)
    if match:
        return f"https://civitai.com/api/download/models/{match.group(1)}"

    match = re.search(r"/models/(\d+)", link)
    if match:
        model_id = match.group(1)
        resp = requests.get(
            f"https://civitai.com/api/v1/models/{model_id}",
            timeout=30,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        versions = data.get("modelVersions") or []
        if versions:
            return versions[0].get("downloadUrl")

    resp = requests.get(link, timeout=30, headers=_headers())
    resp.raise_for_status()
    match = re.search(r"https://civitai.com/api/download/models/\d+", resp.text)
    if match:
        return match.group(0)
    raise ValueError("Unable to resolve download link")


def download_by_link(link: str, dest_dir: str, progress=None) -> str:
    """Download a model using any Civitai link."""
    url = resolve_download_link(link)
    return download_model(url, dest_dir, progress=progress)


def format_metadata(model: dict, version: dict) -> str:
    """Return a simple markdown string summarising model metadata."""

    lines = []
    if model.get("description"):
        lines.append(model["description"].split("\n")[0])
    stats = model.get("stats", {})
    if stats.get("downloadCount"):
        lines.append(f"Downloads: {stats['downloadCount']}")
    if stats.get("rating"):
        lines.append(f"Rating: {stats['rating']:.1f}")
    if version.get("trainedWords"):
        words = ", ".join(version["trainedWords"])
        lines.append(f"Trigger: {words}")
    return "\n".join(lines)
