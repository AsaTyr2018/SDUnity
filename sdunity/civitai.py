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


def search_models(query: str = "", model_type: str = "sd15", sort: str = "Most Downloaded", limit: int = 20):
    """Search models on Civitai and filter by base model."""
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
        versions = item.get("modelVersions") or []
        if not versions:
            continue
        ver = versions[0]
        if ver.get("baseModel") != base:
            continue
        download_url = ver.get("downloadUrl")
        if not download_url:
            continue
        image = None
        images = ver.get("images") or []
        if images:
            image = images[0].get("url")
        items.append({
            "name": item.get("name"),
            "versionId": ver.get("id"),
            "downloadUrl": download_url,
            "image": image,
        })
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
        progress(0, desc=f"Downloading {filename}", total=total)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if progress is not None and total:
                progress(downloaded, desc=f"Downloading {filename}", total=total)
    if progress is not None:
        progress(total, desc="Download complete", total=total)
    return dest
