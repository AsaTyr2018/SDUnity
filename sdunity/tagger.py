from __future__ import annotations

"""WD14-based image tagging backend."""

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    from timm.models.swin_transformer_v2 import SwinTransformerV2
    if not hasattr(SwinTransformerV2, "_initialize_weights") and hasattr(SwinTransformerV2, "_init_weights"):
        SwinTransformerV2._initialize_weights = SwinTransformerV2._init_weights
except Exception:
    # ``timm`` might not be installed during documentation builds
    pass


class WD14Tagger:
    """Tag images using the WD14 anime tagging model."""

    def __init__(self, model_name: str = "SmilingWolf/wd-v1-4-swinv2-tagger-v2") -> None:
        # ``SmilingWolf/wd-v1-4-swinv2-tagger-v2`` relies on custom code stored
        # in the model repository. ``transformers`` requires ``trust_remote_code``
        # to load such models correctly.
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    def tag_image(self, image: Image.Image, threshold: float = 0.35) -> list[str]:
        """Return tags for ``image`` above ``threshold``."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = logits.sigmoid()[0]
        tags = [label for label, score in zip(self.labels, probs) if score.item() > threshold]
        tags.sort()
        return tags

    def tag_file(self, path: str, threshold: float = 0.35) -> list[str]:
        """Open image at ``path`` and return tags."""
        img = Image.open(path).convert("RGB")
        return self.tag_image(img, threshold)
