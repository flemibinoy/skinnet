import io

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import CLASSES
from app.data.transforms import transform
from app.explainability.gradcam import GradCAM
from app.models.checkpoints import load_checkpoint


def validate_upload(raw, max_bytes=20 * 1024 * 1024):
    if not raw or len(raw) > max_bytes:
        raise ValueError("Upload must be nonempty and at most 20 MB")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Only decoded JPEG and PNG images are supported")
            if image.width * image.height > 20_000_000 or min(image.size) < 32:
                raise ValueError("Image must be >=32 pixels per side and <=20 megapixels")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or corrupt image") from exc


class Predictor:
    def __init__(self, checkpoint, allow_smoke=False):
        self.model, self.config, self.metadata = load_checkpoint(checkpoint)
        if self.metadata.get("run_type") == "engineering_smoke" and not allow_smoke:
            raise ValueError(
                "Engineering smoke checkpoint is not a research model. Explicitly enable smoke inspection."
            )
        torch.set_num_threads(self.config.threads)

    def predict(self, image):
        tensor = transform(self.config.input_size)(image).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(tensor).softmax(1)[0].numpy()
        return {
            "probabilities": {k: float(v) for k, v in zip(CLASSES, probs)},
            "predicted_class": CLASSES[int(probs.argmax())],
            "confidence": float(probs.max()),
            "entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum()),
        }

    def explain(self, image, target=None):
        tensor = transform(self.config.input_size)(image).unsqueeze(0)
        return GradCAM(self.model).overlay(image, tensor, target)
