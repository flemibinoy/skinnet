import os
from pathlib import Path

import torch

from app.config import CLASSES, TrainConfig
from app.models.classifier import build_model


def save_checkpoint(path, model, config, metadata, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": 1,
        "architecture": "efficientnet_b0",
        "classes": CLASSES,
        "config": config.dict(),
        "metadata": metadata,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        value["optimizer"] = optimizer.state_dict()
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    try:
        value = torch.load(path, map_location=device, weights_only=True)
        if (
            value["schema"] != 1
            or value["architecture"] != "efficientnet_b0"
            or value["classes"] != CLASSES
        ):
            raise ValueError("Checkpoint schema, architecture or class map mismatch")
        config = TrainConfig(**value["config"])
        config.validate()
        if "partial_epoch" in value["metadata"]:
            raise ValueError(
                "Partial interrupted state is for inspection only; use last.pt or best.pt"
            )
        if value["metadata"].get("epoch", 0) < 1:
            raise ValueError("Checkpoint has no completed training epoch")
        model = build_model(False).to(device)
        model.load_state_dict(value["state_dict"], strict=True)
        model.eval()
        return model, config, value["metadata"]
    except Exception as exc:
        raise ValueError(f"Invalid SkinNet+ checkpoint: {exc}") from exc
