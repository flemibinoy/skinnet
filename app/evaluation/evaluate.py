import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.config import CLASSES
from app.data.transforms import ManifestDataset
from app.data.validation import check_splits, verify_files
from app.evaluation.metrics import classification_metrics
from app.evaluation.reports import save_report
from app.models.checkpoints import load_checkpoint
from app.training.reproducibility import device_for, seed_all


def evaluate(checkpoint, manifest, output, device="auto"):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be empty to preserve prior results")
    device = device_for(device)
    model, config, metadata = load_checkpoint(checkpoint, device)
    raw = Path(manifest).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata.get("manifest_sha256"):
        raise ValueError("Manifest differs from frozen training manifest")
    seed_all(config.seed, config.threads)
    df = pd.read_csv(manifest)
    check_splits(df)
    test = df[df.split.eq("test")].reset_index(drop=True)
    verify_files(test)
    loader = DataLoader(
        ManifestDataset(test, config.input_size),
        batch_size=config.batch_size,
        num_workers=config.workers,
    )
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            probs.extend(model(images.to(device)).softmax(1).cpu().tolist())
    p = np.array(probs)
    metrics = classification_metrics(test.label.to_numpy(), p)
    predictions = test[["image_id", "lesion_id", "group_id", "label"]].copy()
    predictions["predicted"], predictions["confidence"] = p.argmax(1), p.max(1)
    for i, c in enumerate(CLASSES):
        predictions[f"p_{c}"] = p[:, i]
    provenance = {
        **metadata,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "manifest_sha256": digest,
        "test_ids_sha256": hashlib.sha256("\n".join(test.image_id).encode()).hexdigest(),
    }
    save_report(out, metrics, predictions, provenance)
    return metrics
