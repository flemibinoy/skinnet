import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from app.config import CLASSES


def validate_images(df):
    df = df.copy()
    issues, dimensions, hashes, pixels = [], Counter(), [], []

    def inspect(row):
        digest = pixel = ""
        dimension = None
        error = None
        try:
            if row.path_count != 1:
                raise ValueError(f"Expected one image file; found {row.path_count}")
            p = Path(row.path)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with Image.open(p) as image:
                image.load()
                rgb = image.convert("RGB")
                pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                dimension = str(image.size)
        except (OSError, ValueError) as exc:
            error = {"image_id": row.image_id, "error": str(exc)}
        return digest, pixel, dimension, error

    # Two native Pillow workers keep memory modest while overlapping decoding and I/O.
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, (digest, pixel, dimension, error) in enumerate(
            executor.map(inspect, df.itertuples())
        ):
            if index % 1000 == 0:
                print(f"Validated image {index + 1}/{len(df)}", flush=True)
            if error:
                issues.append(error)
            if dimension:
                dimensions[dimension] += 1
            hashes.append(digest)
            pixels.append(pixel)
    df["sha256"], df["pixel_sha256"] = hashes, pixels
    duplicates = [
        g.image_id.tolist()
        for _, g in df[df.pixel_sha256.ne("")].groupby("pixel_sha256")
        if len(g) > 1
    ]
    conflicts = [ids for ids in duplicates if df[df.image_id.isin(ids)].dx.nunique() > 1]
    report = {
        "rows": len(df),
        "inventory": df.attrs.get("inventory", {}),
        "expected_rows": 10015,
        "official_count_matches": len(df) == 10015,
        "columns": list(df.columns),
        "class_counts": df.dx.value_counts().to_dict(),
        "missing_classes": sorted(set(CLASSES) - set(df.dx)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "dimensions": dict(dimensions),
        "issues": issues,
        "exact_duplicate_images": duplicates,
        "duplicate_label_conflicts": conflicts,
        "identifiers": [c for c in ("patient_id", "lesion_id") if c in df],
        "skin_tone": {
            "status": "unavailable",
            "reason": "HAM10000 has no verified skin-tone annotations",
        },
        "acquisition": {"modality": "dermoscopy", "source_by_image": "not established by metadata"},
        "valid": not issues and not conflicts,
    }
    return df, report


def connected_groups(df):
    """Union known patients, lesions, identical file bytes and identical decoded pixels."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in ("patient_id", "lesion_id", "sha256", "pixel_sha256"):
        if col not in df:
            continue
        seen = {}
        for i, value in enumerate(df[col]):
            if pd.isna(value) or not str(value).strip():
                continue
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    return np.array([f"component_{find(i)}" for i in range(len(df))])


def split_manifest(df, seed=42):
    df = df.sort_values("image_id").reset_index(drop=True).copy()
    df["group_id"] = connected_groups(df)
    for label in range(7):
        if df.loc[df.label.eq(label), "group_id"].nunique() < 10:
            raise ValueError("Ten-fold split requires >=10 independent groups per class")
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    df["split"] = "train"
    for fold, (_, indices) in enumerate(folds.split(df, df.label, df.group_id)):
        if fold < 2:
            df.loc[indices, "split"] = "test" if fold == 0 else "val"
    check_splits(df)
    for split in ("train", "val", "test"):
        if set(df.loc[df.split.eq(split), "label"]) != set(range(7)):
            raise ValueError(
                f"{split} lacks classes; inspect group/class distribution rather than silently excluding"
            )
    return df


def check_splits(df):
    required = {
        "image_id",
        "path",
        "label",
        "group_id",
        "split",
        "sha256",
        "pixel_sha256",
        "lesion_id",
    }
    if not required <= set(df):
        raise ValueError(f"Manifest missing columns: {required - set(df)}")
    if df.empty or df.image_id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Empty, duplicate or incomplete manifest")
    if set(df.split) != {"train", "val", "test"} or not df.label.isin(range(7)).all():
        raise ValueError("Invalid splits or labels")
    for col in ("group_id", "patient_id", "lesion_id", "sha256", "pixel_sha256"):
        if col in df:
            valid = df[df[col].notna() & df[col].astype(str).str.strip().ne("")]
            if valid.groupby(col).split.nunique().gt(1).any():
                raise ValueError(f"Leakage across splits in {col}")


def verify_files(df):
    for row in df.itertuples():
        if hashlib.sha256(Path(row.path).read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Image changed since validation: {row.image_id}")
