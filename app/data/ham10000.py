from pathlib import Path

import pandas as pd

from app.config import CLASS_MAP


class HAM10000:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def load(self):
        matches = list(self.root.rglob("HAM10000_metadata.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one HAM10000_metadata.csv; found {len(matches)}")
        df = pd.read_csv(matches[0], dtype={"image_id": str, "lesion_id": str, "dx": str})
        required = {"image_id", "lesion_id", "dx"}
        if not required <= set(df.columns):
            raise ValueError(f"Missing metadata columns: {sorted(required - set(df.columns))}")
        if df.empty or df[list(required)].isna().any().any():
            raise ValueError("Empty metadata or missing required labels/identifiers")
        if df.image_id.duplicated().any():
            raise ValueError("Duplicate image_id metadata rows")
        if not df.dx.isin(CLASS_MAP).all():
            raise ValueError("Unsupported diagnosis labels; no automatic remapping")
        for col in required:
            if df[col].str.strip().eq("").any():
                raise ValueError(f"Blank {col}")
        paths = {}
        for p in self.root.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.setdefault(p.stem, []).append(str(p))
        df["path"] = [paths.get(i, [""])[0] for i in df.image_id]
        df["path_count"] = [len(paths.get(i, [])) for i in df.image_id]
        df["label"] = df.dx.map(CLASS_MAP)
        df.attrs["inventory"] = {
            "image_file_count": sum(len(items) for items in paths.values()),
            "unreferenced_image_ids": sorted(set(paths) - set(df.image_id)),
            "duplicate_filenames": {key: items for key, items in paths.items() if len(items) > 1},
        }
        return df
