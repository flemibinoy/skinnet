"""Presentation adapters for existing local research artifacts; no training or inference here."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from app.config import CLASSES

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
PAGES = ["Inspect an image", "Dataset & evaluation", "Fairness research", "Model & experiments"]


def read_report(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Report must be a JSON object")
        return value
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read report: {exc}") from exc


def default_workspace(root):
    root = Path(root)
    return {
        "checkpoint": str(root / "baseline/best.pt"),
        "evaluation": str(root / "baseline/evaluation"),
        "dataset": str(root / "dataset"),
        "mode": "Research model",
        "allow_smoke": False,
        "label": "Research baseline",
    }


def discover_runs(root):
    runs = []
    for config_path in sorted(Path(root).glob("*/config.json")):
        try:
            config = read_report(config_path)
            if not config or not isinstance(config.get("smoke"), bool):
                continue
            run = config_path.parent
            runs.append(
                {
                    "path": str(run),
                    "label": run.name.replace("-", " ").replace("_", " ").title(),
                    "smoke": config["smoke"],
                    "checkpoint": str(run / "best.pt"),
                }
            )
        except (ValueError, OSError):
            continue
    return runs


def validate_model_purpose(metadata, mode, allow_smoke=False):
    kind = metadata.get("run_type")
    if mode == "Research model" and kind != "research":
        raise ValueError(
            "This checkpoint is not a recorded research model. Select Engineering smoke test to inspect engineering runs."
        )
    if mode == "Engineering smoke test" and (kind != "engineering_smoke" or not allow_smoke):
        raise ValueError(
            "Engineering inspection requires a smoke checkpoint and explicit acknowledgement."
        )
    if mode not in ("Research model", "Engineering smoke test"):
        raise ValueError("Unsupported model purpose")


def inspection_signature(raw, workspace):
    path = Path(workspace["checkpoint"])
    stat = path.stat() if path.is_file() else None
    return (
        hashlib.sha256(raw).hexdigest(),
        str(path),
        (stat.st_mtime_ns, stat.st_size) if stat else None,
        workspace["mode"],
        workspace["allow_smoke"],
    )


def class_distribution(audit):
    counts = audit.get("class_counts")
    if not isinstance(counts, dict) or set(counts) - set(CLASSES):
        raise ValueError("Class counts do not match the supported HAM10000 categories")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
        raise ValueError("Class counts must be nonnegative integers")
    return pd.DataFrame(
        [
            {"Category": c.upper(), "Lesion category": CLASS_NAMES[c], "Images": counts[c]}
            for c in CLASSES
            if c in counts
        ]
    )


def split_distribution(report):
    counts = report.get("counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("Split counts are missing")
    rows = []
    for key, label in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        values = [v.get(key) for v in counts.values() if isinstance(v, dict)]
        if len(values) != len(counts) or any(not isinstance(n, int) or n < 0 for n in values):
            raise ValueError("Split counts are incomplete or invalid")
        rows.append({"Partition": label, "Images": sum(values)})
    return pd.DataFrame(rows)


def validate_evaluation(report, checkpoint=None):
    if not isinstance(report.get("provenance"), dict) or report["provenance"].get(
        "run_type"
    ) not in ("research", "engineering_smoke"):
        raise ValueError("Evaluation provenance is missing or unsupported")
    if report["provenance"].get("classes") != CLASSES:
        raise ValueError("Evaluation class mapping is incompatible")
    n = report.get("n")
    if not isinstance(n, int) or n < 1 or not isinstance(report.get("per_class"), dict):
        raise ValueError("Evaluation sample count or class metrics missing")
    if set(report["per_class"]) != set(CLASSES):
        raise ValueError("Evaluation must describe all seven supported categories")
    for value in (report.get("accuracy"), report.get("macro", {}).get("f1"), report.get("ece")):
        if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
            raise ValueError("Evaluation metric is outside its valid range")
    cm = report.get("confusion_matrix")
    if (
        not isinstance(cm, list)
        or len(cm) != 7
        or any(
            not isinstance(row, list)
            or len(row) != 7
            or any(not isinstance(v, int) or v < 0 for v in row)
            for row in cm
        )
    ):
        raise ValueError("Confusion matrix must contain seven-by-seven nonnegative counts")
    if sum(sum(row) for row in cm) != n:
        raise ValueError("Confusion matrix and sample count disagree")
    if checkpoint and Path(checkpoint).is_file():
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if digest != report["provenance"].get("checkpoint_sha256"):
            raise ValueError("Evaluation belongs to a different checkpoint")
    return report


def fairness_verified(report):
    if not report or report.get("status") != "measured":
        return False
    p = report.get("provenance")
    return (
        isinstance(p, dict)
        and p.get("verified") is True
        and all(p.get(k) for k in ("method", "source", "task_compatibility"))
        and isinstance(report.get("groups"), dict)
        and bool(report["groups"])
    )


def subgroup_rows(report):
    rows = []
    for name, group in report["groups"].items():
        if not isinstance(group, dict) or not isinstance(group.get("n"), int) or group["n"] < 1:
            raise ValueError("Invalid subgroup sample count")
        clusters = group.get("independent_clusters")
        # This is a conservative UI display gate, not a scientific power calculation.
        eligible = (
            group["n"] >= 20
            and isinstance(clusters, int)
            and clusters >= 5
            and not group.get("absent_classes", CLASSES)
        )
        rows.append(
            {
                "Group": str(name),
                "Images": group["n"],
                "Independent groups": clusters,
                "Accuracy": group.get("accuracy") if eligible else None,
                "Macro F1": group.get("macro", {}).get("f1") if eligible else None,
                "Display status": "Exploratory"
                if eligible
                else "Insufficient support for comparison",
            }
        )
    return pd.DataFrame(rows)
