import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.config import CLASSES


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def save_report(out, metrics, predictions, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics["provenance"] = provenance
    write_json(out / "metrics.json", metrics)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(metrics["per_class"]).T.to_csv(out / "per_class.csv")
    errors = predictions[predictions.label.ne(predictions.predicted)]
    errors.to_csv(out / "errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(metrics["confusion_matrix"], cmap="Purples")
    ax.set(
        xticks=range(7),
        yticks=range(7),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="Ground truth",
        title="Held-out confusion matrix",
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    points = [b for b in metrics["reliability_bins"] if b["count"]]
    ax.plot([b["confidence"] for b in points], [b["accuracy"] for b in points], "o-")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="Equal-width reliability bins")
    fig.savefig(out / "calibration.png")
    plt.close(fig)
    out.joinpath("report.md").write_text(
        f"# Evaluation\n\nRun type: {provenance['run_type']}\n\nAccuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['macro']['f1']}\n\n"
        "These results describe this checkpoint and fixed cohort only. Smoke results are engineering checks. "
        "No skin-tone fairness conclusions or clinical validity are established. See per_class.csv, errors.csv and metrics.json.\n"
    )
