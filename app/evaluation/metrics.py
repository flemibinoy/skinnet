import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from app.config import CLASSES


def safe_div(a, b):
    return float(a / b) if b else None


def classification_metrics(labels, probabilities, bins=15):
    if bins < 1:
        raise ValueError("Calibration bins must be positive")
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or len(y) == 0 or p.shape != (len(y), 7):
        raise ValueError("Expected nonempty labels and Nx7 probabilities")
    if (
        not np.isin(y, range(7)).all()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid labels or probabilities")
    y = y.astype(int)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(7))
    per = {}
    for i, name in enumerate(CLASSES):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        fp, fn = predicted - tp, support - tp
        tn = len(y) - tp - fp - fn
        binary = y == i
        per[name] = {
            "precision": safe_div(tp, predicted),
            "recall": safe_div(tp, support),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            "support": support,
            "specificity": safe_div(tn, tn + fp),
            "false_negative_rate": safe_div(fn, support),
            "roc_auc_ovr": float(roc_auc_score(binary, p[:, i])) if 0 < support < len(y) else None,
            "average_precision": float(average_precision_score(binary, p[:, i]))
            if 0 < support < len(y)
            else None,
        }
    macro, weighted, eligible = {}, {}, {}
    for key in ("precision", "recall", "f1"):
        values = [(v[key], v["support"]) for v in per.values() if v[key] is not None]
        macro[key] = float(np.mean([v for v, _ in values])) if values else None
        denom = sum(s for _, s in values)
        weighted[key] = float(sum(v * s for v, s in values) / denom) if denom else None
        eligible[key] = len(values)
    conf = p.max(1)
    correct = pred == y
    reliability, ece = [], 0.0
    for j in range(bins):
        mask = (conf >= j / bins) & (conf < ((j + 1) / bins) if j < bins - 1 else conf <= 1)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        avg = float(conf[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(acc - avg)
        reliability.append(
            {
                "lower": j / bins,
                "upper": (j + 1) / bins,
                "count": n,
                "accuracy": acc,
                "confidence": avg,
            }
        )
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "macro": macro,
        "weighted": weighted,
        "eligible_macro_classes": eligible,
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "brier_multiclass": float(((p - np.eye(7)[y]) ** 2).sum(1).mean()),
        "ece": float(ece),
        "reliability_bins": reliability,
        "mean_confidence": float(conf.mean()),
        "mean_error_confidence": float(conf[~correct].mean()) if (~correct).any() else None,
        "mean_entropy": float(entropy.mean()),
        "absent_classes": [CLASSES[i] for i in range(7) if not (y == i).any()],
        "undefined_policy": "null; aggregate means omit undefined terms; eligible counts disclosed",
    }
