import numpy as np

from app.evaluation.metrics import classification_metrics


def subgroup_metrics(
    labels,
    probabilities,
    groups=None,
    *,
    provenance=None,
    verified=False,
    cluster_ids=None,
    bootstraps=200,
    seed=42,
):
    if groups is None:
        return {"status": "unavailable", "reason": "Verified subgroup annotations are absent"}
    if not verified or not provenance or not str(provenance).strip():
        raise ValueError("Explicit verified annotation provenance required")
    y, p, g = np.asarray(labels), np.asarray(probabilities), np.asarray(groups, dtype=object)
    classification_metrics(y, p)
    if g.shape != y.shape:
        raise ValueError("Group annotations must align with predictions")
    if bootstraps < 0:
        raise ValueError("Bootstrap count cannot be negative")
    missing = np.array([v is None or str(v).strip() in ("", "nan", "<NA>") for v in g])
    clusters = np.arange(len(y)) if cluster_ids is None else np.asarray(cluster_ids)
    if clusters.shape != y.shape:
        raise ValueError("Cluster IDs must align")
    rng = np.random.default_rng(seed)
    result = {}
    for group in sorted(set(str(v) for v in g[~missing])):
        mask = (~missing) & (np.array([str(v) for v in g]) == group)
        indices = np.flatnonzero(mask)
        metrics = classification_metrics(y[mask], p[mask])
        units = np.unique(clusters[mask])
        values = {"accuracy": [], "macro_f1": []}
        recalls = {c: [] for c in metrics["per_class"]}
        for _ in range(bootstraps):
            sampled = rng.choice(units, len(units), replace=True)
            ids = np.concatenate([indices[clusters[indices] == unit] for unit in sampled])
            m = classification_metrics(y[ids], p[ids])
            values["accuracy"].append(m["accuracy"])
            values["macro_f1"].append(m["macro"]["f1"])
            for c in recalls:
                value = m["per_class"][c]["recall"]
                if value is not None:
                    recalls[c].append(value)

        def interval(values):
            return (
                np.quantile(values, [0.025, 0.975]).tolist()
                if len(values) > 1 and len(units) > 1
                else None
            )

        metrics["bootstrap_95"] = {k: interval(v) for k, v in values.items()}
        metrics["recall_bootstrap_95"] = {k: interval(v) for k, v in recalls.items()}
        metrics["independent_clusters"] = len(units)
        metrics["warnings"] = (
            ["Small cohort; estimates are exploratory"]
            if len(indices) < 50 or len(units) < 20
            else []
        )
        result[group] = metrics
    return {
        "status": "measured",
        "provenance": provenance,
        "missing_annotations": int(missing.sum()),
        "groups": result,
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstraps,
        "bootstrap_unit": "cluster" if cluster_ids is not None else "image (independence assumed)",
        "interpretation": "No single aggregate number establishes fairness; compare compatible diseases and uncertainty.",
    }
