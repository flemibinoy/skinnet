import numpy as np
import pytest

from app.evaluation.metrics import classification_metrics
from app.fairness.interventions import disease_strategy
from app.fairness.metrics import subgroup_metrics


def test_perfect_and_undefined():
    y = np.arange(7)
    m = classification_metrics(y, np.eye(7))
    assert m["accuracy"] == 1 and m["macro"]["f1"] == 1
    assert m["brier_multiclass"] == 0 and m["ece"] == 0
    m = classification_metrics([0, 0], np.eye(7)[[0, 0]])
    assert m["per_class"]["bcc"]["recall"] is None
    assert m["per_class"]["akiec"]["roc_auc_ovr"] is None
    assert m["per_class"]["bcc"]["f1"] is None
    assert len(m["absent_classes"]) == 6


def test_known_errors():
    m = classification_metrics([0, 1], np.eye(7)[[0, 0]])
    assert m["accuracy"] == 0.5 and m["ece"] == 0.5 and m["brier_multiclass"] == 1
    assert m["per_class"]["bcc"]["false_negative_rate"] == 1
    assert m["per_class"]["akiec"]["specificity"] == 0
    assert m["per_class"]["akiec"]["precision"] == 0.5


@pytest.mark.parametrize(
    "y,p", [([], []), ([0], [[0] * 7]), ([7], np.eye(7)[[0]]), ([0], [[float("nan")] * 7])]
)
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        classification_metrics(y, p)


def test_fairness_provenance_missing_and_intervals():
    y = np.tile(np.arange(7), 4)
    p = np.eye(7)[y]
    assert subgroup_metrics(y, p)["status"] == "unavailable"
    with pytest.raises(ValueError):
        subgroup_metrics(y, p, ["A"] * len(y))
    groups = ["A"] * 14 + ["B"] * 13 + [None]
    result = subgroup_metrics(
        y, p, groups, verified=True, provenance="Synthetic UNIT TEST ONLY", bootstraps=10
    )
    assert result["missing_annotations"] == 1
    assert result["groups"]["A"]["accuracy"] == 1
    assert result["groups"]["A"]["bootstrap_95"]["accuracy"] == [1, 1]
    again = subgroup_metrics(
        y, p, groups, verified=True, provenance="Synthetic UNIT TEST ONLY", bootstraps=10
    )
    assert result == again
    with pytest.raises(ValueError):
        subgroup_metrics(y, p, ["A"], verified=True, provenance="fixture")


def test_interventions():
    labels = np.array(list(range(7)) + [5] * 10)
    weights, sampler = disease_strategy(labels, "class_weighted", 42)
    assert weights[5] < weights[0] and sampler is None
    _, sampler = disease_strategy(labels, "balanced_sampling", 42)
    assert len(list(sampler)) == len(labels)
    with pytest.raises(ValueError):
        disease_strategy([0, 1], "baseline", 42)


def test_complete_batches_preserve_every_sample():
    from app.training.sampling import CompleteBatchSampler
    from torch.utils.data import SequentialSampler

    batches = list(CompleteBatchSampler(SequentialSampler(range(5)), 2, False))
    assert batches == [[0, 1], [2, 3, 4]]
    assert len(CompleteBatchSampler(SequentialSampler(range(5)), 2, False)) == 2
