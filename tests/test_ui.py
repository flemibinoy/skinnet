import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.config import CLASSES
from app.evaluation.metrics import classification_metrics
from app.ui.helpers import (
    class_distribution,
    default_workspace,
    discover_runs,
    fairness_verified,
    inspection_signature,
    split_distribution,
    subgroup_rows,
    validate_evaluation,
    validate_model_purpose,
)

ENTRY = Path(__file__).resolve().parents[1] / "app/ui/streamlit_app.py"


def launch(root):
    app = AppTest.from_file(ENTRY, default_timeout=30)
    app.session_state["_artifact_root"] = str(root)
    return app.run()


def text(app):
    return "\n".join(item.value for item in app.markdown)


def test_missing_artifacts_and_navigation(tmp_path):
    app = launch(tmp_path)
    assert not app.exception
    assert app.title[0].value == "Inspect an image"
    assert "A research model is needed" in text(app)
    assert next(b for b in app.button if b.label == "Analyze image").disabled
    next(b for b in app.button if b.label == "Set up a model").click().run()
    assert app.title[0].value == "Model & experiments"
    assert "No research model has been recorded" in text(app)
    for page in ["Dataset & evaluation", "Fairness research", "Model & experiments"]:
        app.radio[0].set_value(page).run()
        assert not app.exception
    app.radio[0].set_value("Dataset & evaluation").run()
    assert "Dataset validation has not been recorded" in text(app)
    assert "This model has not been evaluated here" in text(app)
    app.radio[0].set_value("Fairness research").run()
    assert "No verified fairness experiment" in text(app)
    assert any(s.value == "Checklist for a defensible study" for s in app.subheader)


def test_purpose_guards_and_signature(tmp_path):
    for meta, mode, ack in [
        ({"run_type": "engineering_smoke"}, "Research model", True),
        ({"run_type": "research"}, "Engineering smoke test", True),
        ({"run_type": "engineering_smoke"}, "Engineering smoke test", False),
        ({}, "Research model", False),
    ]:
        with pytest.raises(ValueError):
            validate_model_purpose(meta, mode, ack)
    validate_model_purpose({"run_type": "research"}, "Research model")
    validate_model_purpose({"run_type": "engineering_smoke"}, "Engineering smoke test", True)
    workspace = default_workspace(tmp_path)
    a = inspection_signature(b"first", workspace)
    assert a != inspection_signature(b"second", workspace)
    assert a != inspection_signature(b"first", {**workspace, "allow_smoke": True})


def test_distribution_adapters_and_discovery(tmp_path):
    frame = class_distribution({"class_counts": {"nv": 5, "df": 1}})
    assert frame.Images.sum() == 6 and set(frame.Category) == {"NV", "DF"}
    with pytest.raises(ValueError):
        class_distribution({"class_counts": {"madeup": 1}})
    with pytest.raises(ValueError):
        class_distribution({"class_counts": {"nv": -1}})
    splits = split_distribution({"counts": {"nv": {"train": 4, "val": 1, "test": 1}}})
    assert splits.Images.tolist() == [4, 1, 1]
    with pytest.raises(ValueError):
        split_distribution({"counts": {"nv": {"train": 1}}})
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke/config.json").write_text(json.dumps({"smoke": True}))
    assert discover_runs(tmp_path)[0]["smoke"]


def test_evaluation_identity_and_fairness_support(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    metrics = classification_metrics(np.arange(7), np.eye(7))
    metrics["provenance"] = {
        "run_type": "research",
        "classes": CLASSES,
        "checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    assert validate_evaluation(metrics, checkpoint)["n"] == 7
    with pytest.raises(ValueError):
        validate_evaluation({**metrics, "n": 8}, checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different checkpoint"):
        validate_evaluation(metrics, checkpoint)
    report = {
        "status": "measured",
        "provenance": {
            "verified": True,
            "method": "unit fixture",
            "source": "synthetic test only",
            "task_compatibility": "7-class fixture",
        },
        "groups": {
            "small": {
                "n": 2,
                "independent_clusters": 2,
                "accuracy": 1,
                "macro": {"f1": 1},
                "absent_classes": [],
            },
            "supported": {
                "n": 70,
                "independent_clusters": 30,
                "accuracy": 0.5,
                "macro": {"f1": 0.4},
                "absent_classes": [],
            },
        },
    }
    assert fairness_verified(report)
    assert not fairness_verified({**report, "provenance": "unverified string"})
    rows = subgroup_rows(report)
    assert rows.iloc[0].Accuracy != rows.iloc[0].Accuracy  # pandas NaN, not a fabricated zero
    assert rows.iloc[1].Accuracy == 0.5


def test_dataset_validation_does_not_imply_evaluation(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "validation.json").write_text(
        json.dumps(
            {
                "valid": True,
                "rows": 7,
                "class_counts": {c: 1 for c in CLASSES},
                "issues": [],
                "exact_duplicate_images": [],
                "dimensions": {"(600, 450)": 7},
            }
        )
    )
    app = launch(tmp_path)
    app.radio[0].set_value("Dataset & evaluation").run()
    assert not app.exception
    assert any(m.label == "Dataset validation" and m.value == "Passed" for m in app.metric)
    assert any(m.label == "Model evaluation" and m.value == "Not recorded" for m in app.metric)
    assert "This model has not been evaluated here" in text(app)


def test_engineering_selection_requires_acknowledgement(tmp_path):
    run = tmp_path / "smoke"
    run.mkdir()
    (run / "best.pt").write_bytes(b"invalid test fixture")
    (run / "config.json").write_text(json.dumps({"smoke": True}))
    app = launch(tmp_path)
    app.radio[0].set_value("Model & experiments").run()
    app.radio[1].set_value("Engineering smoke test").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Use this model").disabled
    app.checkbox[0].check().run()
    assert not next(b for b in app.button if b.label == "Use this model").disabled
    next(b for b in app.button if b.label == "Use this model").click().run()
    assert not app.exception
    assert any("compatible completed checkpoint" in e.value for e in app.error)
    assert app.session_state["workspace"]["mode"] == "Research model"


def test_corrupt_report_is_actionable_not_an_exception(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "validation.json").write_text("{broken")
    app = launch(tmp_path)
    app.radio[0].set_value("Dataset & evaluation").run()
    assert not app.exception
    assert any("could not be read" in e.value for e in app.error)
    assert str(tmp_path) not in "\n".join(e.value for e in app.error)


def test_successful_engineering_selection_stays_labelled(tmp_path, monkeypatch):
    from app.config import TrainConfig
    import app.ui.pages as pages

    run = tmp_path / "smoke"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"smoke": True}))
    (run / "best.pt").write_bytes(b"mocked checkpoint, not experimental evidence")
    monkeypatch.setattr(
        pages,
        "load_checkpoint",
        lambda path: (
            None,
            TrainConfig(smoke=True),
            {"run_type": "engineering_smoke", "epoch": 1, "version": "fixture"},
        ),
    )
    app = launch(tmp_path)
    app.radio[0].set_value("Model & experiments").run()
    app.radio[1].set_value("Engineering smoke test").run()
    app.checkbox[0].check().run()
    next(b for b in app.button if b.label == "Use this model").click().run()
    assert not app.exception
    assert app.session_state["workspace"]["mode"] == "Engineering smoke test"
    assert app.session_state["workspace"]["allow_smoke"] is True
    next(b for b in app.button if b.label == "Go to image inspection").click().run()
    assert app.title[0].value == "Inspect an image"
    assert any("Engineering test mode" in w.value for w in app.warning)
