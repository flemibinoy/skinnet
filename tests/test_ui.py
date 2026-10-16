import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.config import CLASSES
from app.evaluation.metrics import classification_metrics
from app.ui.helpers import (
    checkpoint_completion_status,
    checkpoint_guidance,
    class_distribution,
    default_workspace,
    discover_runs,
    EvaluationRunTypeMismatch,
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
    assert app.title[0].value == "SkinNet+ research workspace"
    assert "not clinically validated and not a substitute for professional medical evaluation." in text(app)
    assert any(b.label == "Start image inspection" for b in app.button)
    app.radio[0].set_value("Inspect an image").run()
    assert app.title[0].value == "Inspect an image"
    assert "A research model is needed" in text(app)
    assert next(b for b in app.button if b.label == "Analyze image").disabled
    next(b for b in app.button if b.label == "Set up a model").click().run()
    assert app.title[0].value == "Model & experiments"
    assert "No research model has been recorded" in text(app)
    for page in ["Overview", "Dataset & evaluation", "Research limitations", "Model & experiments"]:
        app.radio[0].set_value(page).run()
        assert not app.exception
    app.radio[0].set_value("Dataset & evaluation").run()
    assert "Dataset validation has not been recorded" in text(app)
    assert "This model has not been evaluated here" in text(app)
    app.radio[0].set_value("Research limitations").run()
    assert "No verified fairness experiment" in text(app)
    assert any(s.value == "Current evaluation methodology" for s in app.subheader)
    assert any("cannot establish patient independence" in item.value for item in app.caption)


def test_obsolete_navigation_state_falls_back_to_overview(tmp_path):
    app = AppTest.from_file(ENTRY, default_timeout=30)
    app.session_state["_artifact_root"] = str(tmp_path)
    app.session_state["navigation"] = "Fairness research"
    app.run()
    assert not app.exception
    assert app.title[0].value == "SkinNet+ research workspace"


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
    status_path = Path(workspace["checkpoint"]).parent / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(json.dumps({"status": "completed", "completed_epochs": 1}))
    assert a != inspection_signature(b"first", workspace)


def test_checkpoint_guidance_reports_interrupted_and_missing_runs(tmp_path):
    run = Path(default_workspace(tmp_path)["checkpoint"]).parent
    run.mkdir()
    status = run / "status.json"
    status.write_text(json.dumps({"status": "interrupted", "completed_epochs": 0}))
    message = checkpoint_guidance(run / "best.pt")
    assert "interrupted before any complete epoch" in message
    assert "different output folder" in message

    status.unlink()
    (run / "training.log").write_text("training started")
    assert "may still be running" in checkpoint_guidance(run / "best.pt")
    (run / "best.pt").write_bytes(b"checkpoint")
    assert "Training may still be running" in checkpoint_guidance(run / "best.pt")
    (run / "status.json").write_text("{invalid")
    assert "status file is unreadable" in checkpoint_guidance(run / "best.pt")


def test_interrupted_checkpoint_is_blocked_for_research_and_warned_in_ui(tmp_path):
    run = Path(default_workspace(tmp_path)["checkpoint"]).parent
    run.mkdir()
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    (run / "status.json").write_text(
        json.dumps({"status": "interrupted", "completed_epochs": 1})
    )

    complete, warning = checkpoint_completion_status(checkpoint, checkpoint_epoch=1)
    assert complete is False
    assert "run was interrupted" in warning
    assert "research predictions are disabled" in warning
    (run / "status.json").write_text(
        json.dumps({"status": "completed", "completed_epochs": 1})
    )
    assert checkpoint_completion_status(checkpoint, checkpoint_epoch=1) == (True, None)
    assert checkpoint_completion_status(checkpoint, checkpoint_epoch=2)[0] is False
    (run / "status.json").write_text(
        json.dumps({"status": "interrupted", "completed_epochs": 1})
    )

    app = launch(tmp_path)
    app.radio[0].set_value("Inspect an image").run()
    assert not app.exception
    assert any("run was interrupted" in item.value for item in app.warning)
    assert next(button for button in app.button if button.label == "Analyze image").disabled


def test_research_inference_rejects_interrupted_run_status(monkeypatch, tmp_path):
    import app.ui.pages as pages

    run = Path(default_workspace(tmp_path)["checkpoint"]).parent
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps({"status": "interrupted", "completed_epochs": 1})
    )

    class ResearchPredictor:
        metadata = {"run_type": "research", "epoch": 1}

        def __init__(self, checkpoint, allow_smoke):
            pass

    monkeypatch.setattr(pages, "validate_upload", lambda raw: "decoded-image")
    monkeypatch.setattr(pages, "Predictor", ResearchPredictor)
    with pytest.raises(ValueError, match="run was interrupted"):
        pages.analyze_image(
            b"fixture",
            {
                "checkpoint": str(run / "best.pt"),
                "allow_smoke": False,
                "mode": "Research model",
            },
        )


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
    with pytest.raises(EvaluationRunTypeMismatch, match="selected model purpose"):
        validate_evaluation(metrics, checkpoint, "engineering_smoke")
    with pytest.raises(ValueError, match="Checkpoint is unavailable"):
        validate_evaluation(metrics, tmp_path / "missing.pt")
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


def test_evaluation_run_type_mismatch_is_explained_without_metrics(tmp_path):
    run = Path(default_workspace(tmp_path)["checkpoint"]).parent
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    metrics = classification_metrics(np.arange(7), np.eye(7))
    metrics["provenance"] = {
        "run_type": "engineering_smoke",
        "classes": CLASSES,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (evaluation / "metrics.json").write_text(json.dumps(metrics))

    app = launch(tmp_path)
    app.radio[0].set_value("Dataset & evaluation").run()

    assert not app.exception
    assert any(
        "report is for run type 'engineering_smoke'" in warning.value
        and "selected model purpose is Research model" in warning.value
        and "No metrics are displayed" in warning.value
        for warning in app.warning
    )
    assert not any(metric.label in {"Accuracy", "Macro F1"} for metric in app.metric)
    assert not app.json


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


def test_analysis_requires_compatible_model_and_only_returns_real_gradcam(monkeypatch, tmp_path):
    import app.ui.pages as pages

    run = tmp_path / "smoke"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps({"status": "interrupted", "completed_epochs": 1})
    )

    class FakePredictor:
        metadata = {"run_type": "engineering_smoke"}

        def __init__(self, checkpoint, allow_smoke):
            self.allow_smoke = allow_smoke
            self.prediction_count = 0

        def predict(self, image):
            self.prediction_count += 1
            return {"predicted_class": "nv", "confidence": 0.8}

        def explain(self, image):
            raise RuntimeError("Grad-CAM fixture failure")

    monkeypatch.setattr(pages, "validate_upload", lambda raw: "decoded-image")
    monkeypatch.setattr(pages, "Predictor", FakePredictor)
    workspace = {
        "checkpoint": str(run / "best.pt"),
        "allow_smoke": True,
        "mode": "Engineering smoke test",
    }
    image, prediction, overlay, metadata, explanation_error = pages.analyze_image(
        b"fixture", workspace
    )
    assert image == "decoded-image"
    assert prediction["predicted_class"] == "nv"
    assert overlay is None
    assert metadata["run_type"] == "engineering_smoke"
    assert explanation_error == "Grad-CAM fixture failure"

    with pytest.raises(ValueError, match="not a recorded research model"):
        pages.analyze_image(
            b"fixture",
            {**workspace, "mode": "Research model", "allow_smoke": False},
        )


def test_analysis_prediction_failure_does_not_return_output(monkeypatch):
    import app.ui.pages as pages

    class FailedPredictor:
        metadata = {"run_type": "research"}

        def __init__(self, checkpoint, allow_smoke):
            pass

        def predict(self, image):
            raise ValueError("prediction fixture failure")

    monkeypatch.setattr(pages, "validate_upload", lambda raw: "decoded-image")
    monkeypatch.setattr(pages, "Predictor", FailedPredictor)
    with pytest.raises(ValueError, match="prediction fixture failure"):
        pages.analyze_image(
            b"fixture",
            {
                "checkpoint": "valid.pt",
                "allow_smoke": False,
                "mode": "Research model",
            },
        )


def test_analyze_another_clears_previous_image_state(monkeypatch):
    import app.ui.pages as pages

    state = {
        "upload_generation": 3,
        "analysis": {"result": "old"},
        "analysis_signature": "old-image",
    }
    monkeypatch.setattr(pages.st, "session_state", state)
    pages.reset_inspection()
    assert state == {"upload_generation": 4}
