import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.config import TrainConfig
from app.explainability.gradcam import GradCAM
from app.inference.predictor import Predictor, validate_upload
from app.models.checkpoints import load_checkpoint, save_checkpoint
from app.models.classifier import build_model
from app.training.reproducibility import device_for, seed_all
from app.ui.helpers import read_report


@pytest.fixture
def model():
    seed_all(42, 1)
    return build_model(False).eval()


def test_shapes_checkpoint_and_cam(model, tmp_path):
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape == (1, 7)
    path = tmp_path / "checkpoint.pt"
    config = TrainConfig(input_size=64, pretrained=False, smoke=True)
    save_checkpoint(path, model, config, {"epoch": 1, "run_type": "engineering_smoke"})
    loaded, c, meta = load_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    assert c.input_size == 64 and meta["epoch"] == 1
    with pytest.raises(ValueError, match="smoke"):
        Predictor(path)
    predictor = Predictor(path, allow_smoke=True)
    image = Image.new("RGB", (80, 90), color=(150, 70, 50))
    result = predictor.predict(image)
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    cam = GradCAM(model)
    heat = cam.heatmap(x)
    assert heat.shape == (64, 64) and np.isfinite(heat).all()
    assert heat.min() >= 0 and heat.max() <= 1
    assert cam.overlay(image, x).size == (80, 90)
    assert len(model.features[-1]._forward_hooks) == 0
    with pytest.raises(ValueError):
        cam.heatmap(x, target=7)
    with pytest.raises(ValueError):
        cam.heatmap(x.repeat(2, 1, 1, 1))
    assert len(model.features[-1]._forward_hooks) == 0
    # Gradient explanation really depends on gradients and feature activations.
    with torch.no_grad():
        model.classifier[1].weight.zero_()
    assert np.all(cam.heatmap(x) == 0)


def test_checkpoint_errors(model, tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, TrainConfig(), {"epoch": 1})
    value = torch.load(path, weights_only=True)
    value["classes"] = list(reversed(value["classes"]))
    torch.save(value, path)
    with pytest.raises(ValueError, match="class map"):
        load_checkpoint(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_image_validation_and_reports(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (40, 32)).save(buf, format="PNG")
    assert validate_upload(buf.getvalue()).size == (40, 32)
    for raw in (b"", b"bad"):
        with pytest.raises(ValueError):
            validate_upload(raw)
    with pytest.raises(ValueError):
        validate_upload(buf.getvalue(), max_bytes=1)
    small = io.BytesIO()
    Image.new("RGB", (1, 1)).save(small, format="PNG")
    with pytest.raises(ValueError):
        validate_upload(small.getvalue())
    assert read_report(tmp_path / "missing") is None
    (tmp_path / "report").write_text("{bad")
    with pytest.raises(ValueError):
        read_report(tmp_path / "report")
    (tmp_path / "report").write_text("{}")
    assert read_report(tmp_path / "report") == {}


def test_configuration_and_cpu():
    assert device_for("cpu") == "cpu"
    with pytest.raises(ValueError):
        TrainConfig(input_size=1).validate()
    with pytest.raises(ValueError):
        TrainConfig(max_batches=1).validate()


def test_pretrained_failure_is_explicit(monkeypatch):
    import app.models.classifier as classifier

    def failed_download(**kwargs):
        raise OSError("offline fixture")

    monkeypatch.setattr(classifier, "efficientnet_b0", failed_download)
    with pytest.raises(RuntimeError, match="Pretrained weights unavailable"):
        classifier.build_model(True)


def test_partial_checkpoint_cannot_be_used_for_inference(model, tmp_path):
    path = tmp_path / "partial.pt"
    save_checkpoint(path, model, TrainConfig(), {"epoch": 1, "partial_epoch": 2})
    with pytest.raises(ValueError, match="Partial interrupted"):
        load_checkpoint(path)
