import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.config import CLASS_MAP
from app.data.ham10000 import HAM10000
from app.data.validation import (
    check_splits,
    connected_groups,
    split_manifest,
    validate_images,
    verify_files,
)


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for label, (name, index) in enumerate(CLASS_MAP.items()):
        for i in range(20):
            image_id = f"{name}_{i}"
            rng = np.random.default_rng(label * 100 + i)
            Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(
                tmp_path / f"{image_id}.png"
            )
            rows.append({"image_id": image_id, "lesion_id": f"lesion_{image_id}", "dx": name})
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)
    return tmp_path


def test_realistic_parse_validation_and_split(dataset):
    df = HAM10000(dataset).load()
    assert set(df.label) == set(range(7))
    validated, report = validate_images(df)
    assert report["valid"] and report["dimensions"] == {"(40, 32)": 140}
    assert report["skin_tone"]["status"] == "unavailable"
    a, b = split_manifest(validated, 42), split_manifest(validated, 42)
    pd.testing.assert_frame_equal(a, b)
    check_splits(a)
    verify_files(a)
    bad = a.copy()
    bad.loc[0, "sha256"] = bad.loc[bad.split.ne(bad.loc[0, "split"]), "sha256"].iloc[0]
    with pytest.raises(ValueError, match="Leakage"):
        check_splits(bad)


def test_missing_corrupt_and_duplicates(dataset):
    (dataset / "akiec_0.png").unlink()
    (dataset / "akiec_1.png").write_bytes(b"broken")
    (dataset / "other").mkdir()
    (dataset / "other" / "akiec_2.png").write_bytes((dataset / "akiec_2.png").read_bytes())
    _, report = validate_images(HAM10000(dataset).load())
    assert not report["valid"] and len(report["issues"]) == 3


def test_exact_duplicate_grouping(dataset):
    (dataset / "akiec_1.png").write_bytes((dataset / "akiec_0.png").read_bytes())
    df, report = validate_images(HAM10000(dataset).load())
    assert report["exact_duplicate_images"] == [["akiec_0", "akiec_1"]]
    groups = connected_groups(df)
    assert groups[0] == groups[1]
    df["patient_id"] = [f"p{i}" for i in range(len(df))]
    df.loc[2, "patient_id"] = df.loc[1, "patient_id"]
    df.loc[2, "lesion_id"] = df.loc[3, "lesion_id"]
    groups = connected_groups(df)
    assert len(set(groups[:4])) == 1
    (dataset / "bcc_0.png").write_bytes((dataset / "akiec_0.png").read_bytes())
    _, report = validate_images(HAM10000(dataset).load())
    assert not report["valid"] and report["duplicate_label_conflicts"]


@pytest.mark.parametrize(
    "case", ["missing_column", "missing_label", "unsupported", "duplicate", "empty"]
)
def test_malformed_metadata(dataset, case):
    path = dataset / "HAM10000_metadata.csv"
    df = pd.read_csv(path)
    if case == "missing_column":
        df = df.drop(columns="lesion_id")
    if case == "missing_label":
        df.loc[0, "dx"] = None
    if case == "unsupported":
        df.loc[0, "dx"] = "invented"
    if case == "duplicate":
        df.loc[0, "image_id"] = df.loc[1, "image_id"]
    if case == "empty":
        df = df.iloc[:0]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        HAM10000(dataset).load()


def test_changed_image(dataset):
    df, _ = validate_images(HAM10000(dataset).load())
    (dataset / "akiec_0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_files(df)


def test_end_to_end_training_evaluation(dataset, tmp_path):
    from app.config import TrainConfig
    from app.training.train import train
    from app.evaluation.evaluate import evaluate
    from app.models.checkpoints import load_checkpoint
    import json

    df, _ = validate_images(HAM10000(dataset).load())
    manifest = tmp_path / "manifest.csv"
    split_manifest(df).to_csv(manifest, index=False)
    config = TrainConfig(
        input_size=64,
        batch_size=2,
        epochs=1,
        patience=1,
        pretrained=False,
        max_batches=1,
        smoke=True,
        threads=1,
    )
    run = tmp_path / "run"
    train(manifest, config, run)
    assert (run / "best.pt").is_file()
    assert json.loads((run / "status.json").read_text())["completed_epochs"] == 1
    _, _, meta = load_checkpoint(run / "best.pt")
    assert meta["run_type"] == "engineering_smoke"
    metrics = evaluate(run / "best.pt", run / "manifest.csv", run / "evaluation", device="cpu")
    assert metrics["n"] == len(df) // 10
    assert (run / "evaluation" / "errors.csv").is_file()
    assert (run / "evaluation" / "calibration.png").is_file()
    with pytest.raises(ValueError, match="nonempty"):
        train(manifest, config, run)
    altered = tmp_path / "altered.csv"
    altered.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs"):
        evaluate(run / "best.pt", altered, run / "other")
