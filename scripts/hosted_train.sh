#!/usr/bin/env bash
set -euo pipefail
# Run in a cloned repository on Colab/Kaggle with a user-provided licensed dataset.
# CPU wheels are intentionally NOT forced here: use the host's compatible PyTorch.
python -m pip install -e '.[dev]'
python -m app.main validate --root "${1:?Supply dataset root}" --output artifacts/dataset
python -m app.main train --manifest artifacts/dataset/manifest.csv --config configs/baseline.json --output artifacts/baseline
# Run this final evaluation only after all validation-based model selection is finished.
python -m app.main evaluate --checkpoint artifacts/baseline/best.pt --manifest artifacts/baseline/manifest.csv --output artifacts/baseline/evaluation
