# SkinNet+

A reproducible, explainable skin-lesion classification research prototype. The central question
is whether performance differs across reliably annotated skin-tone groups and whether controlled
interventions reduce disparities without obscuring disease-specific errors. HAM10000 supports
the initial seven-category dermoscopic classification task; it does not supply trustworthy tone
labels for this study. **Not clinically validated; not a substitute for professional medical evaluation.**

## Current research status

**Not yet evaluated as a research baseline.** CPU smoke runs, when recorded in artifacts, verify
software only. No clinical validity, severity prediction or skin-tone fairness improvement is claimed.
The supplied presentation is conceptual; its results are unverified. Severity is disabled.
DDI and SCIN are deferred. The supplied external checkpoint is zero bytes and unusable.

## Architecture

`app/data` validates files and builds grouped manifests; `models` contains EfficientNet-B0 and
validated checkpoints; `training` provides reproducible optimization; `evaluation` produces metrics,
plots and error tables; `fairness` provides annotation-gated subgroup analysis and disease balancing;
`explainability` provides Grad-CAM; `inference` and `ui` provide local image inspection.
See [architecture](docs/ARCHITECTURE.md), [experiment protocol](docs/EXPERIMENT_PLAN.md),
[dataset guide](docs/DATASET_GUIDE.md), [limitations](docs/RESEARCH_LIMITATIONS.md), and
[implementation checklist](PROJECT_PLAN.md).

## Install on Linux (CPU)

Python 3.10+ supported; this workspace was inspected with Python 3.12.3. The existing `.venv`
was reused. For a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check app tests
```

No API keys or environment variables are required. CPU dependencies need roughly 1–2 GB disk.
Use a GPU runtime's compatible PyTorch installation instead of forcing CPU wheels on Colab/Kaggle.
The exact verified dependency snapshot is `requirements-tested.txt`. See [verified runs](docs/VERIFICATION.md).

## Data and frozen splits

Obtain licensed data manually from the [HAM10000 dataset page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
See the dataset guide before redistribution. In this workspace images already exist under `archive/`
and metadata was recovered from the existing `archive.zip`. No dataset is downloaded by these commands.

```bash
.venv/bin/skinnet validate --root archive --output artifacts/dataset --seed 42
```

Validation decodes all images, reports metadata missingness and dimensions, hashes raw files and
RGB pixels, and identifies duplicate images. Invalid data blocks splitting; no difficult cases
are silently removed. Ten stratified group folds give approximate 80/10/10 train/val/test splits.
Connected components combine available patients, lesions and exact duplicates. Each class must
appear in every split. Patient independence cannot be established without patient IDs. Preserve
the generated manifest and reports; nonempty output directories are protected from overwrite.

## Smoke test and full training

```bash
.venv/bin/skinnet train --manifest artifacts/dataset/manifest.csv --config configs/smoke.json --output artifacts/smoke
.venv/bin/skinnet evaluate --checkpoint artifacts/smoke/best.pt --manifest artifacts/smoke/manifest.csv --output artifacts/smoke/evaluation
```

Smoke mode uses two training and two validation batches, one epoch, 64-pixel inputs and random
initialization. Evaluation still processes the complete held-out split. Smoke outputs are prominently
marked as engineering checks, not research findings. A four-thread laptop CPU with 4 GB RAM can
run these settings. Full training on CPU is possible but slow; a hosted GPU is preferable.

```bash
.venv/bin/skinnet train --manifest artifacts/dataset/manifest.csv --config configs/baseline.json --output artifacts/baseline
# Only after validation-based model selection is finished:
.venv/bin/skinnet evaluate --checkpoint artifacts/baseline/best.pt --manifest artifacts/baseline/manifest.csv --output artifacts/baseline/evaluation
```

The baseline explicitly announces use/download of torchvision ImageNet weights (~21 MB). If it
fails, fix connectivity/cache or explicitly set `pretrained` to false in a copied config; this is
training from scratch and must be recorded as such. CUDA is optional and automatically detected.
Set input size, batch size, epochs, LR, weight decay, seed, patience, strategy and device in JSON.
Early stopping selects maximum validation macro-F1. Test is never used in training or selection.
Ctrl-C saves partial state and retains the last complete checkpoint. Start a new run directory
for reruns; automatic resume is not implemented. Do not treat partial checkpoints as final models.
The manifest contains absolute paths: when moving hosts, validate data there and freeze that
host's manifest before training. A run's evaluation requires its exact frozen manifest and images.

Every run records config, environment, seed, frozen manifest/hash, split counts, log, history,
status, last and best checkpoints. Evaluation adds JSON metrics, per-class CSV, predictions,
errors, calibration/confusion plots and a readable report. These generated artifacts are ignored
by Git. Keep private images, reports and manifests out of public commits.

An optional pretrained CPU check is also available:

```bash
.venv/bin/skinnet train --manifest artifacts/dataset/manifest.csv --config configs/smoke_pretrained.json --output artifacts/smoke-pretrained
```

This explicitly downloads ImageNet weights if absent. It remains a four-example engineering smoke
check, not a research baseline. Both smoke commands were executed successfully in this workspace.
Their output directories already exist; use fresh names when repeating commands.

## Windows PowerShell development demo

From the project root, use the already validated manifest and the bounded development
configuration:

```powershell
.\.venv\Scripts\python.exe -m app.main train --manifest .\artifacts\dataset\manifest.csv --config .\configs\dev.json --output .\artifacts\dev-demo-next
.\.venv\Scripts\python.exe -m app.main evaluate --checkpoint .\artifacts\dev-demo-next\best.pt --manifest .\artifacts\dev-demo-next\manifest.csv --output .\artifacts\dev-demo-next\evaluation --device cpu
.\.venv\Scripts\python.exe -m streamlit run .\app\ui\streamlit_app.py
```

`configs/dev.json` is an engineering smoke run: one epoch, two training batches, two validation
batches, 64-pixel inputs, CPU, and random initialization. The resulting checkpoint and its
held-out evaluation are useful for demonstrating the software pipeline only; they are not a
research baseline or performance claim. Use a fresh, empty output folder for each run; the
`artifacts\dev-demo-next` example leaves existing baseline, dataset, and demo outputs untouched.
Select **Engineering smoke test** in **Model & experiments** and acknowledge its limitations
before using that checkpoint in the UI.

## Explain and inspect

```bash
.venv/bin/skinnet gradcam --checkpoint artifacts/baseline/best.pt --image archive/HAM10000_images_part_1/ISIC_0024306.jpg --output artifacts/explanations/example.png
.venv/bin/python -m streamlit run app/ui/streamlit_app.py
```

For an engineering checkpoint, Grad-CAM requires `--allow-smoke`. In the UI, open **Model &
experiments**, select **Engineering smoke test**, acknowledge that it is a software check, choose
an existing experiment, and select **Use this model**. Engineering outputs always carry a visible
label and are never selected automatically for research inspection.

The default **Inspect an image** page supports upload and preview even without a trained model.
A selected research checkpoint enables analysis, category confidence and a Grad-CAM explanation.
Results persist when changing image views and clear when the input or model changes. Confidence
is not the probability that a patient has cancer, and Grad-CAM is not proof of correctness.
Uploaded JPEG/PNG images stay in local session memory, are not sent externally and are not saved
by default. Saved CLI overlays may be sensitive; keep them private.

**Dataset & evaluation** separates validation, training and held-out evaluation. Class counts,
splits and quality checks come from saved artifacts, never from illustrative data. The evaluation
view checks class mapping and checkpoint identity before presenting results. **Fairness research**
explains current annotation gaps and a study checklist; it displays verified subgroup reports only
in a research-run context. Comparative subgroup scores require at least 20 images, 5 independent
groups and all task classes present. This conservative UI display rule is not a power analysis;
raw reports remain available in Advanced details.

Local experiment selection, architecture/version, training history, configuration, setup commands
and reproducibility information live in **Model & experiments**. Custom artifact paths
and technical diagnostics are confined to its **Advanced details** section. Dataset and fairness
JSON reports are available only in their optional advanced sections. All model paths are
validated with the same research/smoke purpose guards. The interface uses restrained near-white
surfaces, readable text and keyboard-accessible navigation; no decorative predictions or activity
are generated. Dermoscopic task limitations, licence requirements and severity exclusions remain.

## Evaluation and fairness

Metrics include accuracy, macro/weighted precision-recall-F1, per-class support and confusion,
OVR ROC-AUC/AP where both positive and negative cases exist, multiclass Brier and 15-bin ECE,
confidence, entropy and error records. Undefined terms are null, and eligible aggregate counts
are disclosed. See the experiment protocol for definitions and interpretation.

Class weighting and disease-balanced sampling have separate configs. Neither establishes skin-tone
fairness. Once compatible verified subgroup annotations exist, prepare CSV columns `image_id,subgroup`
and optional `cluster_id`; provenance JSON must contain `verified: true`, `method`, `source`, and
`task_compatibility`. Verification is a researcher responsibility, not inferred by software.

```bash
.venv/bin/skinnet fairness --predictions artifacts/baseline/evaluation/predictions.csv --annotations /path/to/verified_annotations.csv --provenance /path/to/provenance.json --output artifacts/baseline/evaluation/fairness.json
.venv/bin/skinnet compare artifacts/baseline/evaluation/metrics.json artifacts/weighted/evaluation/metrics.json --output artifacts/comparison.csv
```

Subgroup reports include missingness, sample/cluster counts, macro-F1, per-class recall/FNR,
specificity, confusion and seeded bootstrap intervals. Group estimates are exploratory for small
cohorts. Comparison refuses different frozen test cohorts and includes per-class recall. Full
research comparisons should repeat predetermined seeds and quantify variability; code does not
claim fairness based on one number. Synthetic unit fixtures are never experimental evidence.

## Portable GPU workflow

Clone the source into a hosted Linux runtime, provide legitimately accessed HAM10000 files,
enable GPU if available, and run `bash scripts/hosted_train.sh /path/to/ham10000`.
Copy the experiment directory back privately, including checkpoints, config, manifest and reports.
Evaluation should be performed on the training host's frozen manifest. Inspecting checkpoints
locally does not require the original images. No DDI/SCIN access or paid services are required.

## Citations, rights and roadmap

Tschandl P, Rosendahl C, Kittler H. *The HAM10000 dataset, a large collection of multi-source
dermatoscopic images of common pigmented skin lesions*. Scientific Data 5, 180161 (2018).
[Paper](https://doi.org/10.1038/sdata.2018.161), [dataset record](https://doi.org/10.7910/DVN/DBW86T).
EfficientNet: Tan & Le (2019), [paper](https://proceedings.mlr.press/v97/tan19a.html).
Grad-CAM: Selvaraju et al. (2017), [paper](https://arxiv.org/abs/1610.02391).
[torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).

No dataset rights are granted by this code. Source-specific image licences and research agreements
must be verified before use/redistribution; article open access is insufficient. No software
licence has been selected on your behalf. DDI requires checking its official access process and
agreement. SCIN uses its own Data Use License. Future adapter work requires licence review,
annotation provenance, taxonomy mapping, modality analysis and patient-aware evaluation before
fine-tuning or fairness comparisons. Severity remains deferred until defensible annotated endpoints exist.
