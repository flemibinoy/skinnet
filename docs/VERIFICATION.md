# Verified implementation and research status

Observed locally on 2026-10-14; synthetic tests are engineering fixtures only.

| Milestone | Actual verification | Manual next step |
| --- | --- | --- |
| Foundation | CPU PyTorch + editable package installed; CLI, compileall, pip check, Ruff pass | Fresh machines: follow README install |
| Data | All 10,015 files decode at 600x450; full metadata audit, hashes and grouped split saved | Verify exact dataset licence before redistribution/publication |
| Training | Fixture integration plus two actual CPU smoke optimizations; checkpoints reload | Full baseline on suitable CPU/GPU budget |
| Evaluation/explainability | Both smoke models evaluated on all 1,002 test images; JSON/CSV/plots/errors; actual gradient overlay saved | Final research evaluation after validation-only model selection |
| Fairness | Provenance gates, subgroup metrics, missingness and seeded cluster/image bootstrap tested on fixtures | Obtain trustworthy compatible tone annotations |
| UI | All pages/empty states tested; real reports and model metadata loaded; HTTP health response ok | Launch UI and select a trained model |
| Reproducibility | 26 tests pass; dependency snapshot; configs/manifests/hashes/history/source fingerprint; actual comparison table | Repeated full experiments and hosted GPU execution still pending |

## Data repair evidence

Original extraction had 1,770 missing images and ISIC_0032550 was truncated. Metadata and
missing images were restored from the existing local ZIP. The truncated image was replaced
only after its ZIP member decoded successfully; the original was retained outside the dataset.
Failed validation reports are retained privately. Final report has zero issues, no ambiguous or
unreferenced filenames, two exact-duplicate pairs inside known lesion groups and no duplicate
label conflicts. Metadata contains 7,470 lesions; no patient ID or trustworthy skin-tone labels.

Actual class counts: akiec 327, bcc 514, bkl 1099, df 115, mel 1113, nv 6705, vasc 142.
Seed-42 grouped split: train 8,011; validation 1,002; test 1,002. All seven classes represented.

## Completed experiments and actual artifacts

`artifacts/smoke` uses random initialization; `artifacts/smoke-pretrained` uses downloaded official
ImageNet weights. Both use 64-pixel inputs, one epoch, two train and two validation batches.
Only four training images are optimized. They cannot establish disease performance or fairness.
Each evaluated the same 1,002-image fixed test cohort. Result numbers live in actual
`evaluation/metrics.json`, never in invented dashboards. Each has per-class metrics, confusion,
calibration, predictions, errors and report. `artifacts/smoke-comparison.csv` compares those
engineering runs only. No intervention or fairness finding follows from that comparison.
`artifacts/smoke-pretrained/explanations/example.png` is computed from real gradients, paired
with prediction/training metadata. No uploaded images were stored or sent to external services.
Training status.json describes the training stage; completed evaluation is recorded separately
by evaluation/metrics.json and report.md.

## Findings supported by evidence

Local dataset integrity and the recorded class imbalance are established by actual validation.
CPU optimization, checkpoint inference, saliency, evaluation/reporting and UI report loading
work for these runs. These are software and data-integrity findings, not clinical findings.

## Pending research and untested hypotheses

Full HAM10000 baseline and comparable disease-balancing interventions have not been run.
Hosted GPU script syntax is checked, but execution in Colab/Kaggle has not been observed.
Skin-tone disparities and intervention improvements remain untested because trustworthy
compatible annotations are absent. Severity remains disabled. Dataset rights must be verified
against the exact download/version before sharing. No publication, licence acceptance or
external dataset access was performed on your behalf.

## Verified commands

README's editable installation, pytest, Ruff, validation, random and pretrained smoke training,
checkpoint evaluation, Grad-CAM (`--allow-smoke` for smoke artifacts), comparison and Streamlit
launch were executed. Full-training and future annotation commands are provided but not claimed
as executed. Use a new output directory to repeat a protected experiment. No data exclusions
were made to obtain a passing audit.

## Product redesign verification

The redesigned presentation uses four focused pages and keeps inference/training APIs unchanged.
The latest full suite passed **33 tests**, including missing/corrupt reports, working navigation,
research/smoke purpose guards, explicit engineering acknowledgement, checkpoint/report matching,
class/split adapters and insufficient subgroup support. Ruff lint/format and pip dependency checks
passed. Actual browser verification used existing engineering weights only: upload preview,
prediction, Grad-CAM, image-view persistence, clearing results after a new image, real dataset and
evaluation charts, and the fairness checklist all passed. Desktop (1280px) and mobile (390px)
layouts rendered without horizontal overflow. The live HTTP health endpoint returned `ok`.
No datasets were downloaded and no research training was performed for the redesign.

Changed presentation files: app/ui/streamlit_app.py, pages.py, design.py, helpers.py; theme in
.streamlit/config.toml; UI tests in tests/test_ui.py; README, PROJECT_PLAN and architecture/
verification documentation. Model & experiments owns experiment selection and advanced paths.
The default workspace remains a research model; existing smoke weights require explicit opt-in.
Subgroup display support thresholds are documented UI rules, not proof of statistical adequacy.
