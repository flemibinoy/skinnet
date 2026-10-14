# SkinNet+ implementation plan

Inspected 2026-10-14: Ubuntu Linux, Python 3.12.3, i3-8130U (4 threads), 3.7 GiB RAM,
no NVIDIA tooling. Git has no commits. Existing PDFs and archives are preserved.
External checkpoint is zero bytes and cannot be used. Existing venv contains pip only.

## Phased completion criteria
- [x] M1 Foundation: package install, architecture and research documents verified.
- [x] M2 Data: validate actual files, hash duplicates, grouped fixed manifests; fixture tests pass.
- [x] M3 Training: EfficientNet, checkpoints, history; tiny CPU training completes.
- [x] M4 Evaluation: metrics, reports, prediction and Grad-CAM; tests and smoke evaluation pass.
- [x] M5 Fairness: provenance-gated subgroup interface, bootstrap uncertainty, audit; tests pass.
- [x] M6 UI: local upload, empty states, reports, model metadata; Streamlit helper tests pass.
- [x] M7 Reproducibility: commands verified, environment snapshot, actual smoke report and comparison.

## Dependencies of work
Without data: all code, synthetic fixture tests, UI empty states, documentation.
Actual HAM10000: validation, grouped manifests, real smoke run, full experiments.
Compute: CPU smoke now; full baseline and interventions preferably hosted GPU.
Trustworthy subgroup data: skin-tone metrics, subgroup interventions, fairness findings deferred.
Severity: disabled until clinically defined, independent ground-truth annotations exist.

Full research training is separate from software completion. Smoke metrics cannot support research claims.

## Verification so far
- Foundation: editable install succeeded, CLI help, compileall, pip check, Ruff lint/format pass.
- M2: recovered metadata and 1,770 missing images from the existing ZIP; repaired one truncated
  image from a decoded archive member, retaining the original privately. Earlier failed audits retained.
  Final validation: 10,015 decodable 600x450 files, seven verified labels, 7,470 lesions/groups,
  two same-lesion exact-duplicate pairs, no missing/ambiguous paths or conflicting duplicate labels.
  Seed 42 grouped split: train 8,011; validation 1,002; test 1,002; all classes represented.
- M3/M4: synthetic fixture EfficientNet CPU optimization, checkpoint reload, held-out evaluation,
  CSV/JSON/plots and gradient explanations passed. These are software tests, not research findings.
- M5: verified-provenance gates, absent annotations, per-group metrics and seeded intervals tested
  on synthetic fixtures only. No skin-tone experiment has been performed.
- M6: Streamlit AppTest verifies all four pages/empty states; local HTTP health endpoint returned ok.
- Latest suite: 26 tests passed; lint and formatting pass. Real-data smoke commands are underway.

## Manual responsibilities
Before publication/distribution, verify downloaded image licence/version and source-specific terms.
For full research findings, run baseline and comparable interventions with sufficient compute and
prespecified seeds. Skin-tone analysis remains blocked by trustworthy compatible annotations.
Severity remains disabled. No acceptance of dataset terms or publication is performed by this build.

## Completed real-data engineering verification
- Random-initialization and ImageNet-initialized CPU smoke training each completed one epoch
  with two batches (four examples); best and last checkpoints saved and reloaded.
- Pretrained download observed from official torchvision URL; no silent dataset download.
- Both checkpoints evaluated against all 1,002 fixed test images. JSON/CSV, per-class metrics,
  error records, calibration/confusion plots and readable reports generated.
- Real Grad-CAM overlay and prediction metadata saved under artifacts/smoke-pretrained/explanations.
- CLI comparison of the two actual engineering reports generated; this is not an intervention study.
- AppTest additionally loaded real dataset/evaluation reports and real model metadata without exceptions.
- Full baseline, intervention research runs, hosted GPU execution, external fairness and severity
  experiments are not completed. Software milestone completion does not imply research completion.

Next command (local UI): `.venv/bin/python -m streamlit run app/ui/streamlit_app.py`.
To inspect the verified engineering model, select `artifacts/smoke-pretrained/best.pt`, enable the
smoke inspection checkbox, and point reports to `artifacts/smoke-pretrained/evaluation`.
For an actual baseline use the full-training command in README with a fresh output directory.

## UI refinement from supplied reference
Glass-style lavender/blue panels, rounded cards, pill navigation, decorative empty-state artwork,
responsive layout and local model settings implemented. Predictions persist when switching image
views and clear when the image, checkpoint or smoke-inspection setting changes. No reference
health/severity percentages or fabricated faces/heatmaps were imported.
Verification: 26-test suite passed, page tests rerun after refactoring, and real browser upload,
prediction, Grad-CAM, view switching and stale-result clearing passed. Desktop (1440px) and
mobile (390px) layouts rendered without horizontal overflow; mobile navigation uses a two-column
grid. Ruff and the live health endpoint pass. The app remains served on localhost:8501.

## Product experience redesign
The four pages now have distinct purposes: inspect, dataset/evaluation evidence, fairness planning,
and model/experiment management. Near-white surfaces replace gradient panels and decorative
statistics. Paths and diagnostics are optional advanced content; the landing page has upload,
preview, Analyze and real results. Research and engineering model selection are guarded separately.
Dataset quality does not imply model evaluation. Missing quality fields are not fabricated zeros.
Fairness reports require verified provenance and sufficient display support; otherwise a checklist
and counts explain what remains needed. No datasets were downloaded and no research training was
performed for this redesign. Existing inference and research pipeline modules were preserved.
Verification includes the complete unit suite, AppTest missing/corrupt artifact and purpose guard
cases, and actual browser checks of upload/inference/Grad-CAM with existing engineering weights,
view persistence, stale-result clearing, report pages and 390-pixel mobile layout.
