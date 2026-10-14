# Architecture

The `app` package exposes one CLI and a local Streamlit client. Adapters return a common
image manifest; training and evaluation depend on this manifest rather than dataset names.
HAM10000 is the sole active adapter. Models use torchvision EfficientNet-B0 with a seven-output
head. All preprocessing is stored in the checkpoint configuration. No network API is involved.

Validation decodes each image and hashes file bytes and decoded RGB pixels. Connected components
join patient IDs (if supplied), lesion IDs, and identical images before stratified group folds.
This prevents exact duplicates and known related cases crossing partitions. Near duplicates and
unrecorded patients remain a limitation. Ten folds produce approximately 80/10/10 partitions;
actual class counts are recorded and all classes must be represented in each partition.

Training uses only train and validation. Best validation macro-F1 selects a checkpoint; test
is a separate command. Atomic checkpoints retain optimizer state for interruption inspection.
Experiment directories contain frozen manifests, hashes, configuration, versions and history.
Loading uses `weights_only=True`, validates schema/class mapping and strict state dimensions.
Only locally trusted model artifacts should be used.

Grad-CAM differentiates a target logit through the final feature block, spatially averages its
gradients, weights feature maps, applies ReLU and resizes to the original image. Hooks are removed.
Fairness accepts explicit verified subgroup provenance and never infers tone from pixels.
Severity requires a new labelled task and separate validated head; it is deliberately disabled.

## UI presentation boundary

`app/ui/streamlit_app.py` owns navigation and the session workspace. `pages.py` renders the
four focused pages; `design.py` owns restrained visual styles. `helpers.py` adapts saved artifact
schemas into readable tables, checks evaluation/checkpoint identity and validates research versus
engineering purpose. All model prediction, upload validation and saliency remain in existing
inference/explainability modules; training, splits and metric computation are unchanged.

The landing page supports preview without a model. Model selection is explicit, and engineering
runs require an acknowledgement. Metadata is checked again at inference so file placement or
a misleading config cannot bypass the purpose guard. Session results are keyed by image hash,
checkpoint identity/mtime, purpose and opt-in state. They are cleared when these change. Prediction
can still be shown if Grad-CAM generation fails; the UI does not substitute a fabricated overlay.

Evaluation checks seven-class taxonomy, probability-metric ranges, matrix counts and checkpoint
hash. Subgroup views require verified provenance and a research evaluation context. Comparative
subgroup scores require 20 images, 5 independent groups and all classes present; this is an explicit
UI display policy, not a scientific power calculation. Researchers remain responsible for verifying
annotation quality, cohort linkage, task compatibility and adequate disease-by-group sample sizes.
Raw reports, artifact locations and technical errors are optional advanced content, not the primary
inspection flow. New dataset adapters and research annotation schemas remain separate work.
