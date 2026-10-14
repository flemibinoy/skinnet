# Responsible interpretation

SkinNet+ is an academic research prototype, not clinically validated and not a substitute for
professional medical evaluation. Seven dermoscopic categories are a closed task; arbitrary photos,
unknown diseases and poor quality images may receive confidently wrong predictions. Confidence
is a softmax output, not a probability of clinical correctness. There is no validated rejection
threshold or out-of-distribution detector. Image validation checks file integrity, not clinical suitability.

HAM10000 lacks trustworthy skin-tone annotations. Infrastructure is not evidence of fairness.
Disease class balancing does not establish tone fairness. Confounding, annotation error, selection,
acquisition and distribution shift limit causal interpretations. Small subgroup estimates need
uncertainty and compatible task design. Exact hashes do not detect all near duplicates; absent
patient IDs leave residual leakage risk. Reproducibility across hardware is not bitwise guaranteed.

Grad-CAM is an exploratory saliency map, not proof of clinical reasoning, diagnosis or fairness.
Severity is disabled: it needs a clinically defined endpoint (e.g. independently annotated stage
for a specific disease), qualified annotation, validation and an appropriate cohort. Neither
class nor confidence provides severity. Presentation results are unverified and are not imported.
Uploaded images stay in local memory and are not persisted by default. Offline outputs can
contain sensitive images: keep them private and outside Git. Dataset licences remain distinct
from software and publication licences; verify source-specific rights before release.
