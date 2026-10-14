# Experiment protocol

Research question: does performance vary across verified skin-tone groups, and can an intervention
reduce disparities without hiding losses for individual diseases? HAM10000 cannot answer this alone.

Freeze the validated manifest and seed before model development. Use lesion/patient/duplicate
components for splitting. Reserve test for final evaluation. Select all learning rates, epochs,
thresholds and calibration changes using validation only. Initial comparison: unweighted CE,
inverse-frequency class-weighted CE, and disease-balanced replacement sampling, individually.
These are disease imbalance interventions, not skin-tone interventions. Use identical test hashes
and at least three predetermined seeds (42, 43, 44) for the eventual study. Never tune from test.

Record config, manifest hashes, hardware/software, pretrained source, history and best epoch.
Report per-class precision, recall, F1, support, OVR AUC/AP, confusion, calibration and errors.
ECE uses 15 equal-width confidence bins; multiclass Brier is mean sum of squared probability
errors (range 0–2). Undefined denominators become JSON null; aggregate means omit undefined
terms with the eligible-class count disclosed. Absent classes invalidate complete seven-class claims.
Entropy describes model dispersion, not reliable epistemic or clinical uncertainty.

Future compatible annotated datasets: report disease-by-tone counts, acquisition modality and
annotation provenance; use patient/lesion cluster bootstrap intervals and warn on small groups.
Group recall/FNR and specificity are per disease; any malignant grouping requires an explicit
clinical task definition (AKIEC contains in-situ lesions; never call it a severity score).
Do not pool taxonomies or modalities without defensible mapping. Fine-tuning needs a separate
training cohort and unchanged held-out evaluation. Individual disparities plus uncertainty matter
more than a single fairness score. Record multiple comparisons and confounding limitations.

Completed smoke runs verify engineering only. Full baseline results and fairness hypotheses
remain unevaluated until documented experiments are completed.
