"""User-facing pages over the existing inference and report APIs."""

from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st

from app.config import CLASSES
from app.inference.predictor import Predictor, validate_upload
from app.models.checkpoints import load_checkpoint
from app.ui.design import empty_state, probability_bars
from app.ui.helpers import (
    CLASS_NAMES,
    class_distribution,
    discover_runs,
    fairness_verified,
    inspection_signature,
    read_report,
    split_distribution,
    subgroup_rows,
    validate_evaluation,
    validate_model_purpose,
)


def navigate(page):
    st.session_state["navigation"] = page


def heading(title, description):
    st.title(title)
    st.caption(description)


def empty(title, description):
    st.markdown(empty_state(title, description), unsafe_allow_html=True)


def model_link(label="View model & experiments", key=None):
    st.button(label, key=key, on_click=navigate, args=("Model & experiments",))


def diagnostic(name, error):
    st.session_state.setdefault("diagnostics", {})[name] = str(error)


def report(path, name):
    try:
        return read_report(path)
    except (ValueError, OSError) as exc:
        diagnostic(name, exc)
        st.error(
            f"The {name} report could not be read. Check its location or regenerate it in Model & experiments."
        )
        return None


def smoke_notice(workspace):
    if workspace["mode"] == "Engineering smoke test":
        st.warning(
            "Engineering test mode. This model checks the software pipeline; its predictions and scores are not research results."
        )


def inspect_page(workspace):
    heading(
        "Inspect an image",
        "Upload a dermoscopic image to inspect lesion-category scores and a visual explanation.",
    )
    smoke_notice(workspace)
    left, right = st.columns([0.95, 1.35], gap="large")
    image = None
    signature = None
    with left:
        st.subheader("Upload an image")
        upload = st.file_uploader(
            "Drag an image here or browse files",
            type=["jpg", "jpeg", "png"],
            help="JPEG or PNG, up to 20 MB. At least 32 pixels per side; up to 20 megapixels.",
        )
        st.caption("JPEG / PNG · up to 20 MB · processed locally, not saved by default")
        if upload:
            try:
                raw = upload.getvalue()
                image = validate_upload(raw)
                signature = inspection_signature(raw, workspace)
                st.image(
                    image,
                    caption=f"Input preview · {image.width} × {image.height} pixels",
                    width="stretch",
                )
            except (ValueError, OSError) as exc:
                diagnostic("Image upload", exc)
                st.error(
                    "This image could not be opened. Choose an intact JPEG or PNG under 20 MB, at least 32 pixels per side and no larger than 20 megapixels."
                )
        if st.session_state.get("analysis_signature") != signature:
            st.session_state.pop("analysis", None)
            st.session_state.pop("analysis_signature", None)
        available = Path(workspace["checkpoint"]).is_file()
        if available:
            st.caption("Selected model: " + workspace["label"])
        analyze = st.button(
            "Analyze image",
            type="primary",
            disabled=image is None or not available,
            width="stretch",
        )
        if analyze:
            try:
                with st.spinner("Analyzing the image…"):
                    predictor = Predictor(workspace["checkpoint"], workspace["allow_smoke"])
                    validate_model_purpose(
                        predictor.metadata, workspace["mode"], workspace["allow_smoke"]
                    )
                    result = predictor.predict(image)
                    overlay = None
                    try:
                        overlay = predictor.explain(image)
                    except (ValueError, RuntimeError, OSError) as exc:
                        diagnostic("Visual explanation", exc)
                    st.session_state["analysis"] = {
                        "result": result,
                        "overlay": overlay,
                        "metadata": predictor.metadata,
                    }
                    st.session_state["analysis_signature"] = signature
            except (ValueError, RuntimeError, OSError) as exc:
                st.session_state.pop("analysis", None)
                diagnostic("Image analysis", exc)
                st.error(
                    "Analysis could not finish. Select a compatible trained model in Model & experiments. Engineering models require explicit test-mode acknowledgement."
                )
                model_link(key="analysis_error_model")
    with right:
        analysis = st.session_state.get("analysis")
        if not available:
            empty(
                "A research model is needed",
                "You can upload and preview an image now. Predictions and visual explanations become available after a research model has been trained and selected.",
            )
            model_link("Set up a model", key="empty_model")
            st.subheader("What you will receive")
            st.markdown(
                "A ranked list of seven lesion categories, model confidence, and a Grad-CAM overlay showing regions that influenced the output."
            )
        elif image is None:
            empty(
                "Ready when you are",
                "Choose an image on the left. Review the preview, then select Analyze image to see category scores and model attention.",
            )
        elif analysis is None:
            empty(
                "Image ready for analysis",
                "Review your preview, then select Analyze image. The model supports the seven HAM10000 lesion categories; other conditions and general skin photographs are outside this task.",
            )
        else:
            result = analysis["result"]
            category = result["predicted_class"]
            st.subheader("Model prediction")
            st.markdown(f"**{CLASS_NAMES.get(category, category)} ({category.upper()})**")
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            st.caption(
                "Confidence is the model’s score for this category. It is not the probability that a patient has cancer and does not determine severity."
            )
            st.subheader("Visual explanation")
            if analysis["overlay"] is not None:
                view = st.radio(
                    "View image", ["Grad-CAM explanation", "Original image"], horizontal=True
                )
                st.image(
                    analysis["overlay"] if view == "Grad-CAM explanation" else image,
                    width="stretch",
                )
                st.caption(
                    "Grad-CAM highlights regions that influenced the predicted category. It is a visual explanation, not proof that the model is correct."
                )
            else:
                st.info(
                    "The prediction is available, but a visual explanation could not be generated. Technical details are in Model & experiments."
                )
            with st.expander("All category scores"):
                st.markdown(probability_bars(result["probabilities"]), unsafe_allow_html=True)
                st.caption(
                    f"Prediction entropy: {result['entropy']:.3f} nats. This describes score dispersion, not clinical certainty."
                )


def dataset_page(workspace):
    heading(
        "Dataset & evaluation",
        "Understand the data, its quality, and the evidence from held-out model evaluation.",
    )
    st.subheader("HAM10000")
    st.markdown(
        "Seven lesion categories from dermoscopic images. [Dataset source](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) · [Original paper](https://doi.org/10.1038/sdata.2018.161)"
    )
    audit = report(Path(workspace["dataset"]) / "validation.json", "dataset validation")
    splits = report(Path(workspace["dataset"]) / "split_info.json", "dataset splits")
    status = report(Path(workspace["checkpoint"]).parent / "status.json", "training status")
    metrics = report(Path(workspace["evaluation"]) / "metrics.json", "evaluation")
    a, b, c = st.columns(3)
    a.metric(
        "Dataset validation",
        "Passed"
        if audit and audit.get("valid") is True
        else "Needs review"
        if audit
        else "Not recorded",
    )
    b.metric(
        "Model training",
        "Engineering check"
        if workspace["mode"] == "Engineering smoke test" and status
        else "Recorded"
        if status and status.get("completed_epochs", 0) > 0
        else "Not recorded",
    )
    c.metric("Model evaluation", "Report available" if metrics else "Not recorded")
    st.caption(
        "These are separate stages. Passed data validation does not establish model accuracy or completed evaluation."
    )
    if audit:
        if isinstance(audit.get("rows"), int) and audit["rows"] >= 0:
            st.markdown(
                f"**{audit['rows']:,} metadata records** inspected in the saved validation report."
            )
        if audit.get("valid") is not True:
            st.warning(
                "Data validation has not passed. Resolve reported image or metadata issues before using this dataset for experiments."
            )
        try:
            distribution = class_distribution(audit)
            st.subheader("Lesion-category distribution")
            chart = (
                alt.Chart(distribution)
                .mark_bar(color="#7155a4", cornerRadiusEnd=3)
                .encode(
                    x=alt.X("Images:Q", title="Images"),
                    y=alt.Y("Category:N", sort=[c.upper() for c in CLASSES], title=None),
                    tooltip=["Category", "Lesion category", "Images"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, width="stretch")
            st.dataframe(distribution, hide_index=True, width="stretch")
            st.caption(
                "Disease-class imbalance can be measured here. This distribution does not describe skin-tone representation."
            )
        except (ValueError, TypeError) as exc:
            diagnostic("Class distribution", exc)
            st.error(
                "The saved class counts are incomplete or incompatible. Regenerate the dataset validation report."
            )
        with st.expander("Data quality checks"):
            checks = [
                {
                    "Check": "Missing or unreadable images / ambiguous filenames",
                    "Findings": len(audit["issues"])
                    if isinstance(audit.get("issues"), list)
                    else "Not recorded",
                },
                {
                    "Check": "Exact-duplicate image groups",
                    "Findings": len(audit["exact_duplicate_images"])
                    if isinstance(audit.get("exact_duplicate_images"), list)
                    else "Not recorded",
                },
                {
                    "Check": "Conflicting labels in duplicate groups",
                    "Findings": len(audit["duplicate_label_conflicts"])
                    if isinstance(audit.get("duplicate_label_conflicts"), list)
                    else "Not recorded",
                },
            ]
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
            st.write(
                "Image dimensions: "
                + ", ".join(
                    f"{size}: {n:,} images" for size, n in audit.get("dimensions", {}).items()
                )
            )
            st.caption(
                "Exact duplicates are grouped during splitting. Missing patient identifiers and undetected near-duplicates can leave residual leakage risk."
            )
    else:
        empty(
            "Dataset validation has not been recorded",
            "Validate your local images and metadata to see verified counts, class distribution, and quality checks. The setup command is available in Model & experiments.",
        )
    st.subheader("Training, validation & test partitions")
    if splits:
        try:
            frame = split_distribution(splits)
            st.dataframe(frame, hide_index=True, width="stretch")
            st.caption(
                f"Split seed: {splits.get('seed', 'Not recorded')}. Related lesions, available patients and exact duplicates are grouped; test images are reserved for final evaluation."
            )
        except ValueError as exc:
            diagnostic("Dataset splits", exc)
            st.error(
                "Split information could not be interpreted. Regenerate and preserve the grouped manifest."
            )
    else:
        empty(
            "No recorded split",
            "Generate a validated grouped manifest before training. Dataset validation alone does not establish which images belong to the held-out test set.",
        )
    st.divider()
    st.subheader("Held-out evaluation")
    if not metrics:
        empty(
            "This model has not been evaluated here",
            "Train and select a model, then run the separate evaluation command against its frozen test manifest. Accuracy, class-specific performance and confusion will appear when that report exists.",
        )
        model_link("View experiment setup", key="eval_setup")
    else:
        try:
            validate_evaluation(metrics, workspace["checkpoint"])
            if (
                metrics["provenance"]["run_type"] == "engineering_smoke"
                and workspace["mode"] != "Engineering smoke test"
            ):
                st.info(
                    "This report belongs to an engineering smoke test. Select that test explicitly in Model & experiments to inspect its software-check results."
                )
            else:
                smoke_notice(workspace)
                st.caption(
                    f"{metrics['n']:,} held-out images · selected experiment: {workspace['label']}"
                )
                x, y, z = st.columns(3)
                x.metric("Accuracy", format_score(metrics.get("accuracy")))
                y.metric("Macro F1", format_score(metrics.get("macro", {}).get("f1")))
                z.metric("Calibration error", format_score(metrics.get("ece")))
                st.caption(
                    "Accuracy: fraction classified correctly. Macro F1: average class-level precision/recall balance. Calibration error: difference between confidence and observed correctness; lower is better."
                )
                matrix = pd.DataFrame(
                    [
                        {"Actual": CLASSES[i].upper(), "Predicted": CLASSES[j].upper(), "Images": v}
                        for i, row in enumerate(metrics["confusion_matrix"])
                        for j, v in enumerate(row)
                    ]
                )
                st.subheader("Confusion matrix")
                st.caption(
                    "Rows are the reference labels; columns are the model predictions. Off-diagonal cells are errors."
                )
                heat = (
                    alt.Chart(matrix)
                    .mark_rect()
                    .encode(
                        x=alt.X("Predicted:N", sort=[c.upper() for c in CLASSES]),
                        y=alt.Y("Actual:N", sort=[c.upper() for c in CLASSES]),
                        color=alt.Color("Images:Q", scale=alt.Scale(scheme="purples")),
                        tooltip=["Actual", "Predicted", "Images"],
                    )
                )
                labels = (
                    alt.Chart(matrix)
                    .mark_text(fontSize=12)
                    .encode(
                        x="Predicted:N",
                        y="Actual:N",
                        text="Images:Q",
                        color=alt.condition(
                            alt.datum.Images
                            > max(sum(row) for row in metrics["confusion_matrix"]) / 2,
                            alt.value("white"),
                            alt.value("#272238"),
                        ),
                    )
                )
                st.altair_chart((heat + labels).properties(height=310), width="stretch")
                per = pd.DataFrame(metrics["per_class"]).T.rename(
                    columns={
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "support": "Test images",
                    }
                )
                per.index = per.index.str.upper()
                st.dataframe(per[["Precision", "Recall", "F1", "Test images"]], width="stretch")
                st.caption(
                    "Recall measures how many reference cases were recovered. Undefined metrics remain unavailable rather than being replaced with zero. Low-support categories need careful interpretation."
                )
        except (ValueError, TypeError, KeyError) as exc:
            diagnostic("Evaluation compatibility", exc)
            st.error(
                "This evaluation could not be verified for the selected model. Check that the report and checkpoint belong to the same experiment in Model & experiments."
            )
    with st.expander("Advanced report details"):
        for title, value in [("Validation", audit), ("Splits", splits), ("Evaluation", metrics)]:
            if value:
                st.caption(title)
                st.json(value, expanded=False)


def format_score(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "Not defined"


def fairness_page(workspace):
    heading(
        "Fairness research",
        "Separate measurable disease imbalance from skin-tone questions that require trustworthy annotations.",
    )
    st.markdown(
        '<div class="notice">HAM10000 does not provide reliable skin-tone annotations sufficient to establish skin-tone fairness.</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Supported now")
        st.markdown(
            "- Audit disease-class representation and image quality.\n- Compare disease-class weighting and balanced sampling.\n- Measure overall and per-category errors on the same held-out cohort."
        )
    with b:
        st.subheader("Requires additional evidence")
        st.markdown(
            "- Skin-tone representation and subgroup performance.\n- Disease-by-tone recall and false-negative comparisons.\n- Claims that an intervention reduces skin-tone disparities."
        )
    st.caption(
        "Disease-class balancing is not skin-tone balancing. Skin tone is never inferred from image brightness, RGB values or filenames."
    )
    st.subheader("Completed fairness experiments")
    fairness = report(Path(workspace["evaluation"]) / "fairness.json", "fairness")
    metrics = report(Path(workspace["evaluation"]) / "metrics.json", "evaluation")
    if workspace["mode"] == "Engineering smoke test":
        st.info(
            "Engineering smoke runs verify the software. They are not completed fairness experiments."
        )
    elif (
        fairness_verified(fairness)
        and metrics
        and metrics.get("provenance", {}).get("run_type") == "research"
    ):
        try:
            validate_evaluation(metrics, workspace["checkpoint"])
            rows = subgroup_rows(fairness)
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "Comparative scores are shown only for groups with at least 20 images, 5 independent groups, and all task classes present. This display rule is not a power analysis or evidence of fairness. Smaller cohorts retain counts and diagnostics in Advanced details."
            )
            st.write("Annotation method: " + str(fairness["provenance"]["method"]))
            st.write("Annotation source: " + str(fairness["provenance"]["source"]))
            if not rows["Accuracy"].notna().any():
                st.info(
                    "The recorded subgroups do not have enough support for this comparison view. Collect more compatible, independently grouped cases before interpreting disparities."
                )
            else:
                for name, group in fairness["groups"].items():
                    if rows.loc[rows.Group.eq(str(name)), "Accuracy"].notna().any():
                        with st.expander(f"{name}: per-category performance and uncertainty"):
                            st.dataframe(
                                pd.DataFrame(group.get("per_class", {})).T, width="stretch"
                            )
                            st.write("95% bootstrap intervals", group.get("bootstrap_95", {}))
                            st.write(
                                "Per-category recall intervals",
                                group.get("recall_bootstrap_95", {}),
                            )
                st.caption(
                    "Bootstrap intervals and small sample sizes limit interpretation. Compatible labels and evaluation design must be verified; one aggregate score cannot establish fairness."
                )
        except (ValueError, TypeError, KeyError) as exc:
            diagnostic("Fairness compatibility", exc)
            st.error(
                "The subgroup report could not be verified. Review annotation provenance and cohort compatibility in Advanced details."
            )
    else:
        empty(
            "No verified fairness experiment is recorded for this model",
            "The evaluation infrastructure is implemented. It is not evidence of skin-tone fairness, and no compatible measured subgroup result is available here.",
        )
        if fairness:
            st.caption(
                "A report exists, but its provenance, research-run context or subgroup annotations are insufficient for this view."
            )
    st.subheader("Checklist for a defensible study")
    st.markdown(
        "1. Obtain permitted data with documented skin-tone annotation methods and provenance.\n2. Verify diagnosis mappings, image modality, annotation reliability and missingness.\n3. Record patient or lesion identifiers and freeze a compatible held-out cohort.\n4. Check disease-by-tone sample counts and plan uncertainty/power analysis.\n5. Compare baseline and interventions, reporting each disease and subgroup, including losses."
    )
    st.markdown(
        "**Future data sources:** [DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images) and [SCIN](https://github.com/google-research-datasets/scin) require access/licence review and task compatibility checks. Their integrations remain optional."
    )
    with st.expander("Advanced fairness details"):
        if fairness:
            st.json(fairness, expanded=False)
        else:
            st.caption(
                "No saved subgroup report is present. Annotation provenance and subgroup predictions are required before the fairness evaluator can generate one."
            )


def model_page(workspace, root):
    heading(
        "Model & experiments",
        "Choose a model, understand its training record, and manage reproducible local experiments.",
    )
    st.subheader("Choose a model")
    purpose = st.radio(
        "Model purpose",
        ["Research model", "Engineering smoke test"],
        index=0 if workspace["mode"] == "Research model" else 1,
        horizontal=True,
        key="purpose_selector",
    )
    runs = [r for r in discover_runs(root) if r["smoke"] == (purpose == "Engineering smoke test")]
    acknowledged = False
    if purpose == "Engineering smoke test":
        st.info(
            "Engineering runs use tiny training batches to check the pipeline. Their outputs must not be treated as research predictions or evidence of model accuracy."
        )
        acknowledged = st.checkbox(
            "I understand this is a software check, not a research model",
            value=workspace["allow_smoke"],
            key="smoke_ack",
        )
    if runs:
        index = next(
            (i for i, r in enumerate(runs) if r["checkpoint"] == workspace["checkpoint"]), 0
        )
        chosen = st.selectbox(
            "Available experiment",
            range(len(runs)),
            index=index,
            format_func=lambda i: runs[i]["label"],
            key=f"run_selector_{purpose}",
        )
        checkpoint_available = Path(runs[chosen]["checkpoint"]).is_file()
        st.caption(
            "Saved checkpoint available"
            if checkpoint_available
            else "Training has not produced a saved checkpoint yet."
        )
        if st.button(
            "Use this model",
            type="primary",
            disabled=not checkpoint_available
            or (purpose == "Engineering smoke test" and not acknowledged),
        ):
            selected = runs[chosen]
            try:
                with st.spinner("Checking model compatibility…"):
                    _, _, metadata = load_checkpoint(selected["checkpoint"])
                    validate_model_purpose(metadata, purpose, acknowledged)
                st.session_state["workspace"] = {
                    **workspace,
                    "checkpoint": selected["checkpoint"],
                    "evaluation": str(Path(selected["path"]) / "evaluation"),
                    "mode": purpose,
                    "allow_smoke": acknowledged,
                    "label": selected["label"],
                }
                st.session_state.pop("analysis", None)
                st.session_state.pop("analysis_signature", None)
                st.rerun()
            except (ValueError, OSError) as exc:
                diagnostic("Model selection", exc)
                st.error(
                    "This experiment does not contain a compatible completed checkpoint. Choose another experiment or finish training first."
                )
    else:
        empty(
            "No research model has been recorded"
            if purpose == "Research model"
            else "No engineering run has been recorded",
            "Training happens through the existing local pipeline. Follow the setup instructions below, then return here to select the saved model.",
        )
    st.subheader("Selected model")
    st.write(workspace["label"] + " · " + workspace["mode"])
    metadata = config = None
    if Path(workspace["checkpoint"]).is_file():
        try:
            _, config, metadata = load_checkpoint(workspace["checkpoint"])
            validate_model_purpose(metadata, workspace["mode"], workspace["allow_smoke"])
            a, b, c = st.columns(3)
            a.metric("Architecture", "EfficientNet-B0")
            b.metric("Model version", metadata.get("version", "Not recorded"))
            c.metric("Completed epoch", metadata.get("epoch", "Not recorded"))
            st.caption(
                "Seven-category checkpoint mapping validated. Held-out accuracy is recorded separately on Dataset & evaluation."
            )
            run_dir = Path(workspace["checkpoint"]).parent
            status = report(run_dir / "status.json", "training status")
            st.write(
                "Training record: "
                + str(status.get("status", "Not recorded") if status else "Not recorded")
                .replace("_", " ")
                .capitalize()
            )
            st.write(
                "Evaluation record: "
                + (
                    "Saved report available"
                    if (Path(workspace["evaluation"]) / "metrics.json").is_file()
                    else "Not yet recorded"
                )
            )
            # Training history is a list, so it is intentionally loaded outside read_report.

            if (run_dir / "history.json").is_file():
                try:
                    history = json.loads((run_dir / "history.json").read_text())
                    if history and isinstance(history, list):
                        frame = pd.DataFrame(history)
                        if {"epoch", "train_loss", "val_loss"} <= set(frame):
                            with st.expander("Training history"):
                                st.line_chart(frame.set_index("epoch")[["train_loss", "val_loss"]])
                except (ValueError, OSError) as exc:
                    diagnostic("Training history", exc)
            with st.expander("Experiment configuration"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Setting": [
                                "Input resolution",
                                "Batch size",
                                "Learning rate",
                                "Random seed",
                                "Balancing strategy",
                                "Initial weights",
                            ],
                            "Value": [
                                f"{config.input_size} × {config.input_size}",
                                str(config.batch_size),
                                str(config.learning_rate),
                                str(config.seed),
                                config.strategy.replace("_", " "),
                                "ImageNet" if config.pretrained else "Random initialization",
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        except (ValueError, OSError) as exc:
            diagnostic("Selected model", exc)
            st.error(
                "The selected checkpoint could not be verified. Choose a compatible experiment above or check Advanced details."
            )
    else:
        empty(
            "Training is not complete here",
            "There is no saved checkpoint for the selected research model. You can still inspect dataset quality and plan experiments; image predictions remain disabled.",
        )
    st.button("Go to image inspection", on_click=navigate, args=("Inspect an image",))
    with st.expander(
        "How to prepare a research model", expanded=not Path(workspace["checkpoint"]).is_file()
    ):
        st.markdown(
            "Run these commands from the project directory. Validation checks your existing local dataset; training is a separate action. Use a fresh output directory for each experiment."
        )
        st.code(
            ".venv/bin/skinnet validate --root archive --output artifacts/dataset-new --seed 42\n.venv/bin/skinnet train --manifest artifacts/dataset-new/manifest.csv --config configs/baseline.json --output artifacts/baseline-new",
            language="bash",
        )
        st.caption(
            "If a validated dataset manifest already exists, use it instead of repeating validation. Full training can be slow on a laptop; the README documents CPU and hosted-GPU options."
        )
        st.markdown(
            "After validation-based model selection is finished, evaluate once on the frozen test manifest:"
        )
        st.code(
            ".venv/bin/skinnet evaluate --checkpoint artifacts/baseline-new/best.pt --manifest artifacts/baseline-new/manifest.csv --output artifacts/baseline-new/evaluation",
            language="bash",
        )
    st.subheader("Reproducibility")
    st.write(
        "Each run records configuration, seed, a frozen grouped manifest, source fingerprint, software versions, training history and checkpoint references. Evaluation records its held-out cohort and checkpoint identity."
    )
    st.caption(
        "Severity assessment is disabled because suitable ground-truth severity annotations are absent. Review the README, experiment plan and dataset licence before research use or redistribution."
    )
    with st.expander("Advanced details"):
        st.caption(
            "Local artifact locations and technical diagnostics. Custom checkpoints are subject to the same compatibility and research/smoke separation checks."
        )
        with st.form("custom_artifacts"):
            checkpoint = st.text_input("Checkpoint location", workspace["checkpoint"])
            evaluation = st.text_input("Evaluation directory", workspace["evaluation"])
            dataset = st.text_input("Dataset audit directory", workspace["dataset"])
            if st.form_submit_button("Apply artifact locations"):
                try:
                    if not checkpoint.strip() or not evaluation.strip() or not dataset.strip():
                        raise ValueError("Artifact locations cannot be blank")
                    if Path(checkpoint).is_file():
                        _, _, candidate = load_checkpoint(checkpoint)
                        validate_model_purpose(candidate, purpose, acknowledged)
                    elif purpose == "Engineering smoke test":
                        raise ValueError("Select an existing engineering checkpoint")
                    st.session_state["workspace"] = {
                        **workspace,
                        "checkpoint": checkpoint,
                        "evaluation": evaluation,
                        "dataset": dataset,
                        "mode": purpose,
                        "allow_smoke": acknowledged
                        if purpose == "Engineering smoke test"
                        else False,
                        "label": "Custom local experiment",
                    }
                    st.session_state.pop("analysis", None)
                    st.session_state.pop("analysis_signature", None)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    diagnostic("Custom artifact configuration", exc)
                    st.error(
                        "These locations could not be applied. Check the checkpoint and purpose selection. The previous model remains selected."
                    )
        if metadata:
            st.json({"configuration": config.dict(), "training": metadata}, expanded=False)
        st.json(st.session_state.get("diagnostics", {}), expanded=False)
        env = report(
            Path(workspace["checkpoint"]).parent / "environment.json", "software environment"
        )
        if env:
            st.json(env, expanded=False)
        for filename in ("source.json", "split_counts.json"):
            value = report(Path(workspace["checkpoint"]).parent / filename, filename)
            if value:
                st.json(value, expanded=False)
