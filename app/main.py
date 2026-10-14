import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import CLASSES, TrainConfig
from app.evaluation.reports import write_json


def main():
    parser = argparse.ArgumentParser(description="SkinNet+ local research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Decode/hash all images and freeze grouped splits")
    validate.add_argument("--root", default="archive")
    validate.add_argument("--output", default="artifacts/dataset")
    validate.add_argument("--seed", type=int, default=42)
    training = sub.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="auto")
    cam = sub.add_parser("gradcam")
    cam.add_argument("--checkpoint", required=True)
    cam.add_argument("--image", required=True)
    cam.add_argument("--output", required=True)
    cam.add_argument("--allow-smoke", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--output", required=True)
    fairness = sub.add_parser("fairness")
    fairness.add_argument("--predictions", required=True)
    fairness.add_argument(
        "--annotations", required=True, help="CSV image_id,subgroup and optional cluster_id"
    )
    fairness.add_argument(
        "--provenance", required=True, help="JSON verified=true, method, source, task_compatibility"
    )
    fairness.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            from app.data.ham10000 import HAM10000
            from app.data.validation import split_manifest, validate_images
            from app.fairness.audit import representation_audit

            out = Path(args.output)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Validation output must be empty; preserve frozen manifests")
            out.mkdir(parents=True, exist_ok=True)
            try:
                df, report = validate_images(HAM10000(args.root).load())
            except ValueError as exc:
                write_json(out / "validation.json", {"valid": False, "error": str(exc)})
                raise
            df.to_csv(out / "validated.csv", index=False)
            write_json(out / "validation.json", report)
            write_json(out / "representation.json", representation_audit(df))
            if not report["valid"]:
                raise ValueError(
                    "Dataset validation failed. Inspect validation.json; no exclusions or splits created"
                )
            splits = split_manifest(df, args.seed)
            splits.to_csv(out / "manifest.csv", index=False)
            write_json(
                out / "split_info.json",
                {
                    "seed": args.seed,
                    "strategy": "10 stratified group folds; fold0 test, fold1 val",
                    "class_map": CLASSES,
                    "counts": pd.crosstab(splits.split, splits.dx).to_dict(),
                    "grouping": "connected patient/lesion/file/pixel duplicate components",
                },
            )
            print(
                json.dumps(
                    {k: report[k] for k in ("rows", "class_counts", "dimensions", "valid")},
                    indent=2,
                )
            )
        elif args.command == "train":
            from app.training.train import train

            train(args.manifest, TrainConfig.read(args.config), args.output)
        elif args.command == "evaluate":
            from app.evaluation.evaluate import evaluate

            metrics = evaluate(args.checkpoint, args.manifest, args.output, args.device)
            print(
                f"Evaluation saved. Run type={metrics['provenance']['run_type']}; n={metrics['n']}"
            )
        elif args.command == "gradcam":
            from app.inference.predictor import Predictor, validate_upload

            predictor = Predictor(args.checkpoint, args.allow_smoke)
            image = validate_upload(Path(args.image).read_bytes())
            prediction = predictor.predict(image)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            predictor.explain(image).save(out)
            write_json(
                out.with_suffix(".json"),
                {
                    "prediction": prediction,
                    "metadata": predictor.metadata,
                    "limitation": "Exploratory saliency; not proof of clinical reasoning or fairness",
                },
            )
            print(f"Saved {out}")
        elif args.command == "compare":
            rows, cohort = [], None
            for report in args.reports:
                m = json.loads(Path(report).read_text())
                provenance = m["provenance"]
                identity = (provenance["manifest_sha256"], provenance["test_ids_sha256"])
                if cohort is not None and identity != cohort:
                    raise ValueError("Comparison requires the same frozen held-out cohort")
                cohort = identity
                row = {
                    "report": report,
                    "run_type": provenance["run_type"],
                    "strategy": provenance["strategy"],
                    "pretrained": provenance.get("pretrained"),
                    "seed": provenance.get("seed"),
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro"]["f1"],
                    "ece": m["ece"],
                    "fairness": "unavailable: no verified annotations",
                }
                row.update({f"{c}_recall": v["recall"] for c, v in m["per_class"].items()})
                rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"Comparison saved to {args.output}")
        elif args.command == "fairness":
            from app.fairness.metrics import subgroup_metrics

            pred, ann = pd.read_csv(args.predictions), pd.read_csv(args.annotations)
            provenance = json.loads(Path(args.provenance).read_text())
            if not {"image_id", "subgroup"} <= set(ann) or ann.image_id.duplicated().any():
                raise ValueError("Annotations need unique image_id and subgroup columns")
            if not pred.image_id.is_unique or not set(ann.image_id) <= set(pred.image_id):
                raise ValueError("Duplicate predictions or unknown annotation IDs")
            if not provenance.get("verified") or not all(
                provenance.get(k) for k in ("method", "source", "task_compatibility")
            ):
                raise ValueError(
                    "Provenance must confirm verification, method, source and task compatibility"
                )
            merged = pred.merge(ann, on="image_id", how="left", validate="one_to_one")
            clusters = merged.cluster_id if "cluster_id" in merged else merged.get("group_id")
            if clusters is not None and clusters.isna().any():
                raise ValueError("Incomplete bootstrap cluster identifiers")
            report = subgroup_metrics(
                merged.label,
                merged[[f"p_{c}" for c in CLASSES]],
                merged.subgroup,
                provenance=provenance,
                verified=True,
                cluster_ids=clusters,
            )
            write_json(args.output, report)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"SkinNet+ error: {exc}\n")


if __name__ == "__main__":
    main()
