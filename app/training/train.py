import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from app.training.sampling import CompleteBatchSampler

from app.config import CLASSES
from app.data.transforms import ManifestDataset
from app.data.validation import check_splits, verify_files
from app.evaluation.metrics import classification_metrics
from app.evaluation.reports import write_json
from app.fairness.interventions import disease_strategy
from app.models.checkpoints import save_checkpoint
from app.models.classifier import build_model
from app.training.reproducibility import device_for, environment, seed_all


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    total, n, labels, probs = 0.0, 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (images, targets) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            if optimizer:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(targets)
            n += len(targets)
            labels.extend(targets.cpu().tolist())
            probs.extend(logits.detach().softmax(1).cpu().tolist())
    if not n:
        raise ValueError("Empty training/evaluation loader")
    return total / n, np.array(labels), np.array(probs)


def train(manifest, config, output):
    config.validate()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output directory is nonempty; use a new run directory to preserve experiments"
        )
    df = pd.read_csv(manifest)
    check_splits(df)
    # Deliberately verify/load only development images. Test images remain unused.
    verify_files(df[df.split.ne("test")])
    out.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed, config.threads)
    device = device_for(config.device)
    raw = Path(manifest).read_bytes()
    (out / "manifest.csv").write_bytes(raw)
    write_json(out / "config.json", config.dict())
    write_json(out / "environment.json", environment())
    source_files = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    source_digest = hashlib.sha256()
    for source in source_files:
        source_digest.update(str(source.relative_to(Path(__file__).resolve().parents[1])).encode())
        source_digest.update(source.read_bytes())
    write_json(out / "source.json", {"app_source_sha256": source_digest.hexdigest()})
    (out / "limitations.md").write_text(
        "# Run limitations\n\n"
        "This is a seven-class dermoscopic research task, not clinical diagnosis or severity. "
        "HAM10000 lacks verified skin-tone labels and usually lacks patient identifiers. "
        "Lesion/exact-duplicate grouping does not establish full patient independence. "
        "Smoke runs are engineering checks only. Confidence and saliency are exploratory. "
        "Review docs/RESEARCH_LIMITATIONS.md and the dataset licence before interpretation or sharing.\n"
    )

    write_json(out / "split_counts.json", pd.crosstab(df.split, df.label).to_dict())
    metadata = {
        "version": "0.1.0",
        "classes": CLASSES,
        "seed": config.seed,
        "device": device,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "run_type": "engineering_smoke" if config.smoke else "research",
        "pretrained": config.pretrained,
        "initial_weights": "torchvision.ImageNet1K_V1"
        if config.pretrained
        else "random_initialization",
        "strategy": config.strategy,
    }
    logger = logging.getLogger(str(out))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(out / "training.log")
    logger.addHandler(handler)
    model = build_model(config.pretrained).to(device)
    train_df, val_df = df[df.split.eq("train")], df[df.split.eq("val")]
    weights, sampler = disease_strategy(train_df.label.to_numpy(), config.strategy, config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    if sampler is None:
        sampler = RandomSampler(train_df, generator=generator)
    batch_sampler = CompleteBatchSampler(sampler, config.batch_size, drop_last=False)
    train_loader = DataLoader(
        ManifestDataset(train_df, config.input_size, True),
        batch_sampler=batch_sampler,
        num_workers=config.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        ManifestDataset(val_df, config.input_size),
        batch_size=config.batch_size,
        num_workers=config.workers,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best, stale = [], -1.0, 0
    epoch = 0
    status = "completed"
    try:
        for epoch in range(1, config.epochs + 1):
            loss, _, _ = run_epoch(
                model, train_loader, criterion, device, optimizer, config.max_batches
            )
            val_loss, y, p = run_epoch(
                model,
                val_loader,
                torch.nn.CrossEntropyLoss(),
                device,
                max_batches=config.max_batches,
            )
            metrics = classification_metrics(y, p)
            score = metrics["macro"]["f1"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": score,
                }
            )
            metadata.update(epoch=epoch, validation_macro_f1=score)
            save_checkpoint(out / "last.pt", model, config, metadata, optimizer)
            if score > best:
                best, stale = score, 0
                save_checkpoint(out / "best.pt", model, config, metadata, optimizer)
            else:
                stale += 1
            write_json(out / "history.json", history)
            msg = f"Epoch {epoch}: train_loss={loss:.4f} val_loss={val_loss:.4f} val_macro_f1={score:.4f}"
            print(msg, flush=True)
            logger.info(msg)
            if stale >= config.patience:
                status = "early_stopped"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        # last.pt is the last complete epoch; partial optimizer state is separate.
        save_checkpoint(
            out / "interrupted.pt", model, config, {**metadata, "partial_epoch": epoch}, optimizer
        )
        print("Interrupted; last complete checkpoint and partial state preserved.", flush=True)
    finally:
        write_json(
            out / "status.json",
            {
                "status": status,
                "completed_epochs": len(history),
                "best_validation_macro_f1": best if best >= 0 else None,
                "test_evaluated": False,
                "fairness": "unavailable: verified tone annotations absent",
            },
        )
        logger.removeHandler(handler)
        handler.close()
    return out
