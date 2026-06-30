import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE_BLOCKS,
    FINE_TUNE_LEARNING_RATE,
    INITIAL_EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    MODEL_NAME,
    NUM_WORKERS,
    SAMPLER_WEIGHT_POWER,
    SPLITS_DIR,
    USE_PROCESSED_IMAGES,
)
from src.dataset import RetinaDataset
from src.metrics import compute_metrics
from src.model import build_model, unfreeze_last_blocks
from src.preprocessing import get_eval_transforms, get_train_transforms
from src.utils import persistent_workers_enabled, print_cuda_report, save_json, set_seed, setup_logging


STAGE_DIR = MODELS_DIR / "two_stage"
STAGE1_NAME = "stage1_referable_model.pth"
STAGE2_NAME = "stage2_grade234_model.pth"


def make_loader(dataset, device, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )


def make_weighted_sampler(labels):
    counts = labels.value_counts().sort_index()
    sample_weights = labels.map(lambda label: (1.0 / counts[int(label)]) ** SAMPLER_WEIGHT_POWER).astype(float).tolist()
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_weights(labels, num_classes, device):
    counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).astype(float)
    total = counts.sum()
    weights = total / (num_classes * counts.clip(lower=1))
    return torch.tensor(weights.tolist(), dtype=torch.float32, device=device)


def checkpoint_payload(model, epoch, history, num_classes, class_names, stage_name, original_labels):
    return {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "history": history,
        "num_classes": int(num_classes),
        "class_names": class_names,
        "stage_name": stage_name,
        "original_labels": original_labels,
        "model_name": MODEL_NAME,
        "use_processed_images": USE_PROCESSED_IMAGES,
    }


def validation_score(metrics, priority_classes=None):
    score = 0.35 * metrics.get("f1_macro", 0.0) + 0.35 * metrics.get("balanced_accuracy", 0.0)
    score += 0.30 * metrics.get("kappa_quadratic", 0.0)
    if priority_classes:
        recalls = metrics.get("recall_per_class", [])
        priority_recall = sum(recalls[i] for i in priority_classes if i < len(recalls)) / len(priority_classes)
        score = 0.75 * score + 0.25 * priority_recall
    return float(score)


def run_stage_epoch(model, loader, criterion, optimizer, device, num_classes, train=False):
    model.train(train)
    losses = []
    y_true = []
    y_pred = []
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        y_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        del images, labels, logits, loss
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_one_stage(
    train_df,
    val_df,
    num_classes,
    class_names,
    stage_name,
    checkpoint_name,
    device,
    logger,
    initial_epochs,
    fine_tune_epochs,
    priority_classes=None,
):
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = RetinaDataset(train_df, transform=get_train_transforms(retinal_preprocess=not USE_PROCESSED_IMAGES))
    val_ds = RetinaDataset(val_df, transform=get_eval_transforms(retinal_preprocess=not USE_PROCESSED_IMAGES))
    train_loader = make_loader(train_ds, device, sampler=make_weighted_sampler(train_df["diagnosis"]))
    val_loader = make_loader(val_ds, device)

    weights = class_weights(train_df["diagnosis"].tolist(), num_classes, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    logger.info("%s class weights: %s", stage_name, weights.detach().cpu().tolist())

    model = build_model(num_classes=num_classes, freeze_backbone=True, model_name=MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    checkpoint_path = STAGE_DIR / checkpoint_name

    phases = [
        ("initial", initial_epochs),
        ("finetune", fine_tune_epochs),
    ]
    for phase, epochs in phases:
        if phase == "finetune":
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
            unfreeze_last_blocks(model, num_blocks=FINE_TUNE_BLOCKS)
            model.to(device)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FINE_TUNE_LEARNING_RATE, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        for epoch in range(1, epochs + 1):
            logger.info("%s %s epoch %s/%s", stage_name, phase, epoch, epochs)
            train_metrics = run_stage_epoch(model, train_loader, criterion, optimizer, device, num_classes, train=True)
            val_metrics = run_stage_epoch(model, val_loader, criterion, optimizer, device, num_classes, train=False)
            scheduler.step(val_metrics["loss"])
            row = {"stage": stage_name, "phase": phase, "epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)
            score = validation_score(val_metrics, priority_classes=priority_classes)
            logger.info("%s metrics: %s", stage_name, row)
            logger.info("%s validation score: %.6f", stage_name, score)
            if score > best_score:
                best_score = score
                best_epoch = len(history)
                stale = 0
                torch.save(
                    checkpoint_payload(
                        model=model,
                        epoch=best_epoch,
                        history=history,
                        num_classes=num_classes,
                        class_names=class_names,
                        stage_name=stage_name,
                        original_labels=train_df.get("original_diagnosis", train_df["diagnosis"]).drop_duplicates().sort_values().astype(int).tolist(),
                    ),
                    checkpoint_path,
                )
                logger.info("%s checkpoint saved: %s", stage_name, checkpoint_path)
            else:
                stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                logger.info("%s early stopping.", stage_name)
                break

    return checkpoint_path, history


def stage1_frames(train_df, val_df):
    train = train_df.copy()
    val = val_df.copy()
    train["original_diagnosis"] = train["diagnosis"].astype(int)
    val["original_diagnosis"] = val["diagnosis"].astype(int)
    train["diagnosis"] = (train["original_diagnosis"] >= 2).astype(int)
    val["diagnosis"] = (val["original_diagnosis"] >= 2).astype(int)
    return train, val


def stage2_frames(train_df, val_df, include_external):
    train = train_df[train_df["diagnosis"].isin([2, 3, 4])].copy()
    if include_external:
        external_path = SPLITS_DIR / "external_eyepacs_processed_300.csv"
        if external_path.exists():
            external = pd.read_csv(external_path)
            external = external[external["diagnosis"].isin([2, 3, 4])].copy()
            train = pd.concat([train, external], ignore_index=True, sort=False)
    val = val_df[val_df["diagnosis"].isin([2, 3, 4])].copy()
    train["original_diagnosis"] = train["diagnosis"].astype(int)
    val["original_diagnosis"] = val["diagnosis"].astype(int)
    train["diagnosis"] = train["original_diagnosis"] - 2
    val["diagnosis"] = val["original_diagnosis"] - 2
    return train, val


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena modelos auxiliares de dos etapas para RetinaAI.")
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--no-external-stage2", action="store_true", help="No usa EyePACS externo para el refinador 2/3/4.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed()
    device = print_cuda_report(logger)

    train_path = SPLITS_DIR / "train_processed_300_split.csv"
    val_path = SPLITS_DIR / "val_processed_300_split.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    s1_train, s1_val = stage1_frames(train_df, val_df)
    s2_train, s2_val = stage2_frames(train_df, val_df, include_external=not args.no_external_stage2)

    summary = {
        "stage1_train_distribution": s1_train["diagnosis"].value_counts().sort_index().to_dict(),
        "stage1_val_distribution": s1_val["diagnosis"].value_counts().sort_index().to_dict(),
        "stage2_train_distribution": s2_train["diagnosis"].value_counts().sort_index().to_dict(),
        "stage2_val_distribution": s2_val["diagnosis"].value_counts().sort_index().to_dict(),
        "stage2_external_enabled": not args.no_external_stage2,
    }
    save_json(summary, STAGE_DIR / "training_data_summary.json")
    logger.info("Two-stage data summary: %s", summary)

    s1_path, s1_history = train_one_stage(
        s1_train,
        s1_val,
        num_classes=2,
        class_names=["no_referable_0_1", "referable_2_plus"],
        stage_name="stage1_referable",
        checkpoint_name=STAGE1_NAME,
        device=device,
        logger=logger,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        priority_classes=[1],
    )
    s2_path, s2_history = train_one_stage(
        s2_train,
        s2_val,
        num_classes=3,
        class_names=["grade_2", "grade_3", "grade_4"],
        stage_name="stage2_grade234",
        checkpoint_name=STAGE2_NAME,
        device=device,
        logger=logger,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        priority_classes=[0, 1, 2],
    )
    save_json({"stage1_model": str(s1_path), "stage2_model": str(s2_path), "stage1_history": s1_history, "stage2_history": s2_history}, STAGE_DIR / "training_history.json")
    print(f"Modelos de dos etapas guardados en {STAGE_DIR}")


if __name__ == "__main__":
    main()
