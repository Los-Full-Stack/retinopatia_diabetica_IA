import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config.config import (
    CLASS_NAMES,
    CLASS_WEIGHT_MULTIPLIERS,
    FINE_TUNE_BLOCKS,
    FOCAL_LOSS_ENABLED,
    FOCAL_LOSS_GAMMA,
    ORDINAL_LOSS_ALPHA,
    ORDINAL_LOSS_ENABLED,
    GRADIENT_CLIP_VALUE,
    IMAGE_SIZE,
    MODEL_NAME,
    MODELS_DIR,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    RESULTS_DIR,
    SAMPLER_WEIGHT_POWER,
    SCORE_BALANCED_ACCURACY_WEIGHT,
    SCORE_F1_MACRO_WEIGHT,
    SCORE_KAPPA_WEIGHT,
    SCORE_SEVERE_RECALL_WEIGHT,
    TEST_MODE,
    USE_RAW_SPLITS,
    USE_PROCESSED_IMAGES,
    USE_RETINAL_PREPROCESSING,
    USE_WEIGHTED_SAMPLER,
    USE_EXTERNAL_TRAINING_DATA,
)
from src.metrics import compute_metrics
from src.utils import save_json


class FocalLoss(torch.nn.Module):
    def __init__(self, class_weights=None, gamma: float = FOCAL_LOSS_GAMMA, ordinal_alpha: float = 0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ordinal_alpha = ordinal_alpha
        self.reduction = reduction
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())
        distance = torch.abs(torch.arange(NUM_CLASSES).view(1, -1) - torch.arange(NUM_CLASSES).view(-1, 1)).float()
        self.register_buffer("distance", distance / max(NUM_CLASSES - 1, 1))

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.ordinal_alpha > 0:
            probs = torch.softmax(logits, dim=1)
            ordinal_penalty = (probs * self.distance[targets]).sum(dim=1)
            loss = loss + self.ordinal_alpha * ordinal_penalty
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights=None, alpha: float = ORDINAL_LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())
        distance = torch.abs(torch.arange(NUM_CLASSES).view(1, -1) - torch.arange(NUM_CLASSES).view(-1, 1)).float()
        self.register_buffer("distance", distance / max(NUM_CLASSES - 1, 1))

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)
        ordinal_penalty = (probs * self.distance[targets]).sum(dim=1).mean()
        return ce + self.alpha * ordinal_penalty


def calculate_class_weights(labels, device):
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels))
    multipliers = np.asarray(CLASS_WEIGHT_MULTIPLIERS, dtype=np.float32)
    weights = weights * multipliers
    data = {str(i): float(w) for i, w in enumerate(weights)}
    save_json(data, RESULTS_DIR / "class_weights.json")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_f1, history, best_score=None):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_f1_macro": float(best_val_f1),
        "best_score": float(best_score) if best_score is not None else None,
        "history": history,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "normalization_mean": NORMALIZATION_MEAN,
        "normalization_std": NORMALIZATION_STD,
        "test_mode": TEST_MODE,
        "use_retinal_preprocessing": USE_RETINAL_PREPROCESSING,
        "use_processed_images": USE_PROCESSED_IMAGES,
        "use_raw_splits": USE_RAW_SPLITS,
        "use_external_training_data": USE_EXTERNAL_TRAINING_DATA,
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "sampler_weight_power": SAMPLER_WEIGHT_POWER,
        "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
        "focal_loss_enabled": FOCAL_LOSS_ENABLED,
        "ordinal_loss_enabled": ORDINAL_LOSS_ENABLED,
        "ordinal_loss_alpha": ORDINAL_LOSS_ALPHA,
        "fine_tune_blocks": FINE_TUNE_BLOCKS,
    }


def validation_score(metrics):
    recall = metrics.get("recall_per_class", [0.0] * NUM_CLASSES)
    difficult_recall = (recall[2] + recall[3] + recall[4]) / 3
    return (
        SCORE_F1_MACRO_WEIGHT * metrics.get("f1_macro", 0.0)
        + SCORE_BALANCED_ACCURACY_WEIGHT * metrics.get("balanced_accuracy", 0.0)
        + SCORE_KAPPA_WEIGHT * metrics.get("kappa_quadratic", 0.0)
        + SCORE_SEVERE_RECALL_WEIGHT * difficult_recall
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=False):
    model.train(train)
    losses, y_true, y_pred = [], [], []
    start = time.time()
    for images, labels, _, _ in tqdm(loader, desc="train" if train else "val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        try:
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler and scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                        optimizer.step()
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                print("Memoria CUDA insuficiente. Reduce BATCH_SIZE de 8 a 4.")
            raise
        losses.append(float(loss.detach().cpu()))
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        del images, labels, logits, loss
    metrics = compute_metrics(y_true, y_pred, NUM_CLASSES)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["seconds"] = float(time.time() - start)
    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    checkpoint_name,
    logger,
    resume=False,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_score = -float("inf")
    epochs_without_improvement = 0
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    last_path = MODELS_DIR / "last_checkpoint.pth"
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler.is_enabled() and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = ckpt.get("history", [])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        best_val_f1 = ckpt.get("best_val_f1_macro", best_val_f1)
        best_score = ckpt.get("best_score", best_score) or best_score

    best_path = MODELS_DIR / checkpoint_name
    for epoch in range(1, epochs + 1):
        logger.info("Época %s/%s", epoch, epochs)
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        if scheduler:
            scheduler.step(val_metrics["loss"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        logger.info("Métricas época %s: %s", epoch, row)

        score = validation_score(val_metrics)
        logger.info("Score compuesto validación época %s: %.6f", epoch, score)
        improved = score > best_score
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1_macro"]
            best_score = score
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_f1, history, best_score),
                best_path,
            )
            epochs_without_improvement = 0
            logger.info("Checkpoint guardado: %s", best_path)
        else:
            epochs_without_improvement += 1

        torch.save(
            checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_f1, history, best_score),
            last_path,
        )
        if epochs_without_improvement >= patience:
            logger.info("Early stopping activado.")
            break
    return history, best_path


def save_training_plots(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    steps = list(range(1, len(history) + 1))
    phase_breaks = [
        idx + 1
        for idx in range(1, len(history))
        if history[idx]["epoch"] <= history[idx - 1]["epoch"]
    ]

    def series(split, metric):
        values = []
        for row in history:
            value = row[split].get(metric)
            values.append(np.nan if value is None or not np.isfinite(value) else value)
        return values

    for metric, filename, ylabel in [
        ("loss", "training_loss.png", "Loss"),
        ("accuracy", "training_accuracy.png", "Accuracy"),
        ("balanced_accuracy", "training_balanced_accuracy.png", "Balanced accuracy"),
        ("f1_macro", "training_f1_macro.png", "F1 macro"),
        ("kappa_quadratic", "training_kappa.png", "Kappa cuadrática"),
    ]:
        train_values = series("train", metric)
        val_values = series("val", metric)
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.plot(steps, train_values, label="Entrenamiento", color="#2563eb", linewidth=2.5, marker="o", markersize=4)
        ax.plot(steps, val_values, label="Validacion", color="#ea580c", linewidth=2.5, marker="o", markersize=4)
        finite_val = [(step, value) for step, value in zip(steps, val_values) if np.isfinite(value)]
        if finite_val:
            best_step, best_value = max(finite_val, key=lambda item: item[1] if metric != "loss" else -item[1])
            ax.scatter([best_step], [best_value], s=130, color="#16a34a", zorder=5, label="Mejor validacion")
            ax.annotate(
                f"{best_value:.3f}",
                xy=(best_step, best_value),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="#166534",
                fontweight="bold",
            )
        for phase_break in phase_breaks:
            ax.axvline(phase_break - 0.5, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(
                phase_break - 0.35,
                0.98,
                "fine-tuning",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                color="#475569",
            )
        ax.set_title(ylabel, fontsize=16, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel("Paso de entrenamiento")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.7, len(steps) + 0.3)
        ax.legend(frameon=True, loc="best")
        ax.grid(True, linewidth=0.8, alpha=0.35)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close()


def promote_best_model(source: Path):
    target = MODELS_DIR / "best_retina_model.pth"
    shutil.copy2(source, target)
    return target
