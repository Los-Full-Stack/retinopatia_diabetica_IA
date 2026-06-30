import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import BATCH_SIZE, CLASS_NAMES, MODEL_NAME, NUM_CLASSES, NUM_WORKERS, RESULTS_DIR, SPLITS_DIR
from src.dataset import RetinaDataset
from src.model import build_model
from src.preprocessing import get_eval_transforms
from src.utils import persistent_workers_enabled, save_json, set_seed


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        freeze_backbone=False,
        pretrained=False,
        model_name=checkpoint.get("model_name", MODEL_NAME),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_model(model, loader, device: torch.device) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for images, _, _, _ in tqdm(loader, desc="ensemble predict", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))
            logits = (logits + logits_flip) / 2
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_abs_grade_error": float(np.abs(y_true - y_pred).mean()),
        "large_error_rate_grade_distance_2_or_more": float((np.abs(y_true - y_pred) >= 2).mean()),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "predicted_distribution": np.bincount(y_pred, minlength=NUM_CLASSES).tolist(),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    def binary(threshold: int) -> dict:
        true_pos = y_true >= threshold
        pred_pos = y_pred >= threshold
        tp = int((true_pos & pred_pos).sum())
        tn = int((~true_pos & ~pred_pos).sum())
        fp = int((~true_pos & pred_pos).sum())
        fn = int((true_pos & ~pred_pos).sum())
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
        }

    return {
        "referable_dr_grade_2_or_more": binary(2),
        "severe_dr_grade_3_or_more": binary(3),
    }


def score(metrics: dict) -> float:
    recall = metrics["recall_per_class"]
    difficult_recall = (recall[2] + recall[3] + recall[4]) / 3
    return (
        0.25 * metrics["f1_macro"]
        + 0.20 * metrics["balanced_accuracy"]
        + 0.25 * metrics["kappa_quadratic"]
        + 0.25 * difficult_recall
        - 0.20 * metrics["large_error_rate_grade_distance_2_or_more"]
    )


def save_predictions(df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, path: Path) -> None:
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        out = {
            "id_code": row["id_code"],
            "image_path": row["image_path"],
            "true_class_id": int(row["diagnosis"]),
            "true_class_name": CLASS_NAMES[int(row["diagnosis"])],
            "predicted_class_id": int(preds[idx]),
            "predicted_class_name": CLASS_NAMES[int(preds[idx])],
            "confidence": float(probs[idx].max()),
        }
        for cls in range(NUM_CLASSES):
            out[f"probability_class_{cls}"] = float(probs[idx, cls])
        rows.append(out)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua ensemble de checkpoints RetinaAI.")
    parser.add_argument("--name", default="ensemble")
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=SPLITS_DIR / "test_processed_300_split.csv")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    output_dir = RESULTS_DIR / "ensembles" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.weights is not None and len(args.weights) != len(args.models):
        raise ValueError("--weights debe tener la misma cantidad que --models")
    weights = np.asarray(args.weights if args.weights is not None else [1.0] * len(args.models), dtype=np.float64)
    weights = weights / weights.sum()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(args.test_csv)
    y_true = test_df["diagnosis"].astype(int).to_numpy()
    ds = RetinaDataset(test_df, transform=get_eval_transforms())
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )

    all_probs = []
    for path in args.models:
        model = load_model(path, device)
        all_probs.append(predict_model(model, loader, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = np.stack(all_probs, axis=0)
    ensemble_probs = np.tensordot(weights, stacked, axes=(0, 0))
    preds = ensemble_probs.argmax(axis=1)
    metrics = metrics_dict(y_true, preds, ensemble_probs)
    report = {
        "name": args.name,
        "models": [str(path) for path in args.models],
        "weights": weights.tolist(),
        "metrics": metrics,
        "clinical": clinical_metrics(y_true, preds),
        "score": score(metrics),
    }
    save_json(report, output_dir / "ensemble_report.json")
    save_predictions(test_df, ensemble_probs, preds, output_dir / "ensemble_predictions.csv")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
