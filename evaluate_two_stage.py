import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

from config.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    MODEL_NAME,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    RESULTS_DIR,
    SPLITS_DIR,
    USE_PROCESSED_IMAGES,
)
from src.dataset import RetinaDataset
from src.model import build_model
from src.preprocessing import get_eval_transforms
from src.utils import persistent_workers_enabled, save_json, set_seed


STAGE_DIR = MODELS_DIR / "two_stage"
RESULT_DIR = RESULTS_DIR / "two_stage"


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        freeze_backbone=False,
        pretrained=False,
        model_name=checkpoint.get("model_name", MODEL_NAME),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_probs(model, df, device):
    ds = RetinaDataset(df, transform=get_eval_transforms(retinal_preprocess=not USE_PROCESSED_IMAGES))
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )
    probs = []
    ids = []
    with torch.no_grad():
        for images, _, batch_ids, _ in tqdm(loader, desc="predict", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))
            logits = (logits + logits_flip) / 2
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            ids.extend(batch_ids)
    return np.concatenate(probs, axis=0), ids


def metrics_dict(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_abs_grade_error": float(np.abs(y_true - y_pred).mean()),
        "large_error_rate_grade_distance_2_or_more": float((np.abs(y_true - y_pred) >= 2).mean()),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "predicted_distribution": np.bincount(y_pred, minlength=NUM_CLASSES).tolist(),
    }


def pipeline_predictions(base_probs, stage1_probs, stage2_probs, threshold):
    referable_prob = stage1_probs[:, 1]
    non_ref_preds = np.argmax(base_probs[:, :2], axis=1)
    ref_preds = np.argmax(stage2_probs, axis=1) + 2
    return np.where(referable_prob >= threshold, ref_preds, non_ref_preds).astype(int)


def score_for_threshold(y_true, y_pred):
    metrics = metrics_dict(y_true, y_pred)
    recall = metrics["recall_per_class"]
    difficult_recall = (recall[2] + recall[3] + recall[4]) / 3
    large_error = metrics["large_error_rate_grade_distance_2_or_more"]
    return (
        0.25 * metrics["f1_macro"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.30 * metrics["kappa_quadratic"]
        + 0.25 * difficult_recall
        - 0.20 * large_error
    )


def choose_threshold(y_true, base_probs, stage1_probs, stage2_probs):
    rows = []
    for threshold in np.arange(0.20, 0.86, 0.025):
        preds = pipeline_predictions(base_probs, stage1_probs, stage2_probs, float(threshold))
        metrics = metrics_dict(y_true, preds)
        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score_for_threshold(y_true, preds)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "kappa_quadratic": metrics["kappa_quadratic"],
                "large_error_rate": metrics["large_error_rate_grade_distance_2_or_more"],
                "recall_grade_2": metrics["recall_per_class"][2],
                "recall_grade_3": metrics["recall_per_class"][3],
                "recall_grade_4": metrics["recall_per_class"][4],
            }
        )
    rows = sorted(rows, key=lambda row: row["score"], reverse=True)
    return rows[0], rows


def plot_comparison(base_metrics, two_stage_metrics, path):
    labels = ["Accuracy", "Balanced", "F1 macro", "Kappa", "Recall 2", "Recall 3", "Recall 4"]
    base_values = [
        base_metrics["accuracy"],
        base_metrics["balanced_accuracy"],
        base_metrics["f1_macro"],
        base_metrics["kappa_quadratic"],
        base_metrics["recall_per_class"][2],
        base_metrics["recall_per_class"][3],
        base_metrics["recall_per_class"][4],
    ]
    new_values = [
        two_stage_metrics["accuracy"],
        two_stage_metrics["balanced_accuracy"],
        two_stage_metrics["f1_macro"],
        two_stage_metrics["kappa_quadratic"],
        two_stage_metrics["recall_per_class"][2],
        two_stage_metrics["recall_per_class"][3],
        two_stage_metrics["recall_per_class"][4],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, base_values, width, label="Modelo actual", color="#2563eb")
    ax.bar(x + width / 2, new_values, width, label="Dos etapas", color="#16a34a")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Comparación en test", fontsize=16, fontweight="bold", loc="left")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(df, base_probs, stage1_probs, stage2_probs, preds, path):
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        out = {
            "id_code": row["id_code"],
            "true_class_id": int(row["diagnosis"]),
            "true_class_name": CLASS_NAMES[int(row["diagnosis"])],
            "predicted_class_id": int(preds[idx]),
            "predicted_class_name": CLASS_NAMES[int(preds[idx])],
            "stage1_referable_probability": float(stage1_probs[idx, 1]),
            "base_non_ref_probability_0": float(base_probs[idx, 0]),
            "base_non_ref_probability_1": float(base_probs[idx, 1]),
            "stage2_probability_2": float(stage2_probs[idx, 0]),
            "stage2_probability_3": float(stage2_probs[idx, 1]),
            "stage2_probability_4": float(stage2_probs[idx, 2]),
        }
        rows.append(out)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evalúa pipeline RetinaAI de dos etapas.")
    parser.add_argument("--base-model", type=Path, default=MODELS_DIR / "best_retina_model.pth")
    parser.add_argument("--stage1-model", type=Path, default=STAGE_DIR / "stage1_referable_model.pth")
    parser.add_argument("--stage2-model", type=Path, default=STAGE_DIR / "stage2_grade234_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, _ = load_model(args.base_model, device)
    stage1_model, _ = load_model(args.stage1_model, device)
    stage2_model, _ = load_model(args.stage2_model, device)

    val_df = pd.read_csv(SPLITS_DIR / "val_processed_300_split.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test_processed_300_split.csv")
    y_val = val_df["diagnosis"].astype(int).to_numpy()
    y_test = test_df["diagnosis"].astype(int).to_numpy()

    print("Prediciendo validación para elegir umbral...")
    val_base_probs, _ = predict_probs(base_model, val_df, device)
    val_stage1_probs, _ = predict_probs(stage1_model, val_df, device)
    val_stage2_probs, _ = predict_probs(stage2_model, val_df, device)
    best_threshold, threshold_rows = choose_threshold(y_val, val_base_probs, val_stage1_probs, val_stage2_probs)

    print("Prediciendo test...")
    test_base_probs, _ = predict_probs(base_model, test_df, device)
    test_stage1_probs, _ = predict_probs(stage1_model, test_df, device)
    test_stage2_probs, _ = predict_probs(stage2_model, test_df, device)

    base_preds = np.argmax(test_base_probs, axis=1)
    two_stage_preds = pipeline_predictions(
        test_base_probs,
        test_stage1_probs,
        test_stage2_probs,
        threshold=best_threshold["threshold"],
    )
    base_metrics = metrics_dict(y_test, base_preds)
    two_stage_metrics = metrics_dict(y_test, two_stage_preds)

    report = {
        "selected_threshold_from_validation": best_threshold,
        "threshold_candidates_validation": threshold_rows,
        "base_model_test": base_metrics,
        "two_stage_test": two_stage_metrics,
        "decision": {
            "promote_two_stage": bool(
                two_stage_metrics["kappa_quadratic"] >= base_metrics["kappa_quadratic"]
                and two_stage_metrics["f1_macro"] >= base_metrics["f1_macro"]
                and two_stage_metrics["large_error_rate_grade_distance_2_or_more"]
                <= base_metrics["large_error_rate_grade_distance_2_or_more"]
            ),
            "reason": "Se promueve solo si mejora kappa y F1 macro sin aumentar errores grandes.",
        },
    }
    save_json(report, RESULT_DIR / "two_stage_report.json")
    pd.DataFrame(threshold_rows).to_csv(RESULT_DIR / "threshold_candidates_validation.csv", index=False)
    save_predictions_csv(test_df, test_base_probs, test_stage1_probs, test_stage2_probs, two_stage_preds, RESULT_DIR / "two_stage_predictions.csv")
    plot_comparison(base_metrics, two_stage_metrics, RESULT_DIR / "two_stage_comparison.png")
    print(report)
    print(f"Reporte guardado en {RESULT_DIR / 'two_stage_report.json'}")


if __name__ == "__main__":
    main()
