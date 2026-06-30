import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config.config import CLASS_NAMES, MIN_PREDICTION_CONFIDENCE, RESULTS_DIR
from src.utils import load_json, save_json


def binary_stats(y_true, y_pred, threshold):
    true_pos = y_true >= threshold
    pred_pos = y_pred >= threshold
    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold_grade": int(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def coverage_by_confidence(df):
    rows = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        subset = df[df["confidence"] >= threshold]
        if subset.empty:
            rows.append({"confidence_threshold": threshold, "coverage": 0.0, "accuracy": None, "mean_abs_grade_error": None})
            continue
        abs_error = (subset["predicted_class_id"] - subset["true_class_id"]).abs()
        rows.append(
            {
                "confidence_threshold": threshold,
                "coverage": float(len(subset) / len(df)),
                "accuracy": float((subset["predicted_class_id"] == subset["true_class_id"]).mean()),
                "mean_abs_grade_error": float(abs_error.mean()),
            }
        )
    return rows


def save_confidence_plot(rows, path):
    thresholds = [row["confidence_threshold"] for row in rows]
    coverage = [row["coverage"] for row in rows]
    accuracy = [np.nan if row["accuracy"] is None else row["accuracy"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax1.plot(thresholds, coverage, marker="o", linewidth=2.5, color="#2563eb", label="Cobertura")
    ax1.set_xlabel("Umbral de confianza")
    ax1.set_ylabel("Cobertura")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, accuracy, marker="o", linewidth=2.5, color="#16a34a", label="Accuracy aceptada")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.35)
    fig.suptitle("Cobertura vs accuracy por confianza", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    predictions_path = RESULTS_DIR / "predictions.csv"
    metrics_path = RESULTS_DIR / "metrics.json"
    if not predictions_path.exists():
        raise FileNotFoundError("No existe results/predictions.csv. Ejecuta evaluate.py primero.")
    df = pd.read_csv(predictions_path)
    metrics = load_json(metrics_path) if metrics_path.exists() else {}

    y_true = df["true_class_id"].to_numpy()
    y_pred = df["predicted_class_id"].to_numpy()
    abs_error = np.abs(y_pred - y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    per_class = []
    for grade, name in enumerate(CLASS_NAMES):
        true_mask = y_true == grade
        pred_mask = y_pred == grade
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        tn = int(np.sum(~true_mask & ~pred_mask))
        per_class.append(
            {
                "grade": grade,
                "name": name,
                "support": int(true_mask.sum()),
                "sensitivity_recall": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "precision": float(tp / max(tp + fp, 1)),
            }
        )

    large_errors = df.assign(abs_grade_error=abs_error).sort_values(
        ["abs_grade_error", "confidence"], ascending=[False, False]
    )
    large_errors = large_errors[large_errors["abs_grade_error"] >= 2].head(40)
    large_errors_path = RESULTS_DIR / "large_grade_errors.csv"
    large_errors.to_csv(large_errors_path, index=False)

    confidence_rows = coverage_by_confidence(df)
    save_confidence_plot(confidence_rows, RESULTS_DIR / "confidence_coverage.png")

    selected_policy = next(
        (
            row
            for row in confidence_rows
            if abs(row["confidence_threshold"] - MIN_PREDICTION_CONFIDENCE) < 1e-9
        ),
        None,
    )
    report = {
        "current_model_is_clinical_device": False,
        "clinical_use_statement": "Prototipo de investigacion. No aprobado para diagnostico autonomo.",
        "operating_confidence_threshold": MIN_PREDICTION_CONFIDENCE,
        "selected_confidence_policy": selected_policy,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "kappa_quadratic": metrics.get("kappa_quadratic"),
            "mean_abs_grade_error": float(abs_error.mean()),
            "large_error_rate_grade_distance_2_or_more": float(np.mean(abs_error >= 2)),
        },
        "referable_dr_grade_2_or_more": binary_stats(y_true, y_pred, threshold=2),
        "severe_dr_grade_3_or_more": binary_stats(y_true, y_pred, threshold=3),
        "per_grade_clinical_metrics": per_class,
        "confidence_policy_candidates": confidence_rows,
        "confusion_matrix": cm.tolist(),
        "large_errors_csv": str(large_errors_path),
        "minimum_recommended_next_steps": [
            "Validacion externa con datos locales no vistos.",
            "Revision de errores grandes por especialista.",
            "Control de calidad de imagen obligatorio antes de inferencia.",
            "Umbral de confianza con opcion de repetir captura o derivar.",
            "Prueba prospectiva antes de cualquier uso clinico real.",
        ],
    }
    save_json(report, RESULTS_DIR / "clinical_audit.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
