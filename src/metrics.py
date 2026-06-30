import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


def compute_metrics(y_true, y_pred, num_classes=5):
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "predicted_distribution": np.bincount(np.asarray(y_pred), minlength=num_classes).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def detect_model_collapse(y_true, y_pred, probabilities, majority_baseline) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_predicted = np.unique(y_pred)
    pred_counts = np.bincount(y_pred, minlength=5)
    pred_distribution = (pred_counts / max(len(y_pred), 1)).tolist()
    metrics = compute_metrics(y_true, y_pred, num_classes=5)
    cm = np.asarray(metrics["confusion_matrix"])
    reasons = []

    if len(unique_predicted) == 1:
        reasons.append("El modelo predice una sola clase.")
    if pred_counts.max() / max(len(y_pred), 1) > 0.90:
        reasons.append("Más del 90% de predicciones pertenecen a una sola clase.")
    if all(metrics["recall_per_class"][i] == 0 for i in range(1, 5)):
        reasons.append("El recall de las clases 1-4 es cero.")
    positive_recall_classes = sum(1 for value in metrics["recall_per_class"] if value > 0)
    if positive_recall_classes < 3:
        reasons.append("Menos de tres clases tienen recall mayor que cero.")
    if any(metrics["recall_per_class"][i] == 0 for i in range(1, 5)):
        reasons.append("Al menos una clase clínica minoritaria tiene recall cero.")
    if abs(metrics["accuracy"] - majority_baseline) <= 0.03:
        reasons.append("La accuracy está prácticamente igual al baseline mayoritario.")
    if np.count_nonzero(cm.sum(axis=0)) <= 1:
        reasons.append("La matriz de confusión concentra predicciones en una sola columna.")
    if metrics["f1_macro"] < max(0.15, majority_baseline * 0.5):
        reasons.append("F1 macro muy bajo.")

    collapsed = bool(reasons)
    return {
        "collapsed": collapsed,
        "approved_for_export": not collapsed,
        "unique_predicted_classes": int(len(unique_predicted)),
        "predicted_distribution": pred_distribution,
        "majority_baseline": float(majority_baseline),
        "test_accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "recall_per_class": metrics["recall_per_class"],
        "reasons": reasons,
    }
