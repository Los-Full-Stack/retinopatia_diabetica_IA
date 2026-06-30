import json
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import (
    BATCH_SIZE,
    CALIBRATION_PATH,
    CLASS_NAMES,
    EXPORT_DIR,
    IMAGE_SIZE,
    EXTERNAL_TRAINING_CSV,
    FINE_TUNE_BLOCKS,
    FOCAL_LOSS_ENABLED,
    ORDINAL_LOSS_ALPHA,
    ORDINAL_LOSS_ENABLED,
    IMAGE_SIZE,
    MODEL_NAME,
    MODELS_DIR,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    NUM_WORKERS,
    RESULTS_DIR,
    SAMPLER_WEIGHT_POWER,
    SPLITS_DIR,
    TEST_MODE,
    USE_RAW_SPLITS,
    USE_PROCESSED_IMAGES,
    USE_RETINAL_PREPROCESSING,
    USE_TTA,
    USE_EXTERNAL_TRAINING_DATA,
    USE_WEIGHTED_SAMPLER,
)
from src.dataset import RetinaDataset
from src.metrics import detect_model_collapse
from src.model import build_model
from src.preprocessing import get_eval_transforms
from src.utils import load_json, persistent_workers_enabled, save_json, setup_logging


def load_temperature():
    if not CALIBRATION_PATH.exists():
        return 1.0
    try:
        data = load_json(CALIBRATION_PATH)
        return max(float(data.get("temperature", 1.0)), 1e-3)
    except Exception:
        return 1.0


def plot_confusion(cm, path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES), range(NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f"{cm[i, j]:.2f}" if cm.dtype.kind == "f" else str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_class_metrics(precision, recall, f1, path):
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(NUM_CLASSES)
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bars = [
        ax.bar(x - width, precision, width, label="Precision", color="#2563eb"),
        ax.bar(x, recall, width, label="Recall", color="#ea580c"),
        ax.bar(x + width, f1, width, label="F1", color="#16a34a"),
    ]
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.annotate(
                f"{height * 100:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title("Metricas por grado", fontsize=16, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [f"Grado {idx}" for idx in range(NUM_CLASSES)])
    ax.legend(frameon=True, ncol=3, loc="upper center")
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def export_artifacts(metrics, validation):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for file in EXPORT_DIR.iterdir():
        if file.is_file() and file.name != ".gitkeep":
            file.unlink()
    shutil.copy2(MODELS_DIR / "best_retina_model.pth", EXPORT_DIR / "best_retina_model.pth")
    shutil.copy2(RESULTS_DIR / "metrics.json", EXPORT_DIR / "metrics.json")
    shutil.copy2(RESULTS_DIR / "model_validation.json", EXPORT_DIR / "model_validation.json")
    shutil.copy2(RESULTS_DIR / "classification_report.txt", EXPORT_DIR / "classification_report.txt")
    shutil.copy2(RESULTS_DIR / "confusion_matrix.png", EXPORT_DIR / "confusion_matrix.png")
    shutil.copy2(RESULTS_DIR / "normalized_confusion_matrix.png", EXPORT_DIR / "normalized_confusion_matrix.png")
    class_metrics_path = RESULTS_DIR / "class_metrics.png"
    if class_metrics_path.exists():
        shutil.copy2(class_metrics_path, EXPORT_DIR / "class_metrics.png")
    shutil.copy2("src/inference.py", EXPORT_DIR / "inference.py")
    shutil.copy2("src/quality.py", EXPORT_DIR / "quality.py")
    if CALIBRATION_PATH.exists():
        shutil.copy2(CALIBRATION_PATH, EXPORT_DIR / "calibration.json")
    save_json(CLASS_NAMES, EXPORT_DIR / "class_names.json")
    save_json(
        {
            "image_size": IMAGE_SIZE,
            "channels": 3,
            "color_mode": "RGB",
            "dtype": "float32",
            "normalization_mean": NORMALIZATION_MEAN,
            "normalization_std": NORMALIZATION_STD,
            "base_model": MODEL_NAME,
            "framework": "PyTorch",
        },
        EXPORT_DIR / "preprocessing_config.json",
    )
    if USE_RAW_SPLITS:
        train_split_path = SPLITS_DIR / f"train_with_external_raw_{IMAGE_SIZE}_split.csv" if USE_EXTERNAL_TRAINING_DATA else SPLITS_DIR / f"train_raw_{IMAGE_SIZE}_split.csv"
        val_split_path = SPLITS_DIR / f"val_raw_{IMAGE_SIZE}_split.csv"
        test_split_path = SPLITS_DIR / f"test_raw_{IMAGE_SIZE}_split.csv"
    else:
        processed_suffix = f"_processed_{IMAGE_SIZE}" if USE_PROCESSED_IMAGES else ""
        train_split_path = EXTERNAL_TRAINING_CSV if USE_EXTERNAL_TRAINING_DATA and USE_PROCESSED_IMAGES else SPLITS_DIR / f"train{processed_suffix}_split.csv"
        val_split_path = SPLITS_DIR / f"val{processed_suffix}_split.csv"
        test_split_path = SPLITS_DIR / f"test{processed_suffix}_split.csv"
    splits = {
        "train": len(pd.read_csv(train_split_path)),
        "val": len(pd.read_csv(val_split_path)),
        "test": len(pd.read_csv(test_split_path)),
    }
    save_json(
        {
            "architecture": MODEL_NAME,
            "framework": "PyTorch",
            "transfer_learning": True,
            "fine_tuning": True,
            "retinal_preprocessing": USE_RETINAL_PREPROCESSING,
            "processed_images": USE_PROCESSED_IMAGES,
            "raw_splits": USE_RAW_SPLITS,
            "external_training_data": USE_EXTERNAL_TRAINING_DATA,
            "external_training_csv": str(EXTERNAL_TRAINING_CSV) if USE_EXTERNAL_TRAINING_DATA else None,
            "tta": USE_TTA,
            "calibration_temperature": metrics.get("calibration_temperature", 1.0),
            "weighted_sampler": USE_WEIGHTED_SAMPLER,
            "sampler_weight_power": SAMPLER_WEIGHT_POWER,
            "focal_loss": FOCAL_LOSS_ENABLED,
            "ordinal_loss": ORDINAL_LOSS_ENABLED,
            "ordinal_loss_alpha": ORDINAL_LOSS_ALPHA,
            "fine_tune_blocks": FINE_TUNE_BLOCKS,
            "classes": CLASS_NAMES,
            "image_count": sum(splits.values()),
            "train_size": splits["train"],
            "val_size": splits["val"],
            "test_size": splits["test"],
            "batch_size": BATCH_SIZE,
            "epochs": "2+1 en TEST_MODE, 12+8 en modo completo",
            "test_mode": TEST_MODE,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "kappa_quadratic": metrics["kappa_quadratic"],
            "approved": validation["approved_for_export"],
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        EXPORT_DIR / "model_info.json",
    )
    (EXPORT_DIR / "requirements_inference.txt").write_text("torch\ntorchvision\npillow\n", encoding="utf-8")
    zip_path = EXPORT_DIR / "retina_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in EXPORT_DIR.iterdir():
            if file.is_file() and file.name != zip_path.name:
                zf.write(file, arcname=file.name)


def main():
    logger = setup_logging()
    checkpoint_path = MODELS_DIR / "best_retina_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("No existe models/best_retina_model.pth. Ejecuta primero python train.py")
    if USE_RAW_SPLITS:
        test_path = SPLITS_DIR / f"test_raw_{IMAGE_SIZE}_split.csv"
    else:
        test_path = SPLITS_DIR / (f"test_processed_{IMAGE_SIZE}_split.csv" if USE_PROCESSED_IMAGES else "test_split.csv")
    if not test_path.exists():
        raise FileNotFoundError("No existe data/splits/test_split.csv. Ejecuta primero python prepare_data.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    temperature = load_temperature()
    logger.info("Temperatura de calibracion aplicada en evaluacion: %.6f", temperature)

    test_df = pd.read_csv(test_path)
    ds = RetinaDataset(test_df, transform=get_eval_transforms(retinal_preprocess=USE_RAW_SPLITS or not USE_PROCESSED_IMAGES))
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )
    criterion = torch.nn.CrossEntropyLoss()
    losses, y_true, y_pred, probs_all, ids, paths = [], [], [], [], [], []
    with torch.no_grad():
        for images, labels, image_ids, image_paths in tqdm(loader, desc="evaluate"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if USE_TTA:
                logits_flip = model(torch.flip(images, dims=[3]))
                logits = (logits + logits_flip) / 2
            loss = criterion(logits, labels)
            probs = torch.softmax(logits / temperature, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            probs_all.extend(probs.tolist())
            ids.extend(list(image_ids))
            paths.extend(list(image_paths))

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    probs_np = np.asarray(probs_all)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(NUM_CLASSES)))
    cm_norm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(NUM_CLASSES)), normalize="true")
    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(y_true_np, probs_np, multi_class="ovr", labels=list(range(NUM_CLASSES))))
    except ValueError:
        pass
    class_data = load_json(RESULTS_DIR / "class_distribution.json")
    majority_baseline = class_data["majority_baseline"]
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic")),
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
        "roc_auc_multiclass": roc_auc,
        "true_distribution": np.bincount(y_true_np, minlength=NUM_CLASSES).tolist(),
        "predicted_distribution": np.bincount(y_pred_np, minlength=NUM_CLASSES).tolist(),
        "majority_baseline": majority_baseline,
        "calibration_temperature": temperature,
    }
    save_json(metrics, RESULTS_DIR / "metrics.json")
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_true_np, y_pred_np, target_names=CLASS_NAMES, zero_division=0),
        encoding="utf-8",
    )
    plot_confusion(cm, RESULTS_DIR / "confusion_matrix.png", "Matriz de confusión")
    plot_confusion(cm_norm, RESULTS_DIR / "normalized_confusion_matrix.png", "Matriz normalizada")
    plot_class_metrics(precision, recall, f1, RESULTS_DIR / "class_metrics.png")

    pred_rows = []
    for i, image_id in enumerate(ids):
        row = {
            "id_code": image_id,
            "image_path": paths[i],
            "true_class_id": int(y_true_np[i]),
            "true_class_name": CLASS_NAMES[int(y_true_np[i])],
            "predicted_class_id": int(y_pred_np[i]),
            "predicted_class_name": CLASS_NAMES[int(y_pred_np[i])],
            "confidence": float(probs_np[i].max()),
            "calibration_temperature": temperature,
        }
        for cls in range(NUM_CLASSES):
            row[f"probability_class_{cls}"] = float(probs_np[i, cls])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "predictions.csv", index=False)

    validation = detect_model_collapse(y_true_np, y_pred_np, probs_np, majority_baseline)
    save_json(validation, RESULTS_DIR / "model_validation.json")
    if validation["approved_for_export"]:
        export_artifacts(metrics, validation)
        logger.info("Modelo aprobado y exportado.")
    else:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        for file in EXPORT_DIR.iterdir():
            if file.is_file() and file.name != ".gitkeep":
                file.unlink()
        save_json({"validation": validation, "message": "Modelo rechazado por colapso."}, RESULTS_DIR / "FAILED_MODEL_REPORT.json")
        logger.warning("Modelo rechazado. No se creó retina_export.zip.")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
