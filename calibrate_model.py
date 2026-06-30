import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import (
    BATCH_SIZE,
    CALIBRATION_PATH,
    MODEL_NAME,
    MODELS_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    RESULTS_DIR,
    SPLITS_DIR,
    USE_RAW_SPLITS,
    USE_PROCESSED_IMAGES,
)
from src.dataset import RetinaDataset
from src.model import build_model
from src.preprocessing import get_eval_transforms
from src.utils import persistent_workers_enabled, save_json


def expected_calibration_error(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return ece


def collect_logits():
    checkpoint_path = MODELS_DIR / "best_retina_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("No existe models/best_retina_model.pth")
    if USE_RAW_SPLITS:
        val_path = SPLITS_DIR / f"val_raw_{IMAGE_SIZE}_split.csv"
    else:
        val_path = SPLITS_DIR / (f"val_processed_{IMAGE_SIZE}_split.csv" if USE_PROCESSED_IMAGES else "val_split.csv")
    if not val_path.exists():
        raise FileNotFoundError("No existe split de validacion. Ejecuta prepare_data.py")

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

    df = pd.read_csv(val_path)
    ds = RetinaDataset(df, transform=get_eval_transforms(retinal_preprocess=USE_RAW_SPLITS or not USE_PROCESSED_IMAGES))
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels, _, _ in tqdm(loader, desc="calibration"):
            images = images.to(device, non_blocking=True)
            logits = model(images).detach().cpu()
            logits_all.append(logits)
            labels_all.append(labels.detach().cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def optimize_temperature(logits, labels):
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=80)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.5, 5.0).item())


def probability_metrics(logits, labels, temperature):
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = y_pred == y_true
    return {
        "nll": float(log_loss(y_true, probs, labels=list(range(NUM_CLASSES)))),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "ece_10_bins": expected_calibration_error(confidences, correct, n_bins=10),
    }


def main():
    logits, labels = collect_logits()
    before = probability_metrics(logits, labels, temperature=1.0)
    temperature = optimize_temperature(logits, labels)
    after = probability_metrics(logits, labels, temperature=temperature)
    report = {
        "temperature": temperature,
        "validation_rows": int(len(labels)),
        "before": before,
        "after": after,
        "note": "La calibracion ajusta confianza, no cambia la clase predicha.",
    }
    save_json(report, CALIBRATION_PATH)
    save_json(report, RESULTS_DIR / "calibration.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
