import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE_BLOCKS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    CLASS_WEIGHT_MULTIPLIERS,
    FOCAL_LOSS_ENABLED,
    EXTERNAL_TRAINING_CSV,
    INITIAL_EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    MODEL_NAME,
    NUM_WORKERS,
    SPLITS_DIR,
    SAMPLER_WEIGHT_POWER,
    TEST_MODE,
    USE_RAW_SPLITS,
    USE_PROCESSED_IMAGES,
    USE_EXTERNAL_TRAINING_DATA,
    ORDINAL_LOSS_ENABLED,
    ORDINAL_LOSS_ALPHA,
    USE_RETINAL_PREPROCESSING,
    USE_WEIGHTED_SAMPLER,
)
from src.dataset import RetinaDataset
from src.model import build_model, unfreeze_last_blocks
from src.preprocessing import get_eval_transforms, get_train_transforms
from src.trainer import (
    FocalLoss,
    OrdinalCrossEntropyLoss,
    calculate_class_weights,
    fit,
    promote_best_model,
    save_training_plots,
)
from src.utils import persistent_workers_enabled, print_cuda_report, set_seed, setup_logging


def make_weighted_sampler(labels):
    counts = labels.value_counts().sort_index()
    sample_weights = labels.map(lambda label: (1.0 / counts[int(label)]) ** SAMPLER_WEIGHT_POWER).astype(float).tolist()
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_loader(dataset, shuffle, device, sampler=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers_enabled(NUM_WORKERS),
    )


def main():
    logger = setup_logging()
    set_seed()
    device = print_cuda_report(logger)

    if USE_RAW_SPLITS and USE_EXTERNAL_TRAINING_DATA:
        train_path = SPLITS_DIR / f"train_with_external_raw_{IMAGE_SIZE}_split.csv"
    elif USE_RAW_SPLITS:
        train_path = SPLITS_DIR / f"train_raw_{IMAGE_SIZE}_split.csv"
    elif USE_EXTERNAL_TRAINING_DATA and USE_PROCESSED_IMAGES:
        train_path = EXTERNAL_TRAINING_CSV
    else:
        train_path = SPLITS_DIR / (f"train_processed_{IMAGE_SIZE}_split.csv" if USE_PROCESSED_IMAGES else "train_split.csv")
    if USE_RAW_SPLITS:
        val_path = SPLITS_DIR / f"val_raw_{IMAGE_SIZE}_split.csv"
    else:
        val_path = SPLITS_DIR / (f"val_processed_{IMAGE_SIZE}_split.csv" if USE_PROCESSED_IMAGES else "val_split.csv")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Ejecuta primero: python prepare_data.py, python prepare_processed.py y python prepare_external_data.py")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    apply_retinal_preprocess = USE_RAW_SPLITS or not USE_PROCESSED_IMAGES
    train_ds = RetinaDataset(train_df, transform=get_train_transforms(retinal_preprocess=apply_retinal_preprocess))
    val_ds = RetinaDataset(val_df, transform=get_eval_transforms(retinal_preprocess=apply_retinal_preprocess))
    sampler = make_weighted_sampler(train_df["diagnosis"]) if USE_WEIGHTED_SAMPLER else None
    train_loader = make_loader(train_ds, shuffle=True, device=device, sampler=sampler)
    val_loader = make_loader(val_ds, shuffle=False, device=device)
    logger.info(
        "Experimento: model=%s processed_images=%s raw_splits=%s external_training=%s retinal_preprocessing=%s weighted_sampler=%s focal_loss=%s ordinal_loss=%s fine_tune_blocks=%s",
        MODEL_NAME,
        USE_PROCESSED_IMAGES,
        USE_RAW_SPLITS,
        USE_EXTERNAL_TRAINING_DATA,
        USE_RETINAL_PREPROCESSING,
        USE_WEIGHTED_SAMPLER,
        FOCAL_LOSS_ENABLED,
        ORDINAL_LOSS_ENABLED,
        FINE_TUNE_BLOCKS,
    )
    logger.info("Sampler weight power: %s", SAMPLER_WEIGHT_POWER if USE_WEIGHTED_SAMPLER else None)
    logger.info("Class weight multipliers: %s", CLASS_WEIGHT_MULTIPLIERS)

    sample_images, sample_labels, _, _ = next(iter(train_loader))
    print(f"Batch: shape={tuple(sample_images.shape)}, dtype={sample_images.dtype}, labels={sample_labels.dtype}")
    logger.info("Batch ejemplo: shape=%s dtype=%s", tuple(sample_images.shape), sample_images.dtype)

    class_weights = calculate_class_weights(train_df["diagnosis"].tolist(), device)
    logger.info("Pesos de clase: %s", class_weights.detach().cpu().tolist())
    if FOCAL_LOSS_ENABLED:
        criterion = FocalLoss(
            class_weights=class_weights,
            ordinal_alpha=ORDINAL_LOSS_ALPHA if ORDINAL_LOSS_ENABLED else 0.0,
        )
    elif ORDINAL_LOSS_ENABLED:
        criterion = OrdinalCrossEntropyLoss(class_weights=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(device)

    initial_epochs = 2 if TEST_MODE else INITIAL_EPOCHS
    fine_tune_epochs = 1 if TEST_MODE else FINE_TUNE_EPOCHS

    model = build_model(freeze_backbone=True, model_name=MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    initial_history, initial_path = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        initial_epochs,
        EARLY_STOPPING_PATIENCE,
        "best_initial_model.pth",
        logger,
    )

    checkpoint = torch.load(initial_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    unfreeze_last_blocks(model, num_blocks=FINE_TUNE_BLOCKS)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=FINE_TUNE_LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    fine_history, fine_path = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        fine_tune_epochs,
        EARLY_STOPPING_PATIENCE,
        "best_finetuned_model.pth",
        logger,
    )
    final_path = promote_best_model(fine_path if fine_path.exists() else initial_path)
    save_training_plots(initial_history + fine_history)
    logger.info("Modelo principal guardado en %s", final_path)
    print(f"Modelo principal guardado en {MODELS_DIR / 'best_retina_model.pth'}")


if __name__ == "__main__":
    main()
