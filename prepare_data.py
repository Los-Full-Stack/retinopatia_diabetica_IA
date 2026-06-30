import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from config.config import (
    CHECK_ALL_IMAGES,
    CLASS_NAMES,
    CSV_PATH,
    EXTRACT_DIR,
    FORCE_REEXTRACT,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    REUSE_SPLITS,
    SPLITS_DIR,
    TEST_MODE,
    TEST_SAMPLE_SIZE,
    ZIP_PATH,
)
from src.utils import ensure_dirs, save_json, setup_logging

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def validate_required_files():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe {CSV_PATH}")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"No existe {ZIP_PATH}")
    if not zipfile.is_zipfile(ZIP_PATH):
        raise zipfile.BadZipFile(f"ZIP dañado o inválido: {ZIP_PATH}")


def find_image_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def extract_if_needed(logger):
    ensure_dirs(EXTRACT_DIR)
    existing = find_image_files(EXTRACT_DIR)
    if len(existing) > 100 and not FORCE_REEXTRACT:
        logger.info("Reutilizando extracción existente con %s imágenes.", len(existing))
    else:
        logger.info("Descomprimiendo %s en %s", ZIP_PATH, EXTRACT_DIR)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    return select_image_root(EXTRACT_DIR, logger)


def select_image_root(root: Path, logger):
    image_counts = Counter()
    all_images = find_image_files(root)
    for img in all_images:
        image_counts[img.parent] += 1
    if not image_counts:
        raise RuntimeError(f"No se encontraron imágenes en {root}")
    selected_dir, count = image_counts.most_common(1)[0]
    extensions = sorted({p.suffix.lower() for p in all_images})
    logger.info("Ruta seleccionada: %s", selected_dir)
    logger.info("Número de imágenes: %s", count)
    logger.info("Extensiones encontradas: %s", extensions)
    print(f"Ruta seleccionada: {selected_dir}")
    print(f"Número de imágenes: {count}")
    print(f"Extensiones encontradas: {extensions}")
    return selected_dir, all_images


def validate_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
        return width > 0 and height > 0
    except Exception:
        return False


def validate_csv_and_images(image_files, logger):
    df = pd.read_csv(CSV_PATH)
    required = {"id_code", "diagnosis"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en train.csv: {sorted(missing_cols)}")
    if not df["diagnosis"].isin(range(5)).all():
        bad = sorted(df.loc[~df["diagnosis"].isin(range(5)), "diagnosis"].unique().tolist())
        raise ValueError(f"Etiquetas fuera de 0..4: {bad}")

    duplicate_ids = sorted(df.loc[df["id_code"].duplicated(), "id_code"].astype(str).unique().tolist())
    image_map = {}
    for path in image_files:
        key = path.stem
        if key not in image_map or path.suffix.lower() == ".png":
            image_map[key] = path

    records = []
    missing_images = []
    for _, row in df.iterrows():
        image_id = str(row["id_code"])
        image_path = image_map.get(image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue
        records.append({"id_code": image_id, "diagnosis": int(row["diagnosis"]), "image_path": str(image_path)})

    valid_df = pd.DataFrame(records)
    corrupt = []
    paths_to_check = valid_df["image_path"].tolist() if CHECK_ALL_IMAGES else valid_df["image_path"].sample(
        min(200, len(valid_df)), random_state=RANDOM_SEED
    ).tolist()
    if not CHECK_ALL_IMAGES:
        logger.warning("CHECK_ALL_IMAGES=False: solo se revisa una muestra de imágenes.")

    bad_paths = set()
    for path_text in paths_to_check:
        path = Path(path_text)
        if not validate_image(path):
            corrupt.append(str(path))
            bad_paths.add(str(path))
    if bad_paths:
        valid_df = valid_df.loc[~valid_df["image_path"].isin(bad_paths)].copy()

    class_counts = valid_df["diagnosis"].value_counts().sort_index()
    percentages = (class_counts / len(valid_df) * 100).round(4) if len(valid_df) else class_counts
    report = {
        "original_rows": int(len(df)),
        "images_found": int(len(valid_df) + len(corrupt)),
        "missing_images": {"count": int(len(missing_images)), "ids": missing_images[:100]},
        "corrupt_images": {"count": int(len(corrupt)), "paths": corrupt[:100]},
        "duplicates": {"count": int(len(duplicate_ids)), "ids": duplicate_ids[:100]},
        "valid_records": int(len(valid_df)),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_percentages": {str(k): float(percentages.get(k, 0.0)) for k in range(5)},
    }
    save_json(report, RESULTS_DIR / "dataset_validation.json")
    valid_df.to_csv(RESULTS_DIR / "valid_dataset.csv", index=False)
    return valid_df, report


def analyze_class_distribution(df):
    counts = df["diagnosis"].value_counts().sort_index()
    total = len(df)
    percentages = counts / total * 100
    majority_class = int(counts.idxmax())
    majority_count = int(counts.max())
    majority_baseline = majority_count / total
    data = {
        "counts": {str(k): int(counts.get(k, 0)) for k in range(5)},
        "percentages": {str(k): float(percentages.get(k, 0.0)) for k in range(5)},
        "majority_class": majority_class,
        "majority_class_name": CLASS_NAMES[majority_class],
        "majority_baseline": float(majority_baseline),
    }
    save_json(data, RESULTS_DIR / "class_distribution.json")
    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in range(5)], [counts.get(i, 0) for i in range(5)], color="#2f6f73")
    plt.xlabel("Clase")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución de clases")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
    plt.close()
    print("Distribución de clases:")
    for i in range(5):
        print(f"  {i} - {CLASS_NAMES[i]}: {counts.get(i, 0)} ({percentages.get(i, 0):.2f}%)")
    print(f"Clase mayoritaria: {majority_class}")
    print(f"Baseline clase mayoritaria: {majority_baseline:.4f}")
    return data


def stratified_sample(df):
    if not TEST_MODE or len(df) <= TEST_SAMPLE_SIZE:
        return df.copy()
    sample_size = min(TEST_SAMPLE_SIZE, len(df))
    _, sample = train_test_split(
        df,
        test_size=sample_size,
        stratify=df["diagnosis"],
        random_state=RANDOM_SEED,
    )
    return sample.reset_index(drop=True)


def create_splits(df, logger):
    ensure_dirs(SPLITS_DIR)
    paths = [SPLITS_DIR / "train_split.csv", SPLITS_DIR / "val_split.csv", SPLITS_DIR / "test_split.csv"]
    if REUSE_SPLITS and all(p.exists() for p in paths):
        logger.info("Reutilizando splits existentes.")
        return [pd.read_csv(p) for p in paths]

    work_df = stratified_sample(df)
    if set(work_df["diagnosis"].unique()) != set(range(5)):
        raise RuntimeError("La muestra de trabajo no conserva las cinco clases.")
    train_df, temp_df = train_test_split(
        work_df,
        test_size=0.30,
        stratify=work_df["diagnosis"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["diagnosis"],
        random_state=RANDOM_SEED,
    )
    splits = [train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)]
    names = ["train", "val", "test"]
    for split, name, path in zip(splits, names, paths):
        if set(split["diagnosis"].unique()) != set(range(5)):
            raise RuntimeError(f"El split {name} no contiene las cinco clases.")
        split[["id_code", "diagnosis", "image_path"]].to_csv(path, index=False)

    id_sets = [set(s["id_code"]) for s in splits]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError("Hay intersecciones entre splits.")
    if sum(len(s) for s in splits) != len(work_df):
        raise RuntimeError("La suma de splits no coincide con el dataset de trabajo.")
    logger.info("Splits creados: train=%s val=%s test=%s", *(len(s) for s in splits))
    return splits


def main():
    ensure_dirs(EXTRACT_DIR, SPLITS_DIR, RESULTS_DIR)
    logger = setup_logging()
    logger.info("Inicio preparación de dataset en %s", PROJECT_ROOT)
    validate_required_files()
    selected_dir, image_files = extract_if_needed(logger)
    valid_df, report = validate_csv_and_images(image_files, logger)
    distribution = analyze_class_distribution(valid_df)
    train_df, val_df, test_df = create_splits(valid_df, logger)
    summary = {
        "selected_image_dir": str(selected_dir),
        "valid_records": report["valid_records"],
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "majority_baseline": distribution["majority_baseline"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Preparación completada: %s", summary)


if __name__ == "__main__":
    main()
