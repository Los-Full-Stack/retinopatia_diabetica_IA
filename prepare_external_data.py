import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config.config import (
    EXTERNAL_DIR,
    IMAGE_SIZE,
    PROCESSED_DIR,
    PROCESSED_IMAGE_QUALITY,
    RESULTS_DIR,
    SPLITS_DIR,
)
from src.preprocessing import preprocess_retina_image
from src.utils import save_json, setup_logging


EXTERNAL_ZIP = EXTERNAL_DIR / "retina_eyepacs_resized_subset.zip"
EXTRACTED_EXTERNAL_DIR = EXTERNAL_DIR / "eyepacs_resized_subset"
EXTERNAL_PROCESSED_PREFIX = "eyepacs_"


def extract_external_zip(logger):
    if not EXTERNAL_ZIP.exists():
        raise FileNotFoundError(f"No existe {EXTERNAL_ZIP}")
    marker = EXTRACTED_EXTERNAL_DIR / "external_labels.csv"
    if marker.exists():
        logger.info("Dataset externo ya extraido en %s", EXTRACTED_EXTERNAL_DIR)
        return
    EXTRACTED_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EXTERNAL_ZIP, "r") as zf:
        zf.extractall(EXTRACTED_EXTERNAL_DIR)
    logger.info("Dataset externo extraido en %s", EXTRACTED_EXTERNAL_DIR)


def prepare_external_processed(logger):
    labels_path = EXTRACTED_EXTERNAL_DIR / "external_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"No existe {labels_path}")

    df = pd.read_csv(labels_path)
    required = {"id_code", "diagnosis", "image_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en externo: {sorted(missing)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="process external"):
        source_id = str(row["id_code"])
        image_id = f"{EXTERNAL_PROCESSED_PREFIX}{source_id}"
        source = EXTRACTED_EXTERNAL_DIR / str(row["image_path"])
        target = PROCESSED_DIR / f"{image_id}.jpg"
        if not target.exists():
            try:
                with Image.open(source) as image:
                    processed = preprocess_retina_image(image, output_size=IMAGE_SIZE)
                    processed.save(target, "JPEG", quality=PROCESSED_IMAGE_QUALITY, optimize=True)
            except Exception as exc:
                failed.append({"id_code": source_id, "path": str(source), "error": str(exc)})
                continue
        new_row = row.to_dict()
        new_row["id_code"] = image_id
        new_row["original_id_code"] = source_id
        new_row["original_image_path"] = str(source)
        new_row["image_path"] = str(target)
        new_row["source_dataset"] = new_row.get("source_dataset", "eyepacs_resized")
        rows.append(new_row)

    external_processed = pd.DataFrame(rows)
    external_processed_path = SPLITS_DIR / "external_eyepacs_processed_300.csv"
    external_processed.to_csv(external_processed_path, index=False)
    logger.info("CSV externo procesado: %s filas en %s", len(external_processed), external_processed_path)
    return external_processed, external_processed_path, failed


def combine_train_with_external(external_df, logger):
    train_path = SPLITS_DIR / "train_processed_300_split.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"No existe {train_path}. Ejecuta primero python prepare_processed.py")

    train_df = pd.read_csv(train_path)
    train_df["source_dataset"] = train_df.get("source_dataset", "aptos2019")
    train_df["is_external"] = False
    external_df = external_df.copy()
    external_df["is_external"] = True

    common_columns = sorted(set(train_df.columns).union(external_df.columns))
    combined = pd.concat(
        [train_df.reindex(columns=common_columns), external_df.reindex(columns=common_columns)],
        ignore_index=True,
    ).sample(frac=1, random_state=42)

    output_path = SPLITS_DIR / "train_with_external_processed_300_split.csv"
    combined.to_csv(output_path, index=False)
    logger.info("Train combinado guardado en %s con %s filas", output_path, len(combined))
    return combined, output_path


def main():
    logger = setup_logging()
    extract_external_zip(logger)
    external_df, external_processed_path, failed = prepare_external_processed(logger)
    combined, combined_path = combine_train_with_external(external_df, logger)
    report = {
        "external_zip": str(EXTERNAL_ZIP),
        "extracted_dir": str(EXTRACTED_EXTERNAL_DIR),
        "external_processed_csv": str(external_processed_path),
        "combined_train_csv": str(combined_path),
        "external_rows": int(len(external_df)),
        "combined_rows": int(len(combined)),
        "external_distribution": {
            str(k): int(v) for k, v in external_df["diagnosis"].value_counts().sort_index().items()
        },
        "combined_distribution": {
            str(k): int(v) for k, v in combined["diagnosis"].value_counts().sort_index().items()
        },
        "failed": failed,
    }
    save_json(report, RESULTS_DIR / "external_data_report.json")
    print(report)


if __name__ == "__main__":
    main()
