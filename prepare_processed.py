from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config.config import (
    IMAGE_SIZE,
    PROCESSED_DIR,
    PROCESSED_IMAGE_QUALITY,
    RESULTS_DIR,
    SPLITS_DIR,
)
from src.preprocessing import preprocess_retina_image
from src.utils import save_json, setup_logging


def process_csv(csv_path: Path, output_csv_path: Path, logger):
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError(f"{csv_path} no tiene columna image_path")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_paths = []
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"process {csv_path.name}"):
        image_id = str(row["id_code"])
        source = Path(row["image_path"])
        target = PROCESSED_DIR / f"{image_id}.jpg"
        if not target.exists():
            try:
                with Image.open(source) as image:
                    processed = preprocess_retina_image(image, output_size=IMAGE_SIZE)
                    processed.save(target, "JPEG", quality=PROCESSED_IMAGE_QUALITY, optimize=True)
            except Exception as exc:
                failed.append({"id_code": image_id, "path": str(source), "error": str(exc)})
                continue
        processed_paths.append(str(target))
    if failed:
        bad_ids = {item["id_code"] for item in failed}
        df = df.loc[~df["id_code"].astype(str).isin(bad_ids)].copy()
    df["original_image_path"] = df["image_path"]
    df["image_path"] = processed_paths
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info("CSV procesado %s -> %s con %s filas", csv_path, output_csv_path, len(df))
    return {"source": str(csv_path), "output": str(output_csv_path), "rows": len(df), "failed": failed}


def main():
    logger = setup_logging()
    split_files = ["train_split.csv", "val_split.csv", "test_split.csv"]
    reports = []
    for filename in split_files:
        source = SPLITS_DIR / filename
        if not source.exists():
            raise FileNotFoundError(f"No existe {source}. Ejecuta primero python prepare_data.py")
        output = SPLITS_DIR / filename.replace("_split.csv", f"_processed_{IMAGE_SIZE}_split.csv")
        reports.append(process_csv(source, output, logger))
    total_processed = len(list(PROCESSED_DIR.glob("*.jpg")))
    report = {
        "processed_dir": str(PROCESSED_DIR),
        "image_size": IMAGE_SIZE,
        "quality": PROCESSED_IMAGE_QUALITY,
        "total_processed_files": total_processed,
        "splits": reports,
    }
    save_json(report, RESULTS_DIR / f"processed_{IMAGE_SIZE}_report.json")
    print(report)


if __name__ == "__main__":
    main()
