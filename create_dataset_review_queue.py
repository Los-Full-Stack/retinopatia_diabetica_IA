import argparse
from pathlib import Path

import pandas as pd

from config.config import RESULTS_DIR, SPLITS_DIR


RAW_QUALITY_FILES = {
    "train": SPLITS_DIR / "train_with_external_raw_300_quality.csv",
    "val": SPLITS_DIR / "val_raw_300_quality.csv",
    "test": SPLITS_DIR / "test_raw_300_quality.csv",
}


SPLIT_FILES = {
    "train": SPLITS_DIR / "train_processed_300_split.csv",
    "val": SPLITS_DIR / "val_processed_300_split.csv",
    "test": SPLITS_DIR / "test_processed_300_split.csv",
}


def yes_no(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def load_quality(split: str) -> pd.DataFrame:
    path = RAW_QUALITY_FILES[split]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if split == "train" and "is_external" in df.columns:
        df = df[~df["is_external"].astype(bool)].copy()
    columns = [
        "id_code",
        "raw_quality_score",
        "raw_sharpness",
        "raw_contrast",
        "raw_brightness",
        "raw_retina_coverage",
        "raw_acceptable",
        "raw_warnings",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].drop_duplicates("id_code")


def priority_reason(row: pd.Series) -> str:
    reasons = []
    if row["split"] == "train":
        reasons.append("entrenamiento")
    if int(row["diagnosis"]) in {3, 4}:
        reasons.append("grado_severo_escaso")
    elif int(row["diagnosis"]) == 2:
        reasons.append("grado_2_prioritario")
    if not yes_no(row.get("raw_acceptable", True)):
        reasons.append("calidad_original_baja")
    if float(row.get("raw_sharpness") or 999.0) < 12.0:
        reasons.append("posible_borrosa")
    return "|".join(reasons)


def priority_score(row: pd.Series) -> float:
    score = 0.0
    diagnosis = int(row["diagnosis"])
    if row["split"] == "train":
        score += 20.0
    elif row["split"] == "val":
        score += 8.0
    if diagnosis == 4:
        score += 30.0
    elif diagnosis == 3:
        score += 28.0
    elif diagnosis == 2:
        score += 18.0
    elif diagnosis == 1:
        score += 6.0

    raw_quality = row.get("raw_quality_score")
    if pd.notna(raw_quality):
        score += max(0.0, (0.9 - float(raw_quality)) * 20.0)
    raw_sharpness = row.get("raw_sharpness")
    if pd.notna(raw_sharpness):
        score += max(0.0, (18.0 - float(raw_sharpness)) * 0.75)
    if not yes_no(row.get("raw_acceptable", True)):
        score += 10.0
    return float(score)


def build_queue(max_rows: int) -> pd.DataFrame:
    frames = []
    for split, split_path in SPLIT_FILES.items():
        if not split_path.exists():
            raise FileNotFoundError(f"No existe {split_path}")
        df = pd.read_csv(split_path)
        df["split"] = split
        quality = load_quality(split)
        if not quality.empty:
            df = df.merge(quality, on="id_code", how="left")
        else:
            for column in [
                "raw_quality_score",
                "raw_sharpness",
                "raw_contrast",
                "raw_brightness",
                "raw_retina_coverage",
                "raw_acceptable",
                "raw_warnings",
            ]:
                df[column] = pd.NA
        frames.append(df)

    queue = pd.concat(frames, ignore_index=True, sort=False)
    queue["priority_score"] = queue.apply(priority_score, axis=1)
    queue["priority_reason"] = queue.apply(priority_reason, axis=1)
    queue["review_corrected_label"] = ""
    queue["review_exclude"] = ""
    queue["review_problem"] = ""
    queue["review_notes"] = ""

    priority_mask = (
        queue["diagnosis"].isin([2, 3, 4])
        | (~queue["raw_acceptable"].map(yes_no))
        | (queue["split"].isin(["train", "val"]) & queue["diagnosis"].isin([1]))
    )
    queue = queue[priority_mask].copy()
    queue = queue.sort_values(["priority_score", "split", "diagnosis"], ascending=[False, True, False])
    if max_rows > 0:
        queue = queue.head(max_rows)

    columns = [
        "priority_score",
        "priority_reason",
        "split",
        "id_code",
        "diagnosis",
        "image_path",
        "original_image_path",
        "raw_quality_score",
        "raw_sharpness",
        "raw_contrast",
        "raw_brightness",
        "raw_retina_coverage",
        "raw_acceptable",
        "raw_warnings",
        "review_corrected_label",
        "review_exclude",
        "review_problem",
        "review_notes",
    ]
    return queue[[col for col in columns if col in queue.columns]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea cola de revision manual para depurar el dataset original.")
    parser.add_argument("--max-rows", type=int, default=300, help="0 conserva todos los candidatos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queue = build_queue(args.max_rows)
    output = RESULTS_DIR / "dataset_review_queue.csv"
    queue.to_csv(output, index=False)
    summary = {
        "rows": int(len(queue)),
        "by_split": queue["split"].value_counts().sort_index().astype(int).to_dict(),
        "by_diagnosis": queue["diagnosis"].value_counts().sort_index().astype(int).to_dict(),
        "output": str(output),
    }
    print(summary)


if __name__ == "__main__":
    main()
