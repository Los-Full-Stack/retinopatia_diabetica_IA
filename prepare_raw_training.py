from pathlib import Path

import pandas as pd

from config.config import EXTRACT_DIR, RESULTS_DIR, SPLITS_DIR
from src.quality import assess_retina_image_quality
from PIL import Image
from src.utils import save_json


def resolve_aptos_original_path(id_code: str, fallback: str) -> str:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = EXTRACT_DIR / f"{id_code}{suffix}"
        if path.exists():
            return str(path)
    return fallback


def assess_path(path: str) -> dict:
    try:
        with Image.open(path) as image:
            return assess_retina_image_quality(image).to_dict()
    except Exception as exc:
        return {
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "retina_coverage": None,
            "quality_score": None,
            "acceptable": False,
            "warnings": [f"No se pudo leer: {exc}"],
        }


def build_raw_split(source_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    if "original_image_path" in df.columns:
        raw_paths = df["original_image_path"].fillna(df["image_path"]).astype(str)
    else:
        raw_paths = df.apply(lambda row: resolve_aptos_original_path(str(row["id_code"]), str(row["image_path"])), axis=1)
    out = df.copy()
    out["processed_image_path"] = out["image_path"]
    out["image_path"] = raw_paths
    out["uses_raw_image"] = True
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def add_quality_cache(df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        quality = assess_path(str(row["image_path"]))
        data = row.to_dict()
        data.update({f"raw_{key}": value for key, value in quality.items()})
        rows.append(data)
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    sources = {
        "train": SPLITS_DIR / "train_with_external_processed_300_split.csv",
        "val": SPLITS_DIR / "val_processed_300_split.csv",
        "test": SPLITS_DIR / "test_processed_300_split.csv",
    }
    outputs = {
        "train": SPLITS_DIR / "train_with_external_raw_300_split.csv",
        "val": SPLITS_DIR / "val_raw_300_split.csv",
        "test": SPLITS_DIR / "test_raw_300_split.csv",
    }
    quality_outputs = {
        "train": SPLITS_DIR / "train_with_external_raw_300_quality.csv",
        "val": SPLITS_DIR / "val_raw_300_quality.csv",
        "test": SPLITS_DIR / "test_raw_300_quality.csv",
    }

    report = {}
    for split, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"No existe {source}")
        raw_df = build_raw_split(source, outputs[split])
        quality_df = add_quality_cache(raw_df, quality_outputs[split])
        report[split] = {
            "source": str(source),
            "raw_csv": str(outputs[split]),
            "quality_csv": str(quality_outputs[split]),
            "rows": int(len(raw_df)),
            "class_distribution": {
                str(k): int(v) for k, v in raw_df["diagnosis"].value_counts().sort_index().items()
            },
            "raw_quality_acceptable_rate": float(quality_df["raw_acceptable"].mean()),
            "raw_quality_mean": float(quality_df["raw_quality_score"].mean()),
            "raw_sharpness_mean": float(quality_df["raw_sharpness"].mean()),
        }
    save_json(report, RESULTS_DIR / "raw_training_report.json")
    print(report)


if __name__ == "__main__":
    main()
