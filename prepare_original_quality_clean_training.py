import pandas as pd
from PIL import Image

from config.config import RESULTS_DIR, SPLITS_DIR
from src.quality import assess_retina_image_quality
from src.utils import save_json


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


def main() -> None:
    source_path = SPLITS_DIR / "train_processed_300_split.csv"
    if not source_path.exists():
        raise FileNotFoundError("Ejecuta primero: python prepare_processed.py")

    raw_quality_cache = SPLITS_DIR / "train_with_external_raw_300_quality.csv"
    if raw_quality_cache.exists():
        quality_df = pd.read_csv(raw_quality_cache)
        quality_df = quality_df[~quality_df.get("is_external", False).astype(bool)].copy()
        quality_df["image_path"] = quality_df["processed_image_path"]
        quality_df["raw_image_path"] = quality_df["original_image_path"]
        quality_cache = raw_quality_cache
    else:
        df = pd.read_csv(source_path)
        rows = []
        for _, row in df.iterrows():
            quality_path = row.get("original_image_path") or row["image_path"]
            quality = assess_path(str(quality_path))
            data = row.to_dict()
            data.update({f"raw_{key}": value for key, value in quality.items()})
            rows.append(data)
        quality_df = pd.DataFrame(rows)
        quality_cache = SPLITS_DIR / "train_original_raw_300_quality.csv"
        quality_df.to_csv(quality_cache, index=False)

    clean = quality_df[
        (quality_df["raw_quality_score"] >= 0.60)
        & (quality_df["raw_sharpness"] >= 8.0)
        & (quality_df["raw_retina_coverage"] >= 0.14)
    ].copy()
    clean["quality_clean_rule"] = "raw_quality_score>=0.60_and_raw_sharpness>=8_and_raw_coverage>=0.14"

    output = SPLITS_DIR / "train_original_quality_clean_processed_300_split.csv"
    clean.to_csv(output, index=False)

    removed = quality_df.loc[~quality_df.index.isin(clean.index)]
    report = {
        "source": str(source_path),
        "quality_cache": str(quality_cache),
        "output": str(output),
        "rule": clean["quality_clean_rule"].iloc[0] if len(clean) else "",
        "source_rows": int(len(quality_df)),
        "kept_rows": int(len(clean)),
        "removed_rows": int(len(removed)),
        "kept_distribution": {
            str(k): int(v) for k, v in clean["diagnosis"].value_counts().sort_index().items()
        },
        "removed_distribution": {
            str(k): int(v) for k, v in removed["diagnosis"].value_counts().sort_index().items()
        },
        "mean_raw_quality_by_grade": {
            str(k): float(v)
            for k, v in quality_df.groupby("diagnosis")["raw_quality_score"].mean().sort_index().items()
        },
        "raw_acceptable_rate_by_grade": {
            str(k): float(v)
            for k, v in quality_df.groupby("diagnosis")["raw_acceptable"].mean().sort_index().items()
        },
    }
    save_json(report, RESULTS_DIR / "original_quality_clean_training_report.json")
    print(report)


if __name__ == "__main__":
    main()
