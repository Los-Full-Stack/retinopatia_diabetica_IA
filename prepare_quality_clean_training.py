import pandas as pd

from config.config import RESULTS_DIR, SPLITS_DIR
from src.utils import save_json


def main() -> None:
    quality_path = SPLITS_DIR / "train_with_external_raw_300_quality.csv"
    if not quality_path.exists():
        raise FileNotFoundError("Ejecuta primero: python prepare_raw_training.py")

    df = pd.read_csv(quality_path)
    required = {"processed_image_path", "raw_quality_score", "raw_sharpness", "diagnosis"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    clean = df[(df["raw_quality_score"] >= 0.80) & (df["raw_sharpness"] >= 12.0)].copy()
    clean["raw_image_path"] = clean["image_path"]
    clean["image_path"] = clean["processed_image_path"]
    clean["quality_clean_rule"] = "raw_quality_score>=0.80_and_raw_sharpness>=12"

    output = SPLITS_DIR / "train_with_external_quality_clean_processed_300_split.csv"
    clean.to_csv(output, index=False)

    report = {
        "source": str(quality_path),
        "output": str(output),
        "rule": "raw_quality_score >= 0.80 and raw_sharpness >= 12.0",
        "source_rows": int(len(df)),
        "kept_rows": int(len(clean)),
        "removed_rows": int(len(df) - len(clean)),
        "kept_distribution": {str(k): int(v) for k, v in clean["diagnosis"].value_counts().sort_index().items()},
        "removed_distribution": {
            str(k): int(v) for k, v in df.loc[~df.index.isin(clean.index), "diagnosis"].value_counts().sort_index().items()
        },
        "kept_external_rows": int(clean.get("is_external", False).astype(bool).sum()),
        "kept_aptos_rows": int((~clean.get("is_external", False).astype(bool)).sum()),
    }
    save_json(report, RESULTS_DIR / "quality_clean_training_report.json")
    print(report)


if __name__ == "__main__":
    main()
