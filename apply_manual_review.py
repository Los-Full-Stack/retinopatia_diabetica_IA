import argparse
from pathlib import Path

import pandas as pd

from config.config import RESULTS_DIR, SPLITS_DIR
from src.utils import save_json


SPLIT_FILES = {
    "train": SPLITS_DIR / "train_processed_300_split.csv",
    "val": SPLITS_DIR / "val_processed_300_split.csv",
    "test": SPLITS_DIR / "test_processed_300_split.csv",
}


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí", "exclude", "excluir"}


def corrected_label(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    label = int(float(str(value).strip()))
    if label not in {0, 1, 2, 3, 4}:
        raise ValueError(f"Etiqueta corregida fuera de rango: {value}")
    return label


def load_review(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    review = pd.read_csv(path)
    required = {"split", "id_code", "review_corrected_label", "review_exclude"}
    missing = required.difference(review.columns)
    if missing:
        raise ValueError(f"Faltan columnas en revision: {sorted(missing)}")
    review = review.copy()
    review["id_code"] = review["id_code"].astype(str)
    review["split"] = review["split"].astype(str)
    review["exclude_flag"] = review["review_exclude"].map(truthy)
    review["corrected_label_value"] = review["review_corrected_label"].map(corrected_label)
    actionable = review[review["exclude_flag"] | review["corrected_label_value"].notna()].copy()
    return actionable


def apply_to_split(split: str, review: pd.DataFrame, output_dir: Path) -> dict:
    source = SPLIT_FILES[split]
    df = pd.read_csv(source)
    df["id_code"] = df["id_code"].astype(str)
    split_review = review[review["split"] == split].copy()
    if split_review.empty:
        output = output_dir / source.name.replace("_split.csv", "_manual_review_split.csv")
        df.to_csv(output, index=False)
        return {
            "source": str(source),
            "output": str(output),
            "source_rows": int(len(df)),
            "output_rows": int(len(df)),
            "excluded_rows": 0,
            "corrected_rows": 0,
            "distribution": {str(k): int(v) for k, v in df["diagnosis"].value_counts().sort_index().items()},
        }

    duplicated = split_review["id_code"].duplicated(keep=False)
    if duplicated.any():
        duplicates = split_review.loc[duplicated, "id_code"].tolist()
        raise ValueError(f"IDs duplicados en revision para {split}: {duplicates[:10]}")

    review_by_id = split_review.set_index("id_code")
    unknown = sorted(set(review_by_id.index) - set(df["id_code"]))
    if unknown:
        raise ValueError(f"IDs de {split} no existen en split original: {unknown[:10]}")

    df["manual_review_original_diagnosis"] = df["diagnosis"]
    df["manual_review_action"] = ""
    corrected_rows = 0
    excluded_ids = set()
    for idx, row in df.iterrows():
        id_code = row["id_code"]
        if id_code not in review_by_id.index:
            continue
        review_row = review_by_id.loc[id_code]
        if bool(review_row["exclude_flag"]):
            excluded_ids.add(id_code)
            df.at[idx, "manual_review_action"] = "excluded"
            continue
        label = review_row["corrected_label_value"]
        if pd.notna(label):
            label = int(label)
            if int(row["diagnosis"]) != label:
                corrected_rows += 1
                df.at[idx, "diagnosis"] = label
                df.at[idx, "manual_review_action"] = "corrected_label"

    curated = df[~df["id_code"].isin(excluded_ids)].copy()
    output = output_dir / source.name.replace("_split.csv", "_manual_review_split.csv")
    curated.to_csv(output, index=False)
    return {
        "source": str(source),
        "output": str(output),
        "source_rows": int(len(df)),
        "output_rows": int(len(curated)),
        "excluded_rows": int(len(excluded_ids)),
        "corrected_rows": int(corrected_rows),
        "distribution": {str(k): int(v) for k, v in curated["diagnosis"].value_counts().sort_index().items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica revision manual y genera splits curados.")
    parser.add_argument("--review-csv", type=Path, default=RESULTS_DIR / "dataset_review_queue.csv")
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = load_review(args.review_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "review_csv": str(args.review_csv),
        "actionable_rows": int(len(review)),
        "splits": {},
    }
    for split in SPLIT_FILES:
        report["splits"][split] = apply_to_split(split, review, args.output_dir)
    report_path = RESULTS_DIR / "manual_review_apply_report.json"
    save_json(report, report_path)
    print(report)


if __name__ == "__main__":
    main()
