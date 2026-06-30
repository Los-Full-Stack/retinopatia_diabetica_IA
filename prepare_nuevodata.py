from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config.config import RESULTS_DIR, SPLITS_DIR
from src.quality import assess_retina_image_quality
from src.utils import save_json, setup_logging


NEW_DATA_DIR = Path("nuevodata")
NEW_TRAIN_CSV = NEW_DATA_DIR / "train.csv"
NEW_TEST_CSV = NEW_DATA_DIR / "test.csv"
NEW_TRAIN_IMAGES = NEW_DATA_DIR / "train_images_512" / "train_images_512"
NEW_TEST_IMAGES = NEW_DATA_DIR / "test_images_512" / "test_images_512"

MAX_NEW_GRADE_0 = 6000
TARGETED_MIX_LIMITS = {
    0: 500,
    1: 500,
    2: 1500,
    3: None,
    4: None,
}
RANDOM_STATE = 42


def add_paths(df: pd.DataFrame, image_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    missing = []
    for _, row in df.iterrows():
        image_id = str(row["id_code"])
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            missing.append({"id_code": image_id, "image_path": str(path)})
            continue
        data = row.to_dict()
        data["id_code"] = f"nuevodata_{image_id}"
        data["original_id_code"] = image_id
        data["image_path"] = str(path)
        data["original_image_path"] = str(path)
        data["source_dataset"] = "nuevodata_eyecaps_512"
        data["is_external"] = True
        data["is_new_external"] = True
        rows.append(data)
    return pd.DataFrame(rows), missing


def balanced_new_train(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for diagnosis, group in df.groupby("diagnosis", sort=True):
        limit = MAX_NEW_GRADE_0 if int(diagnosis) == 0 else len(group)
        parts.append(group.sample(n=min(len(group), limit), random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)


def combine_with_local(new_df: pd.DataFrame, local_csv: Path) -> pd.DataFrame:
    local = pd.read_csv(local_csv)
    local = local.copy()
    local["source_dataset"] = local.get("source_dataset", "aptos2019")
    local["is_external"] = False
    local["is_new_external"] = False

    common_columns = sorted(set(local.columns).union(new_df.columns))
    return pd.concat(
        [local.reindex(columns=common_columns), new_df.reindex(columns=common_columns)],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE)


def targeted_new_mix(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for diagnosis, group in df.groupby("diagnosis", sort=True):
        limit = TARGETED_MIX_LIMITS[int(diagnosis)]
        n = len(group) if limit is None else min(len(group), limit)
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)


def quality_sample(df: pd.DataFrame, per_class: int = 120) -> dict:
    summary = {}
    for diagnosis, group in df.groupby("diagnosis", sort=True):
        sample = group.sample(n=min(per_class, len(group)), random_state=RANDOM_STATE)
        rows = []
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"quality class {diagnosis}"):
            try:
                with Image.open(row["image_path"]) as image:
                    quality = assess_retina_image_quality(image).to_dict()
                rows.append(quality)
            except Exception as exc:
                rows.append(
                    {
                        "brightness": None,
                        "contrast": None,
                        "sharpness": None,
                        "retina_coverage": None,
                        "quality_score": None,
                        "acceptable": False,
                        "warnings": [str(exc)],
                    }
                )
        qdf = pd.DataFrame(rows)
        summary[str(int(diagnosis))] = {
            "sample_size": int(len(qdf)),
            "acceptable_rate": float(qdf["acceptable"].mean()),
            "brightness_mean": float(qdf["brightness"].mean()),
            "contrast_mean": float(qdf["contrast"].mean()),
            "sharpness_mean": float(qdf["sharpness"].mean()),
            "retina_coverage_mean": float(qdf["retina_coverage"].mean()),
            "quality_score_mean": float(qdf["quality_score"].mean()),
        }
    return summary


def distribution(df: pd.DataFrame) -> dict:
    return {str(k): int(v) for k, v in df["diagnosis"].value_counts().sort_index().items()}


def main() -> None:
    logger = setup_logging()
    if not NEW_TRAIN_CSV.exists() or not NEW_TEST_CSV.exists():
        raise FileNotFoundError("No existe nuevodata/train.csv o nuevodata/test.csv")

    train_raw = pd.read_csv(NEW_TRAIN_CSV)
    test_raw = pd.read_csv(NEW_TEST_CSV)
    required = {"id_code", "diagnosis"}
    for path, df in [(NEW_TRAIN_CSV, train_raw), (NEW_TEST_CSV, test_raw)]:
        missing_columns = required.difference(df.columns)
        if missing_columns:
            raise ValueError(f"{path} no tiene columnas requeridas: {sorted(missing_columns)}")

    train_df, missing_train = add_paths(train_raw, NEW_TRAIN_IMAGES)
    test_df, missing_test = add_paths(test_raw, NEW_TEST_IMAGES)
    balanced_train = balanced_new_train(train_df)
    targeted_train = targeted_new_mix(train_df)

    local_300 = SPLITS_DIR / "train_processed_300_split.csv"
    local_512 = SPLITS_DIR / "train_processed_512_split.csv"
    baseline_external_300 = SPLITS_DIR / "train_with_external_processed_300_split.csv"
    combined_300 = combine_with_local(balanced_train, local_300)
    combined_512 = combine_with_local(balanced_train, local_512) if local_512.exists() else None
    targeted_combined_300 = combine_with_local(targeted_train, baseline_external_300)

    outputs = {
        "nuevodata_train_512": SPLITS_DIR / "nuevodata_train_512.csv",
        "nuevodata_test_512": SPLITS_DIR / "nuevodata_test_512.csv",
        "nuevodata_train_balanced_512": SPLITS_DIR / "nuevodata_train_balanced_512.csv",
        "nuevodata_train_targeted_512": SPLITS_DIR / "nuevodata_train_targeted_512.csv",
        "train_with_nuevodata_balanced_300_split": SPLITS_DIR / "train_with_nuevodata_balanced_300_split.csv",
        "train_with_external_and_nuevodata_targeted_300_split": (
            SPLITS_DIR / "train_with_external_and_nuevodata_targeted_300_split.csv"
        ),
    }
    train_df.to_csv(outputs["nuevodata_train_512"], index=False)
    test_df.to_csv(outputs["nuevodata_test_512"], index=False)
    balanced_train.to_csv(outputs["nuevodata_train_balanced_512"], index=False)
    targeted_train.to_csv(outputs["nuevodata_train_targeted_512"], index=False)
    combined_300.to_csv(outputs["train_with_nuevodata_balanced_300_split"], index=False)
    targeted_combined_300.to_csv(outputs["train_with_external_and_nuevodata_targeted_300_split"], index=False)

    if combined_512 is not None:
        outputs["train_with_nuevodata_balanced_512_split"] = (
            SPLITS_DIR / "train_with_nuevodata_balanced_512_split.csv"
        )
        combined_512.to_csv(outputs["train_with_nuevodata_balanced_512_split"], index=False)

    report = {
        "source_dir": str(NEW_DATA_DIR),
        "max_new_grade_0": MAX_NEW_GRADE_0,
        "new_train_rows_raw": int(len(train_raw)),
        "new_test_rows_raw": int(len(test_raw)),
        "new_train_rows_valid": int(len(train_df)),
        "new_test_rows_valid": int(len(test_df)),
        "missing_train": missing_train,
        "missing_test": missing_test,
        "new_train_distribution": distribution(train_df),
        "new_test_distribution": distribution(test_df),
        "new_balanced_train_rows": int(len(balanced_train)),
        "new_balanced_train_distribution": distribution(balanced_train),
        "new_targeted_train_rows": int(len(targeted_train)),
        "new_targeted_train_distribution": distribution(targeted_train),
        "new_targeted_limits": {str(k): v for k, v in TARGETED_MIX_LIMITS.items()},
        "combined_300_rows": int(len(combined_300)),
        "combined_300_distribution": distribution(combined_300),
        "targeted_combined_300_rows": int(len(targeted_combined_300)),
        "targeted_combined_300_distribution": distribution(targeted_combined_300),
        "combined_512_rows": int(len(combined_512)) if combined_512 is not None else None,
        "combined_512_distribution": distribution(combined_512) if combined_512 is not None else None,
        "quality_sample": quality_sample(train_df),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    save_json(report, RESULTS_DIR / "nuevodata_report.json")
    logger.info("Nuevo dataset preparado: %s", report)
    print(report)


if __name__ == "__main__":
    main()
