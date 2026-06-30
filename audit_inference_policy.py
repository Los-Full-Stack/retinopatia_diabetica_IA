import json

import pandas as pd
from tqdm import tqdm

from pathlib import Path

from config.config import EXTRACT_DIR, IMAGE_SIZE, RESULTS_DIR, SPLITS_DIR, USE_RAW_SPLITS, USE_PROCESSED_IMAGES
from src.inference import RetinaPredictor
from src.utils import save_json


def original_image_path(id_code: str, fallback: str) -> str:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = EXTRACT_DIR / f"{id_code}{suffix}"
        if candidate.exists():
            return str(candidate)
    return fallback


def main():
    if USE_RAW_SPLITS:
        test_path = SPLITS_DIR / f"test_raw_{IMAGE_SIZE}_split.csv"
    else:
        test_path = SPLITS_DIR / (f"test_processed_{IMAGE_SIZE}_split.csv" if USE_PROCESSED_IMAGES else "test_split.csv")
    if not test_path.exists():
        raise FileNotFoundError("No existe split de test. Ejecuta prepare_data.py/evaluate.py.")
    df = pd.read_csv(test_path)
    predictor = RetinaPredictor()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="policy"):
        inference_path = original_image_path(str(row["id_code"]), row["image_path"])
        result = predictor.predict(inference_path)
        true_class = int(row["diagnosis"])
        accepted = bool(result["prediction_accepted"])
        correct = int(result["class_id"]) == true_class
        rows.append(
            {
                "id_code": row["id_code"],
                "image_path": row["image_path"],
                "inference_image_path": inference_path,
                "used_original_image": Path(inference_path) != Path(row["image_path"]),
                "true_class_id": true_class,
                "predicted_class_id": int(result["class_id"]),
                "confidence": float(result["confidence"]),
                "second_class_id": int(result["second_class_id"]),
                "second_confidence": float(result["second_confidence"]),
                "top2_margin": float(result["top2_margin"]),
                "required_confidence": float(result["required_confidence"]),
                "quality_score": float(result["image_quality"]["quality_score"]),
                "quality_acceptable": bool(result["image_quality"]["acceptable"]),
                "policy_decision": result.get("policy_decision", "accept" if accepted else "reject"),
                "policy_reasons": " | ".join(result.get("policy_reasons", [])),
                "prediction_accepted": accepted,
                "clinical_action": result["clinical_action"],
                "correct": correct,
                "abs_grade_error": abs(int(result["class_id"]) - true_class),
                "warnings": " | ".join(result["warnings"]),
            }
        )
    out_df = pd.DataFrame(rows)
    accepted_df = out_df[out_df["prediction_accepted"]]
    rejected_df = out_df[~out_df["prediction_accepted"]]
    decision_summary = {}
    for decision, group in out_df.groupby("policy_decision"):
        decision_summary[str(decision)] = {
            "count": int(len(group)),
            "coverage": float(len(group) / max(len(out_df), 1)),
            "accuracy_if_forced": float(group["correct"].mean()) if len(group) else None,
            "mean_abs_grade_error_if_forced": float(group["abs_grade_error"].mean()) if len(group) else None,
            "large_error_rate_if_forced": float((group["abs_grade_error"] >= 2).mean()) if len(group) else None,
        }
    report = {
        "total": int(len(out_df)),
        "accepted": int(len(accepted_df)),
        "rejected": int(len(rejected_df)),
        "coverage": float(len(accepted_df) / max(len(out_df), 1)),
        "accepted_accuracy": float(accepted_df["correct"].mean()) if len(accepted_df) else None,
        "accepted_mean_abs_grade_error": float(accepted_df["abs_grade_error"].mean()) if len(accepted_df) else None,
        "accepted_large_error_rate_grade_distance_2_or_more": float((accepted_df["abs_grade_error"] >= 2).mean()) if len(accepted_df) else None,
        "rejected_accuracy_if_forced": float(rejected_df["correct"].mean()) if len(rejected_df) else None,
        "rejected_large_error_rate_if_forced": float((rejected_df["abs_grade_error"] >= 2).mean()) if len(rejected_df) else None,
        "accepted_distribution": accepted_df["predicted_class_id"].value_counts().sort_index().to_dict(),
        "rejected_distribution": rejected_df["predicted_class_id"].value_counts().sort_index().to_dict(),
        "policy_decision_summary": decision_summary,
    }
    out_csv = RESULTS_DIR / "inference_policy_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    report["predictions_csv"] = str(out_csv)
    save_json(report, RESULTS_DIR / "inference_policy_audit.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
