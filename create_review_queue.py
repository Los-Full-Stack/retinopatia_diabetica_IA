import pandas as pd

from config.config import RESULTS_DIR


def priority_reason(row) -> str:
    reasons = []
    if row["abs_grade_error"] >= 2:
        reasons.append("error_grande")
    if row["confidence"] >= 0.65 and row["abs_grade_error"] > 0:
        reasons.append("error_alta_confianza")
    if row.get("raw_acceptable") is False:
        reasons.append("calidad_original_baja")
    if int(row["true_class_id"]) in {2, 3, 4}:
        reasons.append("grado_clinico_prioritario")
    return "|".join(reasons)


def main() -> None:
    path = RESULTS_DIR / "predictions_with_quality.csv"
    if not path.exists():
        raise FileNotFoundError("Ejecuta primero: python analyze_errors_quality.py")
    df = pd.read_csv(path)
    df["abs_grade_error"] = (df["predicted_class_id"] - df["true_class_id"]).abs()
    queue = df[
        (df["abs_grade_error"] >= 2)
        | ((df["confidence"] >= 0.65) & (df["abs_grade_error"] > 0))
        | ((df["true_class_id"].isin([2, 3, 4])) & (df["confidence"] >= 0.55) & (df["abs_grade_error"] > 0))
    ].copy()
    queue["priority_score"] = (
        queue["abs_grade_error"] * 10
        + queue["confidence"] * 5
        + queue["true_class_id"].isin([2, 3, 4]).astype(int) * 3
        + (1 - queue.get("raw_acceptable", True).astype(float)) * 2
    )
    queue["priority_reason"] = queue.apply(priority_reason, axis=1)
    queue["review_corrected_label"] = ""
    queue["review_exclude"] = ""
    queue["review_problem"] = ""
    queue["review_notes"] = ""
    columns = [
        "priority_score",
        "priority_reason",
        "id_code",
        "image_path",
        "raw_path",
        "true_class_id",
        "predicted_class_id",
        "confidence",
        "abs_grade_error",
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
        "probability_class_3",
        "probability_class_4",
        "raw_quality_score",
        "raw_sharpness",
        "raw_contrast",
        "raw_acceptable",
        "raw_warnings",
        "review_corrected_label",
        "review_exclude",
        "review_problem",
        "review_notes",
    ]
    existing_columns = [col for col in columns if col in queue.columns]
    queue = queue.sort_values("priority_score", ascending=False)[existing_columns]
    output = RESULTS_DIR / "manual_review_queue.csv"
    queue.to_csv(output, index=False)
    print(f"{output} rows={len(queue)}")


if __name__ == "__main__":
    main()
