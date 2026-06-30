import base64
import html
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config.config import CLASS_NAMES, EXTRACT_DIR, RESULTS_DIR, SPLITS_DIR
from src.quality import assess_retina_image_quality


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def image_data_uri(path: Path, max_size=240) -> str:
    if not path.exists():
        return ""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        tmp = RESULTS_DIR / "_tmp_quality_thumb.jpg"
        img.save(tmp, format="JPEG", quality=86)
        data = base64.b64encode(tmp.read_bytes()).decode("ascii")
        tmp.unlink(missing_ok=True)
    return f"data:image/jpeg;base64,{data}"


def original_image_path(row: pd.Series) -> Path:
    id_code = str(row["id_code"])
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = EXTRACT_DIR / f"{id_code}{suffix}"
        if candidate.exists():
            return candidate
    return Path(row["image_path"])


def add_quality(df: pd.DataFrame, source: str = "image_path", prefix: str = "quality") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        path = original_image_path(row) if source == "original" else Path(row["image_path"])
        quality = {}
        try:
            with Image.open(path) as img:
                quality = assess_retina_image_quality(img).to_dict()
        except Exception as exc:
            quality = {
                "brightness": None,
                "contrast": None,
                "sharpness": None,
                "retina_coverage": None,
                "quality_score": None,
                "acceptable": False,
                "warnings": [f"No se pudo leer: {exc}"],
            }
        data = row.to_dict()
        data[f"{prefix}_path"] = str(path)
        data.update({f"{prefix}_{key}": value for key, value in quality.items()})
        rows.append(data)
    return pd.DataFrame(rows)


def make_contact_sheet(df: pd.DataFrame, output_path: Path, title: str, limit=48) -> None:
    rows = df.head(limit).copy()
    cols = 4
    thumb_w, thumb_h = 270, 310
    sheet_w = cols * thumb_w
    sheet_h = max(thumb_h, ((len(rows) + cols - 1) // cols) * thumb_h + 42)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)

    for idx, (_, row) in enumerate(rows.iterrows()):
        x = (idx % cols) * thumb_w
        y = 42 + (idx // cols) * thumb_h
        path = Path(row["image_path"])
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((250, 210))
                sheet.paste(img, (x + 10, y + 10))
        except Exception:
            draw.text((x + 10, y + 10), "No se pudo abrir", fill="red", font=font)
        warnings = row.get("raw_warnings", [])
        if isinstance(warnings, str):
            warning_text = warnings[:46]
        else:
            warning_text = ", ".join(warnings)[:46]
        label = (
            f"{row['id_code']}\n"
            f"real {row['true_class_id']} -> pred {row['predicted_class_id']} | conf {row['confidence']:.2f}\n"
            f"err {row['abs_grade_error']} | raw q {row.get('raw_quality_score', 0):.2f}\n"
            f"{warning_text}"
        )
        draw.multiline_text((x + 10, y + 226), label, fill="black", font=font, spacing=3)
    sheet.save(output_path, quality=92)


def table_html(df: pd.DataFrame, cols: list[str], max_rows=30) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in cols)
    rows = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def image_rows_html(df: pd.DataFrame, max_rows=50) -> str:
    rows = []
    for _, row in df.head(max_rows).iterrows():
        path = Path(row["image_path"])
        warnings = row.get("raw_warnings", [])
        if not isinstance(warnings, str):
            warnings = ", ".join(warnings)
        rows.append(
            f"""
            <tr>
              <td><img src="{image_data_uri(path)}" alt="{html.escape(str(row['id_code']))}"></td>
              <td>{html.escape(str(row['id_code']))}</td>
              <td>{row['true_class_id']} - {html.escape(CLASS_NAMES[int(row['true_class_id'])])}</td>
              <td>{row['predicted_class_id']} - {html.escape(CLASS_NAMES[int(row['predicted_class_id'])])}</td>
              <td>{row['confidence']:.3f}</td>
              <td>{row['abs_grade_error']}</td>
              <td>{row['raw_quality_score']:.3f}</td>
              <td>{html.escape(str(row.get('raw_warnings', '')))}</td>
            </tr>
            """
        )
    return "".join(rows)


def main() -> None:
    predictions_path = RESULTS_DIR / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("Ejecuta primero evaluate.py para crear results/predictions.csv")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(predictions_path)
    pred["abs_grade_error"] = (pred["predicted_class_id"] - pred["true_class_id"]).abs()
    pred["is_correct"] = pred["abs_grade_error"] == 0
    pred["is_large_error"] = pred["abs_grade_error"] >= 2
    pred_q = add_quality(pred, source="image_path", prefix="processed")
    pred_q = add_quality(pred_q, source="original", prefix="raw")

    pred_q.to_csv(RESULTS_DIR / "predictions_with_quality.csv", index=False)
    large = pred_q[pred_q["is_large_error"]].sort_values(["abs_grade_error", "confidence"], ascending=[False, False])
    large.to_csv(RESULTS_DIR / "large_errors_with_quality.csv", index=False)

    quality_summary = (
        pred_q.groupby(["true_class_id", "is_correct"], dropna=False)
        .agg(
            count=("id_code", "count"),
            mean_raw_quality=("raw_quality_score", "mean"),
            mean_raw_sharpness=("raw_sharpness", "mean"),
            mean_raw_contrast=("raw_contrast", "mean"),
            mean_raw_brightness=("raw_brightness", "mean"),
            raw_acceptable_rate=("raw_acceptable", "mean"),
            mean_processed_quality=("processed_quality_score", "mean"),
        )
        .reset_index()
    )
    quality_summary.to_csv(RESULTS_DIR / "quality_by_grade_and_correctness.csv", index=False)

    confusion_summary = (
        pred_q.groupby(["true_class_id", "predicted_class_id"])
        .agg(
            count=("id_code", "count"),
            mean_confidence=("confidence", "mean"),
            mean_raw_quality=("raw_quality_score", "mean"),
            raw_acceptable_rate=("raw_acceptable", "mean"),
            mean_processed_quality=("processed_quality_score", "mean"),
        )
        .reset_index()
        .sort_values(["true_class_id", "count"], ascending=[True, False])
    )
    confusion_summary.to_csv(RESULTS_DIR / "confusion_quality_summary.csv", index=False)

    train_quality_path = SPLITS_DIR / "train_with_external_processed_300_split.csv"
    train_quality_summary = pd.DataFrame()
    if train_quality_path.exists():
        train_df = pd.read_csv(train_quality_path)
        train_q = add_quality(train_df.rename(columns={"diagnosis": "true_class_id"}), source="image_path", prefix="processed")
        train_quality_summary = (
            train_q.groupby("true_class_id")
            .agg(
                count=("id_code", "count"),
                mean_processed_quality=("processed_quality_score", "mean"),
                mean_processed_sharpness=("processed_sharpness", "mean"),
                mean_processed_contrast=("processed_contrast", "mean"),
                processed_acceptable_rate=("processed_acceptable", "mean"),
            )
            .reset_index()
        )
        train_quality_summary.to_csv(RESULTS_DIR / "train_quality_by_grade.csv", index=False)

    groups = {
        "grade4_pred_0_or_1": large[(large["true_class_id"] == 4) & (large["predicted_class_id"].isin([0, 1]))],
        "grade3_pred_0_or_1": large[(large["true_class_id"] == 3) & (large["predicted_class_id"].isin([0, 1]))],
        "grade2_pred_0": large[(large["true_class_id"] == 2) & (large["predicted_class_id"] == 0)],
        "grade2_pred_4": large[(large["true_class_id"] == 2) & (large["predicted_class_id"] == 4)],
        "high_confidence_large_errors": large[large["confidence"] >= 0.65],
    }
    contact_links = []
    for name, group in groups.items():
        if group.empty:
            continue
        path = RESULTS_DIR / f"{name}_contact_sheet.jpg"
        make_contact_sheet(group, path, name)
        contact_links.append((name, path, len(group)))

    top_wrong = pred_q[pred_q["abs_grade_error"] > 0].sort_values("confidence", ascending=False)
    top_wrong.to_csv(RESULTS_DIR / "high_confidence_wrong_predictions.csv", index=False)

    summary = {
        "test_rows": int(len(pred_q)),
        "correct": int(pred_q["is_correct"].sum()),
        "large_errors": int(pred_q["is_large_error"].sum()),
        "large_error_rate": float(pred_q["is_large_error"].mean()),
        "high_confidence_wrong_ge_065": int(((pred_q["abs_grade_error"] > 0) & (pred_q["confidence"] >= 0.65)).sum()),
        "raw_quality_acceptable_rate": float(pred_q["raw_acceptable"].mean()),
        "large_error_raw_quality_acceptable_rate": float(large["raw_acceptable"].mean()) if len(large) else 0.0,
        "outputs": {
            "predictions_with_quality": str(RESULTS_DIR / "predictions_with_quality.csv"),
            "large_errors_with_quality": str(RESULTS_DIR / "large_errors_with_quality.csv"),
            "quality_by_grade_and_correctness": str(RESULTS_DIR / "quality_by_grade_and_correctness.csv"),
            "confusion_quality_summary": str(RESULTS_DIR / "confusion_quality_summary.csv"),
            "high_confidence_wrong_predictions": str(RESULTS_DIR / "high_confidence_wrong_predictions.csv"),
        },
    }

    cards = "".join(
        f"<div class='card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in [
            ("Test", summary["test_rows"]),
            ("Aciertos", summary["correct"]),
            ("Errores grandes", f"{summary['large_errors']} ({pct(summary['large_error_rate'])})"),
            ("Errores alta confianza", summary["high_confidence_wrong_ge_065"]),
            ("Calidad original aceptable", pct(summary["raw_quality_acceptable_rate"])),
            ("Original aceptable en errores grandes", pct(summary["large_error_raw_quality_acceptable_rate"])),
        ]
    )
    links = "".join(
        f"<li><a href='{path.name}'>{html.escape(name)}</a> - {count} casos</li>"
        for name, path, count in contact_links
    )
    html_doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>RetinaAI - Analisis de errores y calidad</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #d9dee3; border-radius: 8px; padding: 14px; }}
    .card span {{ display: block; color: #475569; font-size: 13px; }}
    .card strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d9dee3; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef2f7; text-align: left; }}
    img {{ max-width: 240px; max-height: 240px; }}
    .note {{ color: #475569; max-width: 980px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Analisis de errores y calidad</h1>
  <p class="note">Este reporte separa errores grandes, errores con alta confianza y calidad de imagen. Sirve para decidir si conviene corregir etiquetas, filtrar imagenes o cambiar la politica de rechazo antes de reentrenar.</p>
  <div class="grid">{cards}</div>

  <h2>Hojas visuales para revision</h2>
  <ul>{links}</ul>

  <h2>Calidad por grado y acierto</h2>
  {table_html(quality_summary, ["true_class_id", "is_correct", "count", "mean_raw_quality", "mean_raw_sharpness", "mean_raw_contrast", "raw_acceptable_rate", "mean_processed_quality"], 50)}

  <h2>Confusiones con calidad</h2>
  {table_html(confusion_summary, ["true_class_id", "predicted_class_id", "count", "mean_confidence", "mean_raw_quality", "raw_acceptable_rate", "mean_processed_quality"], 80)}

  <h2>Calidad de entrenamiento por grado</h2>
  {table_html(train_quality_summary, ["true_class_id", "count", "mean_processed_quality", "mean_processed_sharpness", "mean_processed_contrast", "processed_acceptable_rate"], 20)}

  <h2>Errores grandes principales</h2>
  <table>
    <tr><th>Imagen</th><th>ID</th><th>Real</th><th>Prediccion</th><th>Confianza</th><th>Error</th><th>Calidad</th><th>Alertas</th></tr>
    {image_rows_html(large)}
  </table>
</body>
</html>"""
    report_path = RESULTS_DIR / "error_quality_report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    pd.Series(summary).to_json(RESULTS_DIR / "error_quality_summary.json", force_ascii=False, indent=2)
    print(report_path)


if __name__ == "__main__":
    main()
