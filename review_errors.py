import base64
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config.config import CLASS_NAMES, RESULTS_DIR


def image_to_data_uri(path: Path, max_size=220) -> str:
    if not path.exists():
        return ""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = Path(RESULTS_DIR / "_tmp_error_thumb.jpg")
        img.save(buffer, format="JPEG", quality=88)
        data = base64.b64encode(buffer.read_bytes()).decode("ascii")
        buffer.unlink(missing_ok=True)
    return f"data:image/jpeg;base64,{data}"


def make_contact_sheet(df, output_path: Path, limit=40):
    rows = df.head(limit).copy()
    thumb_w, thumb_h = 220, 250
    cols = 4
    sheet_w = cols * thumb_w
    sheet_h = ((len(rows) + cols - 1) // cols) * thumb_h
    sheet = Image.new("RGB", (sheet_w, max(sheet_h, thumb_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (_, row) in enumerate(rows.iterrows()):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        path = Path(row["image_path"])
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((200, 180))
                sheet.paste(img, (x + 10, y + 10))
        except Exception:
            draw.text((x + 10, y + 10), "No se pudo abrir", fill="red", font=font)
        label = (
            f"real {row['true_class_id']} -> pred {row['predicted_class_id']}\n"
            f"conf {row['confidence']:.2f} err {row['abs_grade_error']}"
        )
        draw.text((x + 10, y + 195), label, fill="black", font=font)
    sheet.save(output_path)


def main():
    predictions_path = RESULTS_DIR / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("No existe results/predictions.csv. Ejecuta evaluate.py primero.")
    df = pd.read_csv(predictions_path)
    df["abs_grade_error"] = (df["predicted_class_id"] - df["true_class_id"]).abs()
    df = df.sort_values(["abs_grade_error", "confidence"], ascending=[False, False])
    large = df[df["abs_grade_error"] >= 2].copy()
    large_csv = RESULTS_DIR / "large_grade_errors_review.csv"
    large.to_csv(large_csv, index=False)

    contact_sheet = RESULTS_DIR / "large_grade_errors_contact_sheet.jpg"
    make_contact_sheet(large, contact_sheet)

    rows_html = []
    for _, row in large.head(80).iterrows():
        path = Path(row["image_path"])
        rows_html.append(
            f"""
            <tr>
              <td><img src="{image_to_data_uri(path)}" alt="{row['id_code']}"></td>
              <td>{row['id_code']}</td>
              <td>{row['true_class_id']} - {CLASS_NAMES[int(row['true_class_id'])]}</td>
              <td>{row['predicted_class_id']} - {CLASS_NAMES[int(row['predicted_class_id'])]}</td>
              <td>{row['confidence']:.3f}</td>
              <td>{row['abs_grade_error']}</td>
              <td>{path}</td>
            </tr>
            """
        )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>RetinaAI - Revisión de errores grandes</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d9dee3; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; text-align: left; }}
    img {{ max-width: 220px; max-height: 220px; }}
    .note {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Revisión de errores grandes</h1>
  <p class="note">Casos donde el modelo se equivocó por 2 o más grados. Estos son candidatos para revisar etiqueta, calidad de imagen o patrón clínico difícil.</p>
  <p>Total: {len(large)} | CSV: {large_csv} | Contact sheet: {contact_sheet}</p>
  <table>
    <tr><th>Imagen</th><th>ID</th><th>Real</th><th>Predicción</th><th>Confianza</th><th>Error</th><th>Ruta</th></tr>
    {''.join(rows_html)}
  </table>
</body>
</html>"""
    output = RESULTS_DIR / "large_grade_errors_review.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
