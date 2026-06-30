import base64
import html
from pathlib import Path

import pandas as pd
from PIL import Image

from config.config import CLASS_NAMES, RESULTS_DIR


def image_data_uri(path: str, max_size: int = 220) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        tmp = RESULTS_DIR / "_tmp_dataset_review_thumb.jpg"
        image.save(tmp, format="JPEG", quality=85)
        data = base64.b64encode(tmp.read_bytes()).decode("ascii")
        tmp.unlink(missing_ok=True)
    return f"data:image/jpeg;base64,{data}"


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_row(row: pd.Series) -> str:
    diagnosis = int(row["diagnosis"])
    image_src = image_data_uri(str(row["image_path"]))
    raw_src = image_data_uri(str(row.get("original_image_path", row["image_path"])))
    return f"""
    <tr>
      <td>{fmt(row["priority_score"], 2)}<br><span>{html.escape(str(row["priority_reason"]))}</span></td>
      <td>{html.escape(str(row["split"]))}</td>
      <td>{html.escape(str(row["id_code"]))}</td>
      <td>{diagnosis} - {html.escape(CLASS_NAMES[diagnosis])}</td>
      <td><img src="{image_src}" alt="procesada"></td>
      <td><img src="{raw_src}" alt="original"></td>
      <td>
        calidad {fmt(row.get("raw_quality_score"))}<br>
        sharp {fmt(row.get("raw_sharpness"), 1)}<br>
        contraste {fmt(row.get("raw_contrast"), 1)}<br>
        aceptable {html.escape(str(row.get("raw_acceptable", "")))}<br>
        <span>{html.escape(str(row.get("raw_warnings", "")))}</span>
      </td>
      <td>
        corrected_label: 0/1/2/3/4<br>
        exclude: 1 si se excluye<br>
        problem: etiqueta|calidad|otro
      </td>
    </tr>
    """


def main() -> None:
    queue_path = RESULTS_DIR / "dataset_review_queue.csv"
    if not queue_path.exists():
        raise FileNotFoundError("Ejecuta primero: python create_dataset_review_queue.py")
    df = pd.read_csv(queue_path)
    rows = "\n".join(render_row(row) for _, row in df.iterrows())
    summary = (
        df.groupby(["split", "diagnosis"]).size().reset_index(name="count").to_html(index=False, escape=True)
    )
    doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>RetinaAI - Cola de revision del dataset</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ max-width: 980px; line-height: 1.45; color: #374151; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d9dee3; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef2f7; text-align: left; position: sticky; top: 0; z-index: 1; }}
    img {{ max-width: 220px; max-height: 220px; }}
    span {{ color: #64748b; font-size: 12px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Cola de revision del dataset original</h1>
  <p>
    Usa este HTML para mirar las imagenes. Las decisiones se escriben en
    <code>results/dataset_review_queue.csv</code>: coloca una etiqueta en
    <code>review_corrected_label</code> si la etiqueta original esta mal, o
    <code>1</code> en <code>review_exclude</code> si la imagen debe salir del dataset.
  </p>
  <h2>Resumen</h2>
  {summary}
  <h2>Casos</h2>
  <table>
    <tr>
      <th>Prioridad</th>
      <th>Split</th>
      <th>ID</th>
      <th>Etiqueta actual</th>
      <th>Procesada</th>
      <th>Original</th>
      <th>Calidad</th>
      <th>Como marcar</th>
    </tr>
    {rows}
  </table>
</body>
</html>"""
    output = RESULTS_DIR / "dataset_review_queue.html"
    output.write_text(doc, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
