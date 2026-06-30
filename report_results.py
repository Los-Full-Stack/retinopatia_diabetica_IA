import base64
import json
from pathlib import Path

from config.config import CLASS_NAMES, RESULTS_DIR


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value):
    if value is None:
        return "N/D"
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def format_int(value):
    if value is None:
        return "N/D"
    return f"{int(value):,}".replace(",", " ")


CHART_NOTES = {
    "Distribución de clases": "Muestra cuántos ejemplos hay por grado. Si los grados 3 y 4 tienen pocos casos, el modelo suele aprenderlos peor.",
    "Loss": "Mide el error. Lo ideal es que baje en entrenamiento y validación sin separarse demasiado; si entrenamiento baja y validación sube, hay sobreajuste.",
    "Accuracy": "Porcentaje total de aciertos. Es útil, pero puede verse alta aunque el modelo falle clases raras como 3 y 4.",
    "Balanced Accuracy": "Promedia el acierto de cada grado. Para este proyecto pesa más que la accuracy porque las clases están desbalanceadas.",
    "F1 Macro": "Resume precisión y recall dando el mismo peso a cada grado. Ayuda a ver si el modelo funciona en las 5 clases, no solo en la clase 0.",
    "Kappa": "Mide acuerdo ordinal: castiga más confundir grados lejanos. Es importante porque los grados 0-4 tienen orden clínico.",
    "Métricas por grado": "Compara precisión, recall y F1 por grado. Aquí revisamos especialmente los grados 2, 3 y 4.",
    "Matriz de confusión": "Filas son grados reales y columnas predicciones. La diagonal son aciertos; los errores cercanos a la diagonal son menos graves que saltos grandes.",
    "Matriz normalizada": "Versión porcentual por grado real. Permite ver errores aunque haya pocos ejemplos en una clase.",
    "Cobertura por confianza": "Muestra qué porcentaje de casos aceptaríamos al exigir más confianza y qué accuracy queda en los casos aceptados.",
}


def external_section(report):
    if not report:
        return ""
    external = report.get("external_distribution", {})
    combined = report.get("combined_distribution", {})
    external_rows = "".join(
        f"<tr><td>{grade}</td><td>{format_int(count)}</td><td>{format_int(combined.get(str(grade), combined.get(grade)))}</td></tr>"
        for grade, count in sorted(external.items(), key=lambda item: int(item[0]))
    )
    return f"""
    <section>
      <h2>Datos externos usados solo para entrenamiento</h2>
      <p class="note">
        Se añadieron imágenes EyePACS redimensionadas al conjunto de entrenamiento, pero validación y test se mantienen con tus datos originales.
        Esto ayuda a que el modelo vea más ejemplos de grados 2, 3 y 4 sin contaminar la prueba final.
      </p>
      <div class="grid compact">
        <div class="card"><span>Imágenes externas</span><strong>{format_int(report.get("external_rows"))}</strong></div>
        <div class="card"><span>Entrenamiento combinado</span><strong>{format_int(report.get("combined_rows"))}</strong></div>
        <div class="card"><span>Imágenes fallidas</span><strong>{format_int(len(report.get("failed", [])))}</strong></div>
      </div>
      <table>
        <tr><th>Grado</th><th>Externas añadidas</th><th>Total en entrenamiento</th></tr>
        {external_rows}
      </table>
    </section>
    """


def clinical_section(report):
    if not report:
        return ""
    ref = report.get("referable_dr_grade_2_or_more", {})
    sev = report.get("severe_dr_grade_3_or_more", {})
    policy = report.get("selected_confidence_policy") or {}
    rows = ""
    for item in report.get("per_grade_clinical_metrics", []):
        rows += (
            f"<tr><td>{item.get('grade')}</td><td>{item.get('name')}</td>"
            f"<td>{format_int(item.get('support'))}</td>"
            f"<td>{format_value(item.get('sensitivity_recall'))}</td>"
            f"<td>{format_value(item.get('specificity'))}</td>"
            f"<td>{format_value(item.get('precision'))}</td></tr>"
        )
    return f"""
    <section>
      <h2>Auditoría clínica</h2>
      <p class="note">
        Este bloque evalúa el modelo como apoyo de investigación, no como dispositivo médico aprobado.
        Sensibilidad significa cuántos casos enfermos detecta; especificidad, cuántos sanos/no severos no marca de más.
      </p>
      <div class="grid compact">
        <div class="card"><span>Sensibilidad referible 2+</span><strong>{format_value(ref.get("sensitivity"))}</strong></div>
        <div class="card"><span>Especificidad referible 2+</span><strong>{format_value(ref.get("specificity"))}</strong></div>
        <div class="card"><span>Sensibilidad severa 3+</span><strong>{format_value(sev.get("sensitivity"))}</strong></div>
        <div class="card"><span>Error grande >=2 grados</span><strong>{format_value(report.get("metrics", {}).get("large_error_rate_grade_distance_2_or_more"))}</strong></div>
        <div class="card"><span>Umbral operativo</span><strong>{report.get("operating_confidence_threshold", "N/D")}</strong></div>
        <div class="card"><span>Coverage aceptada</span><strong>{format_value(policy.get("coverage"))}</strong></div>
        <div class="card"><span>Accuracy aceptada</span><strong>{format_value(policy.get("accuracy"))}</strong></div>
      </div>
      <table>
        <tr><th>Grado</th><th>Nombre</th><th>Casos</th><th>Sensibilidad</th><th>Especificidad</th><th>Precisión</th></tr>
        {rows}
      </table>
    </section>
    """


def calibration_section(report):
    if not report:
        return ""
    before = report.get("before", {})
    after = report.get("after", {})
    return f"""
    <section>
      <h2>Calibración de confianza</h2>
      <p class="note">
        La calibración ajusta la confianza reportada sin cambiar la clase predicha. Un ECE menor significa confianza más alineada con aciertos reales.
      </p>
      <div class="grid compact">
        <div class="card"><span>Temperatura</span><strong>{report.get("temperature", "N/D")}</strong></div>
        <div class="card"><span>ECE antes</span><strong>{format_value(before.get("ece_10_bins"))}</strong></div>
        <div class="card"><span>ECE después</span><strong>{format_value(after.get("ece_10_bins"))}</strong></div>
        <div class="card"><span>Confianza media</span><strong>{format_value(after.get("mean_confidence"))}</strong></div>
      </div>
    </section>
    """


def inference_policy_section(report):
    if not report:
        return ""
    return f"""
    <section>
      <h2>Política real de inferencia</h2>
      <p class="note">
        Esta auditoría usa el mismo flujo que predict.py: TTA, calibración, calidad de imagen, umbral por clase y margen top 1/top 2.
      </p>
      <div class="grid compact">
        <div class="card"><span>Casos aceptados</span><strong>{format_int(report.get("accepted"))}</strong></div>
        <div class="card"><span>Casos rechazados</span><strong>{format_int(report.get("rejected"))}</strong></div>
        <div class="card"><span>Cobertura real</span><strong>{format_value(report.get("coverage"))}</strong></div>
        <div class="card"><span>Accuracy aceptada</span><strong>{format_value(report.get("accepted_accuracy"))}</strong></div>
        <div class="card"><span>Error grande aceptado</span><strong>{format_value(report.get("accepted_large_error_rate_grade_distance_2_or_more"))}</strong></div>
        <div class="card"><span>Accuracy si se fuerza rechazo</span><strong>{format_value(report.get("rejected_accuracy_if_forced"))}</strong></div>
      </div>
    </section>
    """


def main():
    metrics = load_json_if_exists(RESULTS_DIR / "metrics.json")
    validation = load_json_if_exists(RESULTS_DIR / "model_validation.json")
    processed = load_json_if_exists(RESULTS_DIR / "processed_300_report.json")
    external = load_json_if_exists(RESULTS_DIR / "external_data_report.json")
    calibration = load_json_if_exists(RESULTS_DIR / "calibration.json")
    clinical = load_json_if_exists(RESULTS_DIR / "clinical_audit.json")
    inference_policy = load_json_if_exists(RESULTS_DIR / "inference_policy_audit.json")
    cards = [
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced Accuracy", metrics.get("balanced_accuracy")),
        ("F1 Macro", metrics.get("f1_macro")),
        ("F1 Weighted", metrics.get("f1_weighted")),
        ("Kappa Cuadrática", metrics.get("kappa_quadratic")),
        ("Aprobado", validation.get("approved_for_export")),
    ]
    card_html = "\n".join(
        f"<div class='card'><span>{label}</span><strong>{format_value(value)}</strong></div>"
        for label, value in cards
    )
    recall = metrics.get("recall_per_class", [])
    recall_rows = "\n".join(
        f"<tr><td>{idx}</td><td>{CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else ''}</td><td>{value * 100:.2f}%</td></tr>"
        for idx, value in enumerate(recall)
    )
    images = [
        ("Distribución de clases", RESULTS_DIR / "class_distribution.png"),
        ("Loss", RESULTS_DIR / "training_loss.png"),
        ("Accuracy", RESULTS_DIR / "training_accuracy.png"),
        ("Balanced Accuracy", RESULTS_DIR / "training_balanced_accuracy.png"),
        ("F1 Macro", RESULTS_DIR / "training_f1_macro.png"),
        ("Kappa", RESULTS_DIR / "training_kappa.png"),
        ("Métricas por grado", RESULTS_DIR / "class_metrics.png"),
        ("Matriz de confusión", RESULTS_DIR / "confusion_matrix.png"),
        ("Matriz normalizada", RESULTS_DIR / "normalized_confusion_matrix.png"),
        ("Cobertura por confianza", RESULTS_DIR / "confidence_coverage.png"),
    ]
    image_html = "\n".join(
        f"<section class='chart'><h2>{title}</h2><p class='note'>{CHART_NOTES.get(title, '')}</p><img src='{image_data_uri(path)}' alt='{title}'></section>"
        for title, path in images
        if path.exists()
    )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>RetinaAI - Reporte de Entrenamiento</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f3f5f7; color: #111827; }}
    header {{ background: #0f172a; color: white; padding: 28px 36px; }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }}
    .grid.compact {{ margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #d9dee3; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(15,23,42,.05); }}
    .card span {{ display: block; color: #64748b; font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 25px; color: #0f172a; }}
    section {{ background: white; border: 1px solid #d9dee3; border-radius: 8px; padding: 18px; margin-top: 18px; box-shadow: 0 1px 2px rgba(15,23,42,.05); }}
    h2 {{ margin-top: 0; font-size: 19px; letter-spacing: 0; }}
    .chart img {{ width: 100%; max-height: 820px; object-fit: contain; background: white; border: 1px solid #edf0f3; border-radius: 6px; }}
    .note {{ margin: -4px 0 14px; color: #475569; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin-top: 12px; }}
    td, th {{ border: 1px solid #d9dee3; padding: 9px 11px; text-align: left; }}
    th {{ background: #f8fafc; }}
    pre {{ white-space: pre-wrap; background: #101820; color: #e9f1f7; padding: 16px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>RetinaAI - Reporte de Entrenamiento</h1>
    <p>Resumen local con métricas, gráficas y explicación de cada resultado.</p>
  </header>
  <main>
    <div class="grid">{card_html}</div>
    {external_section(external)}
    {clinical_section(clinical)}
    {inference_policy_section(inference_policy)}
    {calibration_section(calibration)}
    <section>
      <h2>Recall por clase</h2>
      <p class="note">Recall significa: de todos los casos reales de ese grado, cuántos detectó correctamente el modelo.</p>
      <table><tr><th>Clase</th><th>Nombre</th><th>Recall</th></tr>{recall_rows}</table>
    </section>
    {image_html}
    <section>
      <h2>Validación del modelo</h2>
      <pre>{json.dumps(validation, indent=2, ensure_ascii=False)}</pre>
    </section>
    <section>
      <h2>Cache procesado</h2>
      <pre>{json.dumps(processed, indent=2, ensure_ascii=False)}</pre>
    </section>
  </main>
</body>
</html>"""
    output = RESULTS_DIR / "report.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
