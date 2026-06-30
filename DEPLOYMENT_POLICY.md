# RetinaAI - politica de uso del modelo congelado

Modelo protegido: `original_only_b1`.

Este modelo no se debe usar como diagnostico autonomo. La forma recomendada de uso es como asistente con abstencion y revision.

## Estados de inferencia

La salida de `predict.py` ahora incluye `policy_decision`:

| Estado | Significado | Accion |
|---|---|---|
| `accept` | Imagen aceptable, confianza alta y margen suficiente | Se puede mostrar como prediccion automatica del prototipo |
| `review` | Confianza o margen intermedio | Mostrar como orientativa y enviar a revision |
| `reject` | Imagen mala, confianza baja o prediccion muy ambigua | Repetir captura o derivar |

## Umbrales activos

| Parametro | Valor |
|---|---:|
| `AUTO_ACCEPT_CONFIDENCE` | 0.80 |
| `REVIEW_MIN_CONFIDENCE` | 0.70 |
| `MIN_TOP2_MARGIN` | 0.18 |
| `REVIEW_MIN_TOP2_MARGIN` | 0.10 |

Tambien se mantiene el control de calidad de imagen:

- brillo;
- contraste;
- nitidez;
- cobertura de retina;
- calidad global.

Si la imagen falla calidad, la decision es `reject` aunque la red tenga confianza.

## Comandos

Prediccion individual:

```powershell
.\.venv\Scripts\python.exe predict.py --image "ruta\imagen.png"
```

App visual local:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Auditar la politica contra el test local:

```powershell
.\.venv\Scripts\python.exe audit_inference_policy.py
```

Resultados:

- `results/inference_policy_audit.json`
- `results/inference_policy_predictions.csv`

## Decision tecnica

No seguir entrenando variantes a ciegas. Los experimentos con externo, `nuevodata`, B2, two-stage, filtros de calidad y ajustes orientados a accuracy no superaron el checkpoint actual.

Siguiente mejora real del checkpoint: mas datos locales o depuracion manual de etiquetas/casos dudosos del dataset original.
