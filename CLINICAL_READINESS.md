# RetinaAI - ruta hacia confiabilidad medica

Estado actual: prototipo de investigacion. No debe usarse como diagnostico autonomo ni reemplazar a un oftalmologo.

## Politica actual de seguridad

- El modelo predice grados 0 a 4.
- La inferencia revisa calidad de imagen antes de aceptar una prediccion.
- La confianza esta calibrada con validacion mediante `calibrate_model.py`.
- Si la imagen es borrosa, oscura, mal encuadrada o la confianza es baja, el sistema debe devolver `repeat_image_or_refer`.
- La inferencia devuelve tres estados: `accept`, `review` o `reject`.
- Umbral activo: `MIN_PREDICTION_CONFIDENCE = 0.70`.
- Aceptacion automatica: `AUTO_ACCEPT_CONFIDENCE = 0.80`.
- Revision minima: `REVIEW_MIN_CONFIDENCE = 0.70`.
- Margen minimo entre top 1 y top 2: `MIN_TOP2_MARGIN = 0.18`.
- Margen minimo para revision: `REVIEW_MIN_TOP2_MARGIN = 0.10`.
- Umbrales por grado: `[0.78, 0.70, 0.70, 0.72, 0.74]`.

Con el test actual y confianza calibrada:

- Accuracy global del checkpoint: 79.09%.
- Accuracy aceptada por la politica real de inferencia: 97.72%.
- Cobertura aceptada: 39.82%.
- Error grande aceptado de 2 o mas grados: 0.91%.
- Accuracy de casos rechazados si se forzara prediccion: 65.63%.
- Casos en revision: 1.45% del test, con 87.50% si se forzara prediccion.
- Error grande global del checkpoint: 4.55%.

Interpretacion: el sistema mejora mucho cuando puede abstenerse, pero todavia no es clinicamente confiable para diagnostico autonomo. La politica actual prioriza seguridad sobre cobertura: acepta pocos casos, principalmente predicciones muy claras, y deriva lo demas.

## Comandos de validacion

```powershell
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe calibrate_model.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe clinical_audit.py
.\.venv\Scripts\python.exe audit_inference_policy.py
.\.venv\Scripts\python.exe review_errors.py
.\.venv\Scripts\python.exe report_results.py
```

Abrir:

```powershell
Start-Process (Resolve-Path results\report.html)
```

## Lo que falta para uso clinico real

1. Validacion externa con imagenes locales nuevas, no usadas en entrenamiento ni ajuste.
2. Revision de errores grandes por un especialista.
3. Protocolo de captura con celular y adaptador retinal.
4. Dataset prospectivo con consentimiento y etiquetas verificadas.
5. Prueba por subgrupos: camara, iluminacion, edad, calidad de imagen y severidad.
6. Criterio formal de rechazo de imagenes de baja calidad.
7. Auditoria de calibracion y actualizacion del umbral antes de cada despliegue.
8. Documentacion regulatoria si se busca uso clinico real.

## Meta tecnica siguiente

Subir sensibilidad de grados 2, 3 y 4 sin perder especificidad. Para eso conviene:

- revisar `results/large_grade_errors.csv`;
- conseguir mas casos reales de grados 3 y 4;
- hacer preentrenamiento externo y fine-tuning solo con datos locales;
- probar arquitectura mas fuerte solo si la validacion externa mejora.
