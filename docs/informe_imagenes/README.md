# Imagenes para el informe RetinaAI

Carpeta preparada para subir junto con el informe `construccion.md`.

## Verificacion del modelo

Las imagenes de metricas copiadas en esta carpeta corresponden al modelo actual exportado como:

- Checkpoint: `models/best_retina_model.pth`
- Peso aproximado: 71.26 MB
- Estado de validacion: aprobado para exportacion
- Modelo colapsado: no
- Clases predichas: 5
- Dataset final usado para el modelo consolidado: dataset original

Metricas verificadas desde `results/metrics.json`:

| Metrica | Valor |
|---|---:|
| Accuracy general | 79.09% |
| Balanced accuracy | 67.33% |
| F1 macro | 65.04% |
| F1 weighted | 79.57% |
| Kappa cuadratico | 86.16% |
| ROC AUC multiclase | 92.01% |
| Error grande >= 2 grados | 4.55% |

Politica de seguridad verificada desde `results/inference_policy_audit.json`:

| Grupo | Valor |
|---|---:|
| Total evaluado | 550 |
| Predicciones aceptadas automaticamente | 219 |
| Cobertura aceptada | 39.82% |
| Accuracy en aceptadas | 97.72% |
| Error grande en aceptadas | 0.91% |

## Imagenes incluidas

| Imagen | Usar en seccion | Descripcion |
|---|---|---|
| `class_distribution.png` | 4.2 Exploracion de los Datos | Distribucion de clases del dataset original. |
| `normalized_confusion_matrix.png` | 6.3 Matriz de Confusion | Matriz de confusion normalizada del modelo actual. |
| `confusion_matrix.png` | 6.3 Matriz de Confusion | Matriz de confusion con conteos absolutos. |
| `class_metrics.png` | 6.2 Metricas de Evaluacion | Precision, recall y F1 por clase. |
| `training_accuracy.png` | 6.4 Curvas de Aprendizaje | Evolucion del accuracy durante entrenamiento. |
| `training_loss.png` | 6.4 Curvas de Aprendizaje | Evolucion de la perdida durante entrenamiento. |
| `training_f1_macro.png` | 6.4 Curvas de Aprendizaje | Evolucion del F1 macro durante entrenamiento. |
| `training_kappa.png` | 6.4 Curvas de Aprendizaje | Evolucion del kappa cuadratico durante entrenamiento. |
| `training_balanced_accuracy.png` | 6.4 Curvas de Aprendizaje | Evolucion del balanced accuracy durante entrenamiento. |
| `confidence_coverage.png` | 6.5 Analisis de Resultados | Relacion entre confianza, cobertura y politica de aceptacion. |

## Rutas recomendadas para Markdown

Si el informe se mantiene en la raiz del proyecto, usar:

```md
![Distribucion de clases](docs/informe_imagenes/class_distribution.png)
![Matriz de confusion normalizada](docs/informe_imagenes/normalized_confusion_matrix.png)
![Metricas por clase](docs/informe_imagenes/class_metrics.png)
![Curva de accuracy](docs/informe_imagenes/training_accuracy.png)
![Curva de perdida](docs/informe_imagenes/training_loss.png)
![Curva F1 macro](docs/informe_imagenes/training_f1_macro.png)
![Curva kappa](docs/informe_imagenes/training_kappa.png)
![Confianza y cobertura](docs/informe_imagenes/confidence_coverage.png)
```
