# RetinaAI - registro de experimentos

Este archivo resume decisiones, resultados y rutas descartadas para no repetir pruebas que ya demostraron empeorar el modelo.

## Entorno

- Fecha de trabajo: 2026-06-29.
- GPU detectada: NVIDIA GeForce RTX 4050 Laptop GPU.
- VRAM: 6 GB.
- Python recomendado: `.\.venv\Scripts\python.exe`.
- PyTorch en `.venv`: `2.12.1+cu126`.
- CUDA en `.venv`: activo.
- Tests: `7 passed`.

Comando de verificacion:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pytest
```

## Baseline actual conservado

Este es el modelo que se debe proteger hasta que un experimento lo supere claramente.

Actualizacion 2026-06-29: el baseline conservado ahora es `original_only_b1`, entrenado solo con dataset local/original. Reemplazo al baseline anterior con externo bruto porque subio el score compuesto y redujo errores grandes.

Dataset del baseline actual:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Filas de entrenamiento: 2563.
- Validacion: `data/splits/val_processed_300_split.csv`.
- Test: `data/splits/test_processed_300_split.csv`.
- Datos externos: no.
- `nuevodata`: no.
- Imagenes procesadas: 300 px.
- Arquitectura: `efficientnet_b1`.

Distribucion de entrenamiento del baseline actual:

| Grado | Filas |
|---:|---:|
| 0 | 1263 |
| 1 | 259 |
| 2 | 699 |
| 3 | 135 |
| 4 | 207 |

Metricas del baseline actual:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.7909 |
| Balanced accuracy | 0.6733 |
| F1 macro | 0.6504 |
| Kappa cuadratica | 0.8616 |
| Recall grado 2 | 0.6333 |
| Recall grado 3 | 0.5517 |
| Recall grado 4 | 0.5682 |
| Error grande, distancia >= 2 grados | 0.0455 |
| Sensibilidad referible, grado 2+ | 0.8341 |
| Sensibilidad severa, grado 3+ | 0.8356 |

Politica de confianza actual:

| Umbral | Cobertura | Accuracy aceptada | Error medio aceptado |
|---:|---:|---:|---:|
| 0.70 | 0.6255 | 0.9099 | 0.1076 |

## Baseline anterior con externo bruto

Dataset del baseline exportado:

- Entrenamiento: `data/splits/train_with_external_processed_300_split.csv`.
- Filas de entrenamiento: 5944.
- Filas externas dentro del entrenamiento: 3381.
- Validacion: `data/splits/val_processed_300_split.csv`.
- Test: `data/splits/test_processed_300_split.csv`.
- Imagenes procesadas: 300 px.
- Arquitectura: `efficientnet_b1`.

Distribucion de entrenamiento del baseline:

| Grado | Filas |
|---:|---:|
| 0 | 1563 |
| 1 | 559 |
| 2 | 1899 |
| 3 | 1008 |
| 4 | 915 |

| Metrica | Valor |
|---|---:|
| Accuracy | 0.7855 |
| Balanced accuracy | 0.6793 |
| F1 macro | 0.6474 |
| Kappa cuadratica | 0.8627 |
| Recall grado 2 | 0.5933 |
| Recall grado 3 | 0.5862 |
| Recall grado 4 | 0.5000 |
| Error grande, distancia >= 2 grados | 0.0491 |
| Sensibilidad referible, grado 2+ | 0.8072 |
| Sensibilidad severa, grado 3+ | 0.7808 |

Politica de confianza reportada en `clinical_audit.json`:

| Umbral | Cobertura | Accuracy aceptada | Error medio aceptado |
|---:|---:|---:|---:|
| 0.70 | 0.6345 | 0.8968 | 0.1089 |

## Decision de promocion

Un experimento solo debe reemplazar al baseline si:

- sube el score compuesto de `run_experiment.py`;
- no aumenta el error grande de 2 o mas grados;
- no baja la kappa mas de 0.01;
- idealmente mejora recall de grados 3 y 4 sin destruir grado 2.

El score compuesto pondera F1 macro, balanced accuracy, kappa, recall de grados 2/3/4 y penaliza errores grandes.

## Regla de datasets

- El dataset externo se puede usar para entrenamiento porque aporta muchos casos de grados 3 y 4.
- Validacion y test final deben quedarse locales para medir utilidad real en el dominio objetivo.
- Cada experimento debe registrar el CSV exacto de entrenamiento.
- Las comparaciones mas justas contra el baseline deben usar `data/splits/train_with_external_processed_300_split.csv` o una variante externa limpia declarada.
- No comparar conclusiones fuertes entre un modelo entrenado con externo y otro entrenado solo local sin dejarlo anotado.

## Nuevodata

Preparado con `prepare_nuevodata.py`.

Fuente:

- `nuevodata/train.csv`
- `nuevodata/test.csv`
- `nuevodata/train_images_512/train_images_512`
- `nuevodata/test_images_512/test_images_512`

Validacion:

- Train original: 35126 filas.
- Train valido con imagen existente: 35121 filas.
- Test original: 53576 filas.
- Test valido con imagen existente: 53571 filas.
- Faltantes: 5 imagenes en train y 5 en test.
- Solapamiento por `id_code` con datos anteriores: 0 detectado.
- Muestra de calidad: aceptable por clase entre 95.83% y 100%.

Distribucion train valido:

| Grado | Filas |
|---:|---:|
| 0 | 25808 |
| 1 | 2441 |
| 2 | 5291 |
| 3 | 873 |
| 4 | 708 |

Se limito grado 0 nuevo a 6000 para evitar que aplaste el entrenamiento.

Tambien se preparo una mezcla dirigida para aprovechar la mejora observada en grado 4 sin dejar que `nuevodata` domine todo:

| Grado nuevodata | Límite usado |
|---:|---:|
| 0 | 500 |
| 1 | 500 |
| 2 | 1500 |
| 3 | todos, 873 |
| 4 | todos, 708 |

CSV dirigido:

| CSV | Filas | Distribucion |
|---|---:|---|
| `data/splits/nuevodata_train_targeted_512.csv` | 4081 | 0:500, 1:500, 2:1500, 3:873, 4:708 |
| `data/splits/train_with_external_and_nuevodata_targeted_300_split.csv` | 10025 | 0:2063, 1:1059, 2:3399, 3:1881, 4:1623 |

CSV generados:

| CSV | Filas | Uso |
|---|---:|---|
| `data/splits/nuevodata_train_512.csv` | 35121 | nuevo completo |
| `data/splits/nuevodata_train_balanced_512.csv` | 15313 | nuevo con grado 0 limitado |
| `data/splits/nuevodata_train_targeted_512.csv` | 4081 | nuevo dirigido a 3/4 |
| `data/splits/train_with_nuevodata_balanced_300_split.csv` | 17876 | local + nuevodata balanceado, entrada 300 |
| `data/splits/train_with_external_and_nuevodata_targeted_300_split.csv` | 10025 | baseline externo bruto + nuevodata dirigido |
| `data/splits/train_with_nuevodata_balanced_512_split.csv` | 17876 | local + nuevodata balanceado, entrada 512 |

Primer uso recomendado:

- Entrenar con `train_with_nuevodata_balanced_300_split.csv`.
- Validar/testear con splits locales existentes.
- No usar `nuevodata/test.csv` como juez principal; sirve para analisis secundario.

## Cambios hechos al codigo

- `run_experiment.py`: runner seguro que guarda snapshot en `experiments/`, corre entrenamiento/evaluacion/calibracion/auditorias, compara metricas y restaura el baseline si empeora.
- `run_experiment.py`: agrega `--train-csv` para elegir explicitamente el CSV de entrenamiento.
- `prepare_nuevodata.py`: valida y prepara `nuevodata`, genera CSVs balanceados y combinados con los datos locales.
- `prepare_original_quality_clean_training.py`: genera un CSV original-only filtrado por calidad cruda, usando la cache `train_with_external_raw_300_quality.csv` cuando existe.
- `create_dataset_review_queue.py`: crea una cola priorizada para revision manual del dataset original, enfocada en entrenamiento y grados 2/3/4.
- `render_dataset_review_queue.py`: genera un HTML visual de la cola de revision con imagen procesada/original y metricas de calidad.
- `apply_manual_review.py`: aplica `review_corrected_label` y `review_exclude` desde la cola manual para crear splits curados.
- `config/config.py`: `MODEL_NAME`, resolucion y augmentations ahora se pueden configurar con variables `RETINAAI_*`.
- `src/preprocessing.py`: augmentations configurables: crop, rotacion, color jitter, blur leve y sharpness.
- `src/model.py`: soporte para `efficientnet_b3`.
- `.gitignore`: `experiments/*` queda fuera del repo porque guarda checkpoints y reportes pesados.

## Experimentos realizados

### 20260629_103257_severe_focus_b1

Objetivo: subir recall de grados 3 y 4 usando focal loss, mas peso en clases severas y augmentations mas fuertes.

Dataset usado por este experimento:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Filas de entrenamiento: 2563.
- Filas externas: 0.
- Validacion/test: splits locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name severe_focus_b1 --class-multipliers "1.0,1.0,1.2,1.6,1.5" --sampler-power 0.6 --focal-loss --ordinal-alpha 0.25 --crop-min-scale 0.80 --rotation 12 --color-jitter 0.16 --blur-prob 0.10 --sharpness-prob 0.20
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7200 |
| Balanced accuracy | 0.6793 | 0.6449 |
| F1 macro | 0.6474 | 0.5823 |
| Kappa cuadratica | 0.8627 | 0.7884 |
| Recall grado 2 | 0.5933 | 0.4267 |
| Recall grado 3 | 0.5862 | 0.6207 |
| Recall grado 4 | 0.5000 | 0.5227 |
| Error grande | 0.0491 | 0.0745 |
| Sensibilidad referible 2+ | 0.8072 | 0.7758 |
| Sensibilidad severa 3+ | 0.7808 | 0.8356 |

Decision:

- No repetir esta configuracion.
- Focal loss y pesos severos agresivos subieron algo el recall severo, pero redujeron demasiado grado 2, bajaron kappa y subieron errores grandes.
- Nota importante: no fue una comparacion estricta contra el baseline, porque este experimento entreno solo con datos locales mientras el baseline exportado habia usado `train_with_external_processed_300_split.csv`.
- Proxima prueba debe ser mas conservadora: sin focal loss, pesos moderados y augmentations suaves.

### 20260629_111040_moderate_severe_weights_no_focal

Objetivo: conservar el aprendizaje del experimento anterior sobre severos, pero sin focal loss y con pesos menos agresivos.

Dataset usado por este experimento:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Filas de entrenamiento: 2563.
- Filas externas: 0.
- Validacion/test: splits locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name moderate_severe_weights_no_focal --class-multipliers "1.0,1.0,1.18,1.45,1.35" --sampler-power 0.55 --ordinal-alpha 0.30 --crop-min-scale 0.82 --rotation 10 --color-jitter 0.14 --blur-prob 0.08 --sharpness-prob 0.18
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7655 |
| Balanced accuracy | 0.6793 | 0.6836 |
| F1 macro | 0.6474 | 0.6327 |
| Kappa cuadratica | 0.8627 | 0.8497 |
| Recall grado 2 | 0.5933 | 0.5200 |
| Recall grado 3 | 0.5862 | 0.6552 |
| Recall grado 4 | 0.5000 | 0.5909 |
| Error grande | 0.0491 | 0.0527 |
| Sensibilidad referible 2+ | 0.8072 | 0.8251 |
| Sensibilidad severa 3+ | 0.7808 | 0.8904 |

Decision:

- No promover.
- La sensibilidad severa subio bastante, pero el experimento aumento errores grandes y bajo kappa/F1 macro.
- Esta direccion puede servir si el objetivo fuera solo tamizaje sensible, pero no como reemplazo general del baseline.
- Nota importante: no fue una comparacion estricta contra el baseline, porque este experimento entreno solo con datos locales mientras el baseline exportado habia usado `train_with_external_processed_300_split.csv`.
- Proxima prueba: resolucion 512 conservadora, para intentar recuperar detalle sin empujar clases artificialmente.

### 20260629_113758_b1_512_conservative

Objetivo: probar si mayor resolucion recupera detalles retinales pequenos sin cambiar fuerte la funcion de perdida.

Dataset usado por este experimento:

- Entrenamiento: `data/splits/train_processed_512_split.csv`.
- Filas de entrenamiento: 2563.
- Filas externas: 0.
- Validacion/test: splits locales procesados de 512 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name b1_512_conservative --image-size 512 --batch-size 2 --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7618 |
| Balanced accuracy | 0.6793 | 0.5804 |
| F1 macro | 0.6474 | 0.5873 |
| Kappa cuadratica | 0.8627 | 0.8502 |
| Recall grado 2 | 0.5933 | 0.7667 |
| Recall grado 3 | 0.5862 | 0.2759 |
| Recall grado 4 | 0.5000 | 0.3409 |
| Error grande | 0.0491 | 0.0527 |
| Sensibilidad referible 2+ | 0.8072 | 0.8430 |
| Sensibilidad severa 3+ | 0.7808 | 0.4658 |

Decision:

- No promover.
- No repetir 512 con EfficientNet-B1 y esta configuracion.
- La resolucion 512 mejoro moderada/referible, pero destruyo la sensibilidad severa. Puede ser falta de epocas, batch pequeno o que el preprocesamiento 512 cambio la distribucion de detalles.
- Nota importante: no fue una comparacion estricta contra el baseline, porque este experimento entreno solo con datos locales mientras el baseline exportado habia usado datos externos procesados de 300 px.
- Si se vuelve a probar 512, debe ser con una estrategia distinta: mas paciencia, learning rate menor o dos etapas, no como reemplazo directo del baseline.

### 20260629_122314_b2_external_clean_conservative

Objetivo: probar arquitectura mas fuerte (`efficientnet_b2`) con dataset externo limpio y test local.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name b2_external_clean_conservative --model-name efficientnet_b2 --batch-size 2 --train-csv data\splits\train_with_external_quality_clean_processed_300_split.csv --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: interrumpido por timeout, no comparable.

Notas:

- El entrenamiento excedio 40 minutos y el proceso quedo vivo tras el timeout.
- Se detuvieron manualmente los procesos Python asociados.
- Se restauro el snapshot `before` de modelos, resultados y export.
- No tomar este intento como fracaso de B2; simplemente no termino.

Decision:

- Para B2 usar una prueba corta primero: menos epocas o `TEST_MODE`/sample temporal.
- Alternativa mas practica: continuar con EfficientNet-B1 usando el dataset externo limpio o bruto, porque B2 completo tarda demasiado para iterar.

### 20260629_130739_b1_external_clean_conservative

Objetivo: comparar de forma mas justa contra el baseline usando EfficientNet-B1, 300 px, test local y dataset externo limpio.

Dataset usado:

- Entrenamiento: `data/splits/train_with_external_quality_clean_processed_300_split.csv`.
- Filas de entrenamiento: 5041.
- Filas externas: 3362.
- Validacion/test: locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name b1_external_clean_conservative --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_with_external_quality_clean_processed_300_split.csv --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7218 |
| Balanced accuracy | 0.6793 | 0.5383 |
| F1 macro | 0.6474 | 0.5335 |
| Kappa cuadratica | 0.8627 | 0.8345 |
| Recall grado 2 | 0.5933 | 0.6067 |
| Recall grado 3 | 0.5862 | 0.2414 |
| Recall grado 4 | 0.5000 | 0.5455 |
| Error grande | 0.0491 | 0.0582 |
| Sensibilidad referible 2+ | 0.8072 | 0.8969 |
| Sensibilidad severa 3+ | 0.7808 | 0.6712 |

Decision:

- No promover.
- El externo limpio subio sensibilidad referible y grado 4, pero destruyo grado 3 y subio errores grandes.
- No asumir que "limpio" es automaticamente mejor que el externo bruto del baseline; la limpieza pudo haber quitado ejemplos utiles o cambiado la distribucion.
- Siguiente prueba razonable: replicar baseline con el CSV externo bruto explicito para confirmar reproducibilidad, o pasar a ajuste de politica/umbrales sin reentrenar.

### 20260629_142112_b1_nuevodata_balanced_300

Objetivo: entrenar EfficientNet-B1 con `nuevodata` completo balanceado, conservando validacion/test locales.

Dataset usado:

- Entrenamiento: `data/splits/train_with_nuevodata_balanced_300_split.csv`.
- Filas de entrenamiento: 17876.
- Distribucion: grado 0 = 7263, grado 1 = 2700, grado 2 = 5990, grado 3 = 1008, grado 4 = 915.
- Validacion/test: locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name b1_nuevodata_balanced_300 --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_with_nuevodata_balanced_300_split.csv --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7036 |
| Balanced accuracy | 0.6793 | 0.5732 |
| F1 macro | 0.6474 | 0.5473 |
| Kappa cuadratica | 0.8627 | 0.8186 |
| Recall grado 2 | 0.5933 | 0.5267 |
| Recall grado 3 | 0.5862 | 0.3103 |
| Recall grado 4 | 0.5000 | 0.7500 |
| Error grande | 0.0491 | 0.1164 |
| Sensibilidad referible 2+ | 0.8072 | 0.8789 |
| Sensibilidad severa 3+ | 0.7808 | 0.8082 |

Decision:

- No promover.
- `nuevodata` balanceado mejora mucho grado 4 y sube sensibilidad referible, pero destruye grado 3 y mas que duplica errores grandes.
- El dominio de `nuevodata` probablemente no calza directo con el test local o el balance usado empuja demasiado hacia enfermedad avanzada.
- No usar `nuevodata` completo balanceado como reemplazo directo del baseline.
- Siguiente enfoque recomendado: usar `nuevodata` de forma auxiliar, no como reemplazo directo: preentrenamiento, mezcla mas pequena, o segunda etapa 2/3/4.

### 20260629_160228_b1_nuevodata_targeted_34

Objetivo: aprovechar la mejora observada en grado 4 agregando una mezcla dirigida de `nuevodata` encima del dataset externo bruto del baseline.

Dataset usado:

- Entrenamiento: `data/splits/train_with_external_and_nuevodata_targeted_300_split.csv`.
- Filas de entrenamiento: 10025.
- Distribucion: grado 0 = 2063, grado 1 = 1059, grado 2 = 3399, grado 3 = 1881, grado 4 = 1623.
- Validacion/test: locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name b1_nuevodata_targeted_34 --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_with_external_and_nuevodata_targeted_300_split.csv --class-multipliers "1.0,1.0,1.10,1.25,1.35" --sampler-power 0.45 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline | Experimento |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7218 |
| Balanced accuracy | 0.6793 | 0.5602 |
| F1 macro | 0.6474 | 0.5457 |
| Kappa cuadratica | 0.8627 | 0.8329 |
| Recall grado 2 | 0.5933 | 0.4867 |
| Recall grado 3 | 0.5862 | 0.2759 |
| Recall grado 4 | 0.5000 | 0.5682 |
| Error grande | 0.0491 | 0.0727 |
| Sensibilidad referible 2+ | 0.8072 | 0.8475 |
| Sensibilidad severa 3+ | 0.7808 | 0.7123 |

Decision:

- No promover.
- Aunque grado 4 subio, el costo en grado 2/3 y errores grandes sigue siendo demasiado alto.
- `nuevodata` no debe mezclarse directamente en el clasificador principal con este enfoque.
- Siguiente camino mas razonable: usar `nuevodata` fuera del entrenamiento principal, por ejemplo como segunda etapa/refinador o como criterio de derivacion/politica, no como reemplazo de los datos base.

### 20260629_170240_original_only_b1

Objetivo: entrenar solo con el dataset original/local, sin externo anterior y sin `nuevodata`, para evitar mezcla de dominios.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Filas de entrenamiento: 2563.
- Distribucion: grado 0 = 1263, grado 1 = 259, grado 2 = 699, grado 3 = 135, grado 4 = 207.
- Validacion/test: locales procesados de 300 px.
- Externo anterior: no.
- `nuevodata`: no.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name original_only_b1 --model-name efficientnet_b1 --batch-size 4 --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: promovido y conservado.

| Metrica | Baseline externo anterior | Original only |
|---|---:|---:|
| Accuracy | 0.7855 | 0.7909 |
| Balanced accuracy | 0.6793 | 0.6733 |
| F1 macro | 0.6474 | 0.6504 |
| Kappa cuadratica | 0.8627 | 0.8616 |
| Recall grado 2 | 0.5933 | 0.6333 |
| Recall grado 3 | 0.5862 | 0.5517 |
| Recall grado 4 | 0.5000 | 0.5682 |
| Error grande | 0.0491 | 0.0455 |
| Sensibilidad referible 2+ | 0.8072 | 0.8341 |
| Sensibilidad severa 3+ | 0.7808 | 0.8356 |

Decision:

- Promover.
- Aunque balanced accuracy y kappa bajan levemente, mejora accuracy, F1 macro, recall grado 2, recall grado 4, sensibilidad referible, sensibilidad severa y baja errores grandes.
- Este modelo es mas limpio para produccion/investigacion porque no mezcla dominios externos en entrenamiento.
- Nuevo baseline protegido: `original_only_b1`.

### 20260629_173243_original_only_seed7

Objetivo: probar estabilidad del baseline original-only cambiando solo la semilla.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.
- Semilla: 7.

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | Seed 7 |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7673 |
| Balanced accuracy | 0.6733 | 0.6671 |
| F1 macro | 0.6504 | 0.6303 |
| Kappa cuadratica | 0.8616 | 0.8436 |
| Recall grado 2 | 0.6333 | 0.5333 |
| Recall grado 3 | 0.5517 | 0.6207 |
| Recall grado 4 | 0.5682 | 0.5682 |
| Error grande | 0.0455 | 0.0564 |
| Sensibilidad referible 2+ | 0.8341 | 0.8296 |
| Sensibilidad severa 3+ | 0.8356 | 0.8493 |

Decision:

- No promover.
- La semilla 7 mejora grado 3 y sensibilidad severa, pero baja demasiado grado 2, accuracy, F1 macro y kappa, y sube errores grandes.
- Puede ser candidata para ensemble si se implementa promediado de probabilidades, pero no como modelo unico.

### 20260629_175703_original_only_soft_aug

Objetivo: probar original-only con augmentations mas suaves para no distorsionar lesiones sutiles.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.
- Semilla: 42.
- Cambios: crop min 0.90, rotacion 6, color jitter 0.08, blur 0.04, sharpness 0.08.

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | Soft aug |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7800 |
| Balanced accuracy | 0.6733 | 0.6777 |
| F1 macro | 0.6504 | 0.6430 |
| Kappa cuadratica | 0.8616 | 0.8485 |
| Recall grado 2 | 0.6333 | 0.6000 |
| Recall grado 3 | 0.5517 | 0.6552 |
| Recall grado 4 | 0.5682 | 0.4773 |
| Error grande | 0.0455 | 0.0527 |
| Sensibilidad referible 2+ | 0.8341 | 0.8161 |
| Sensibilidad severa 3+ | 0.8356 | 0.7671 |

Decision:

- No promover.
- Augmentations suaves mejoraron grado 3, pero bajaron grado 4 y subieron errores grandes.
- El baseline original-only sigue siendo mejor como modelo unico.

### Ensembles original-only

Objetivo: evaluar si los checkpoints `original_only_seed7` y `original_only_soft_aug`, aunque peores como modelos unicos, aportaban diversidad al promediar probabilidades con el baseline.

Resultado: descartados.

| Ensemble | Pesos | Accuracy | F1 macro | Kappa | Error grande | Recall G3 | Recall G4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_seed7` | 0.50 / 0.50 | 0.6291 | 0.4634 | 0.7159 | 0.1655 | 0.6207 | 0.6818 |
| `baseline_softaug` | 0.50 / 0.50 | 0.6618 | 0.5199 | 0.7491 | 0.1273 | 0.6207 | 0.6364 |
| `baseline80_seed7` | 0.80 / 0.20 | 0.6255 | 0.4654 | 0.6888 | 0.1764 | 0.6207 | 0.7045 |
| `baseline80_softaug` | 0.80 / 0.20 | 0.6345 | 0.4893 | 0.6944 | 0.1582 | 0.6207 | 0.6818 |
| `baseline90_seed7` | 0.90 / 0.10 | 0.6127 | 0.4569 | 0.6789 | 0.1818 | 0.6207 | 0.6818 |

Decision:

- No promover ningun ensemble.
- Los ensembles aumentan sensibilidad severa en algunos cortes, pero sobrepredicen grados 3/4, bajan mucho accuracy, bajan kappa y multiplican los errores grandes.
- No repetir promediado simple de probabilidades con estos checkpoints.

### 20260629_192049_original_quality_clean_b1

Objetivo: subir el modelo sin mezclar datasets, eliminando imagenes originales con baja calidad cruda antes de entrenar.

CSV preparado:

- Script: `prepare_original_quality_clean_training.py`.
- Entrada: `data/splits/train_processed_300_split.csv`.
- Cache de calidad: `data/splits/train_with_external_raw_300_quality.csv`.
- Salida: `data/splits/train_original_quality_clean_processed_300_split.csv`.
- Regla: `raw_quality_score>=0.60`, `raw_sharpness>=8`, `raw_retina_coverage>=0.14`.
- Filas originales: 2563.
- Filas conservadas: 2131.
- Filas removidas: 432.

Distribucion conservada:

| Grado | Filas |
|---:|---:|
| 0 | 1250 |
| 1 | 202 |
| 2 | 451 |
| 3 | 87 |
| 4 | 141 |

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name original_quality_clean_b1 --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_original_quality_clean_processed_300_split.csv --class-multipliers "1.0,1.05,1.25,1.55,1.45" --sampler-power 0.55 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | Quality clean |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7436 |
| Balanced accuracy | 0.6733 | 0.6436 |
| F1 macro | 0.6504 | 0.6020 |
| Kappa cuadratica | 0.8616 | 0.8189 |
| Recall grado 2 | 0.6333 | 0.5067 |
| Recall grado 3 | 0.5517 | 0.6207 |
| Recall grado 4 | 0.5682 | 0.4318 |
| Error grande | 0.0455 | 0.0600 |
| Sensibilidad referible 2+ | 0.8341 | 0.7758 |
| Sensibilidad severa 3+ | 0.8356 | 0.7945 |

Decision:

- No promover.
- Limpiar por calidad cruda removio demasiados casos utiles de grados 2/3/4.
- La calidad debe usarse mejor como politica de rechazo/repeticion de captura en inferencia, no como filtro duro del entrenamiento actual.

### 20260629_194047_original_only_b2

Objetivo: subir capacidad del modelo sin mezclar datasets, cambiando `efficientnet_b1` por `efficientnet_b2` sobre el dataset original-only.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.
- Validacion/test: locales procesados de 300 px.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name original_only_b2 --model-name efficientnet_b2 --batch-size 4 --train-csv data\splits\train_processed_300_split.csv --class-multipliers "1.0,1.0,1.15,1.35,1.25" --sampler-power 0.50 --ordinal-alpha 0.35 --crop-min-scale 0.85 --rotation 10 --color-jitter 0.12 --blur-prob 0.08 --sharpness-prob 0.15
```

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | EfficientNet-B2 |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7800 |
| Balanced accuracy | 0.6733 | 0.6380 |
| F1 macro | 0.6504 | 0.6204 |
| Kappa cuadratica | 0.8616 | 0.8551 |
| Recall grado 2 | 0.6333 | 0.6533 |
| Recall grado 3 | 0.5517 | 0.5172 |
| Recall grado 4 | 0.5682 | 0.4773 |
| Error grande | 0.0455 | 0.0618 |
| Sensibilidad referible 2+ | 0.8341 | 0.8610 |
| Sensibilidad severa 3+ | 0.8356 | 0.7945 |

Decision:

- No promover.
- B2 mejora grado 2 y sensibilidad referible, pero baja grados 3/4 y sube errores grandes.
- No repetir B2 original-only con esta configuracion como reemplazo general.

### Two-stage original-only 2/3/4

Objetivo: entrenar un pipeline de dos etapas sin datasets externos:

- Etapa 1: clasificador binario `0/1` vs `2+`.
- Etapa 2: refinador especializado solo para grados `2`, `3` y `4`.

Dataset usado:

- Entrenamiento base: `data/splits/train_processed_300_split.csv`.
- Validacion: `data/splits/val_processed_300_split.csv`.
- Test: `data/splits/test_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.

Distribucion de entrenamiento:

| Etapa | Clase | Filas |
|---|---:|---:|
| Stage 1 | 0/1 | 1522 |
| Stage 1 | 2+ | 1041 |
| Stage 2 | grado 2 | 699 |
| Stage 2 | grado 3 | 135 |
| Stage 2 | grado 4 | 207 |

Comandos:

```powershell
.\.venv\Scripts\python.exe train_two_stage.py --no-external-stage2
.\.venv\Scripts\python.exe evaluate_two_stage.py
```

Resultado: descartado.

El umbral elegido en validacion fue `0.35`, pero en test el pipeline completo empeoro frente al modelo actual.

| Metrica | Modelo actual | Two-stage original-only |
|---|---:|---:|
| Accuracy | 0.7909 | 0.6909 |
| Balanced accuracy | 0.6733 | 0.5306 |
| F1 macro | 0.6504 | 0.4865 |
| Kappa cuadratica | 0.8616 | 0.8070 |
| Error grande | 0.0455 | 0.1036 |
| Recall grado 2 | 0.6333 | 0.5267 |
| Recall grado 3 | 0.5517 | 0.5517 |
| Recall grado 4 | 0.5682 | 0.5227 |

Matriz de confusion two-stage:

```text
Real\Pred   0    1    2    3    4
0         256    4    8    2    1
1           6    6   37    1    6
2           6    3   79   39   23
3           0    0    4   16    9
4           1    1    8   11   23
```

Decision:

- No promover.
- El refinador 2/3/4 solo original no tiene suficiente señal para separar bien las clases severas.
- El stage 1 aprende bien `0/1` vs `2+`, pero al conectar el refinador se generan demasiados falsos severos y errores grandes.
- Si se retoma esta idea, debe ser con mas datos revisados de grados 3/4 o con correccion manual de etiquetas; no repetir esta configuracion.

### Two-stage con externo solo en refinador 2/3/4

Objetivo: probar el uso menos riesgoso del dataset externo: mantener intacto el modelo principal `original_only_b1` y usar datos externos solo para entrenar el refinador `2/3/4`.

Dataset usado:

- Stage 1: original/local solamente.
- Stage 2: original/local grados `2/3/4` + `external_eyepacs_processed_300.csv` grados `2/3/4`.
- Validacion/test: originales/locales.
- `nuevodata`: no.

Distribucion de entrenamiento:

| Etapa | Clase | Filas |
|---|---:|---:|
| Stage 1 | 0/1 | 1522 |
| Stage 1 | 2+ | 1041 |
| Stage 2 | grado 2 | 1899 |
| Stage 2 | grado 3 | 1008 |
| Stage 2 | grado 4 | 915 |

Comandos:

```powershell
.\.venv\Scripts\python.exe train_two_stage.py
.\.venv\Scripts\python.exe evaluate_two_stage.py
```

Resultado: descartado.

El umbral elegido en validacion fue `0.35`. En test local, el pipeline no supero al modelo actual.

| Metrica | Modelo actual | Two-stage con externo |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7000 |
| Balanced accuracy | 0.6733 | 0.5062 |
| F1 macro | 0.6504 | 0.4877 |
| Kappa cuadratica | 0.8616 | 0.8314 |
| Error grande | 0.0455 | 0.0709 |
| Recall grado 2 | 0.6333 | 0.6000 |
| Recall grado 3 | 0.5517 | 0.3793 |
| Recall grado 4 | 0.5682 | 0.5000 |

Matriz de confusion two-stage con externo:

```text
Real\Pred   0    1    2    3    4
0         256    4    9    0    2
1           6    6   42    0    2
2           6    3   90   44    7
3           0    0    7   11   11
4           1    1   11    9   22
```

Decision:

- No promover.
- Usar externo solo como refinador reduce el dano frente a mezclarlo en el modelo principal, pero igual mete ruido suficiente para bajar F1, balanced accuracy y recall de grados 3/4.
- El modelo actual `original_only_b1` sigue siendo el mejor modelo protegido.
- No repetir este two-stage con externo sin una estrategia nueva de curacion/seleccion de externos.

### 20260629_211924_original_accuracy_plain_ce

Objetivo: subir accuracy del checkpoint, no sensibilidad clinica, probando un entrenamiento mas simple: sin sampler ponderado, sin perdida ordinal y con pesos de clase neutros.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name original_accuracy_plain_ce --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_processed_300_split.csv --no-weighted-sampler --no-ordinal-loss --class-multipliers "1.0,1.0,1.0,1.0,1.0" --crop-min-scale 0.88 --rotation 8 --color-jitter 0.08 --blur-prob 0.04 --sharpness-prob 0.08
```

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | Plain CE |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7855 |
| Balanced accuracy | 0.6733 | 0.6213 |
| F1 macro | 0.6504 | 0.6261 |
| Kappa cuadratica | 0.8616 | 0.8448 |
| Recall grado 2 | 0.6333 | 0.6800 |
| Recall grado 3 | 0.5517 | 0.3448 |
| Recall grado 4 | 0.5682 | 0.5682 |
| Error grande | 0.0455 | 0.0709 |

Decision:

- No promover.
- Quitar sampler y perdida ordinal no sube accuracy y destruye parte de grado 3.
- No repetir esta configuracion.

### 20260629_213858_original_accuracy_mild_weights

Objetivo: subir accuracy con una variante menos agresiva que el baseline: pesos severos mas bajos, sampler mas suave y augmentations suaves.

Dataset usado:

- Entrenamiento: `data/splits/train_processed_300_split.csv`.
- Externo: no.
- `nuevodata`: no.

Comando:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --name original_accuracy_mild_weights --model-name efficientnet_b1 --batch-size 4 --train-csv data\splits\train_processed_300_split.csv --class-multipliers "1.0,1.0,1.05,1.10,1.08" --sampler-power 0.35 --ordinal-alpha 0.25 --crop-min-scale 0.88 --rotation 8 --color-jitter 0.08 --blur-prob 0.04 --sharpness-prob 0.08
```

Resultado: descartado y restaurado.

| Metrica | Baseline original-only | Mild weights |
|---|---:|---:|
| Accuracy | 0.7909 | 0.7764 |
| Balanced accuracy | 0.6733 | 0.6486 |
| F1 macro | 0.6504 | 0.6296 |
| Kappa cuadratica | 0.8616 | 0.8197 |
| Recall grado 2 | 0.6333 | 0.6467 |
| Recall grado 3 | 0.5517 | 0.4828 |
| Recall grado 4 | 0.5682 | 0.4545 |
| Error grande | 0.0455 | 0.0618 |

Decision:

- No promover.
- Bajar pesos/sampler tampoco sube accuracy y empeora clases severas.
- El baseline `original_only_b1` sigue siendo mejor incluso si el objetivo principal es accuracy.

## Flujo de revision manual del dataset

Objetivo: mejorar precision y confiabilidad sin meter ruido externo. La idea es revisar etiquetas/imagenes dudosas del dataset original, aplicar correcciones y reentrenar solo con datos originales depurados.

Archivos generados:

| Archivo | Uso |
|---|---|
| `results/dataset_review_queue.csv` | CSV editable donde se marcan correcciones/exclusiones |
| `results/dataset_review_queue.html` | Vista visual para revisar imagen procesada/original y calidad |
| `data/splits/train_processed_300_manual_review_split.csv` | Split de train curado cuando se aplican decisiones |
| `data/splits/val_processed_300_manual_review_split.csv` | Split de validacion curado |
| `data/splits/test_processed_300_manual_review_split.csv` | Split de test curado |
| `results/manual_review_apply_report.json` | Resumen de cambios aplicados |

Comandos:

```powershell
.\.venv\Scripts\python.exe create_dataset_review_queue.py --max-rows 300
.\.venv\Scripts\python.exe render_dataset_review_queue.py
```

Despues de revisar visualmente `results/dataset_review_queue.html`, editar `results/dataset_review_queue.csv`:

- `review_corrected_label`: usar `0`, `1`, `2`, `3` o `4` si la etiqueta actual esta mal.
- `review_exclude`: usar `1` si la imagen debe excluirse.
- `review_problem`: opcional, por ejemplo `etiqueta`, `calidad`, `encuadre`, `otro`.
- `review_notes`: notas libres.

Aplicar decisiones:

```powershell
.\.venv\Scripts\python.exe apply_manual_review.py
```

Estado inicial:

- Cola creada: 300 filas.
- Distribucion por split: train 298, val 2.
- Distribucion por etiqueta: grado 2 = 38, grado 3 = 92, grado 4 = 170.
- Aplicacion inicial sin marcas: 0 cambios, solo genera splits equivalentes para validar el flujo.

Decision:

- No reentrenar con splits curados hasta que haya correcciones/exclusiones reales.
- Esta es la via recomendada para intentar subir el modelo sin usar datasets externos.

## Politica de despliegue del modelo congelado

Decision: detener entrenamientos a ciegas. El checkpoint `original_only_b1` queda congelado como mejor modelo actual y se mejora la confiabilidad mediante abstencion/control de inferencia.

Cambios:

- `config/config.py`: agrega `AUTO_ACCEPT_CONFIDENCE`, `REVIEW_MIN_CONFIDENCE` y `REVIEW_MIN_TOP2_MARGIN`.
- `src/inference.py`: devuelve `policy_decision` con tres estados: `accept`, `review`, `reject`.
- `audit_inference_policy.py`: resume metricas por estado de politica.
- `DEPLOYMENT_POLICY.md`: documenta la politica operativa.
- `streamlit_app.py`: app visual con login demo, pacientes en memoria, diagnostico asistido, historial de sesion y descarga JSON.
- `.streamlit/secrets.example.toml`: ejemplo de credenciales para Streamlit Cloud.

Umbrales activos:

| Parametro | Valor |
|---|---:|
| `AUTO_ACCEPT_CONFIDENCE` | 0.80 |
| `REVIEW_MIN_CONFIDENCE` | 0.70 |
| `MIN_TOP2_MARGIN` | 0.18 |
| `REVIEW_MIN_TOP2_MARGIN` | 0.10 |

Auditoria con `audit_inference_policy.py` sobre test local:

| Estado | Casos | Cobertura | Accuracy si se usa | Error grande |
|---|---:|---:|---:|---:|
| `accept` | 219 | 0.3982 | 0.9772 | 0.0091 |
| `review` | 8 | 0.0145 | 0.8750 | 0.1250 |
| `reject` | 323 | 0.5873 | 0.6563 | 0.1176 |

Resumen:

- Predicciones aceptadas: 219/550.
- Accuracy aceptada: 0.9772.
- Error medio aceptado: 0.0320.
- Error grande aceptado: 0.0091.
- La politica acepta pocos casos y deriva/rechaza el resto, que son justamente los mas riesgosos si se forzara prediccion.

## App visual Streamlit

Objetivo: convertir el modelo congelado en una demo usable para GitHub/Streamlit sin base de datos.

Funciones:

- Login demo sin base de datos.
- HTML/CSS embebido para mejorar visualmente la app dentro de las limitaciones de Streamlit.
- Pacientes en memoria con `st.session_state`.
- Carga de imagen retinal.
- Vista previa grande de la imagen.
- Inferencia con `RetinaPredictor`.
- Decision visual `accept/review/reject`.
- Barra lateral con flujo y leyenda de decisiones.
- Ficha destacada del paciente activo.
- Interpretacion del grado predicho y accion sugerida.
- Calidad de imagen, confianza, top-2 y probabilidades por grado.
- Historial temporal por paciente.
- Descarga del resultado en JSON.

Comando local:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Verificacion:

- `streamlit_app.py` compila correctamente.
- Tests: `7 passed`.
- Servidor local Streamlit respondio HTTP 200 en `http://localhost:8501`.
- Se agrego `.streamlit/config.toml`, `packages.txt` y `STREAMLIT_DEPLOYMENT.md` para despliegue.

Notas para GitHub/Streamlit:

- `.streamlit/secrets.toml` queda ignorado.
- Usar `.streamlit/secrets.example.toml` como plantilla.
- `models/*.pth` esta ignorado; para despliegue usar Git LFS, descarga externa o carga manual del checkpoint.

## Cola de experimentos sugerida

1. `threshold_policy_tuning`: no cambia el checkpoint, ajusta decision clinica para aumentar precision aceptada y reducir errores grandes en predicciones aceptadas.
2. `manual_label_review`: revisar `results/manual_review_queue.csv` y corregir/excluir casos dudosos antes de reentrenar. Esta es la via con mas probabilidad real de subir el modelo sin mas datos.
3. `external_curation`: si se quiere usar externo, primero filtrar/curar ejemplos externos visualmente parecidos al dominio local; no mezclar ni refinar con externo bruto.

## Comandos utiles

Ver resumen de un experimento:

```powershell
Get-Content experiments\<carpeta>\comparison.json
```

Ver si el baseline actual sigue restaurado:

```powershell
Get-Content results\metrics.json
Get-Content results\clinical_audit.json
```

## App web FastAPI + HTML/CSS/JS para Render

Decision: usar repo normal con el modelo principal incluido, sin descarga externa en runtime.

Motivo:

- El checkpoint bueno pesa aprox. 71.26 MB, por debajo del limite normal de GitHub de 100 MB por archivo.
- El usuario quiere que la web funcione completa sin descargar el modelo aparte.
- Streamlit funciona, pero la interfaz principal necesita mejor UI/UX y mas control visual.

Cambios:

- Se creo `webapp/main.py` con FastAPI.
- Se creo `webapp/static/index.html`, `styles.css` y `app.js`.
- Se agrego `render.yaml` para Render.
- Se actualizo `.gitignore` para ignorar otros checkpoints pero permitir `models/best_retina_model.pth`.
- Se agrego `RENDER_DEPLOYMENT.md`.
- Se limpio `README.md` para reflejar la arquitectura actual.

Funciones implementadas:

- Login demo sin base de datos: `admin/admin123`.
- Creacion de pacientes en el navegador.
- Carga de imagen retinal.
- Inferencia con `RetinaPredictor`.
- Visualizacion de decision `accept/review/reject`.
- Calidad de imagen, probabilidades por grado e historial local.
- Descarga del resultado en JSON.

Verificacion local:

- `webapp/main.py` compila correctamente.
- Tests: `7 passed`.
- `/health` responde HTTP 200 y detecta `models/best_retina_model.pth` con 71.26 MB.
- `/` responde HTTP 200.
- `/predict` probado con `data/processed_512/000c1434d8d7.jpg`; la API devolvio resultado JSON y aplico politica `reject` por baja confianza/top-2 casi empatado.

## Rediseño UI inspirado en Sumaq Qhali

Decision: adaptar RetinaAI al lenguaje visual del proyecto Sumaq Qhali sin migrar a React/Vite.

Referencia local revisada:

- `C:\Users\efrai\Desktop\Sistemas\sumaq-qhali`
- `C:\Users\efrai\Desktop\Sistemas\sumaq qhali final`

Elementos tomados:

- Azul institucional oscuro `#00355f`.
- Cyan como acento secundario.
- Fondo claro tipo sistema clinico.
- Tarjetas blancas con sombra suave.
- Login tipo portal medico.
- Sidebar interno con perfil, modulos y estado del sistema.
- Dashboard compacto orientado a uso clinico, no landing comercial.

Cambios:

- `webapp/static/index.html` reestructurado con landing, login y portal con sidebar.
- `webapp/static/styles.css` reemplazado para acercarse al UI/UX de Sumaq Qhali.
- `webapp/static/app.js` ajustado para textos limpios y flujo directo de paciente a diagnostico.

Verificacion:

- `/` responde HTTP 200.
- `/static/styles.css` responde HTTP 200.
- `/static/app.js` responde HTTP 200.
- Tests: `7 passed`.
- Revision de IDs HTML/JS sin faltantes criticos; solo aparecen botones dinamicos creados por JS.

## Separacion login y dashboard

Problema detectado: el login y el panel vivian en el mismo `index.html`, ocultando y mostrando secciones. Visualmente se sentia como una sola pantalla y no como un sistema real.

Decision: separar paginas.

Cambios:

- `/` sirve `webapp/static/index.html` solo para landing/login.
- `/dashboard` sirve `webapp/static/dashboard.html` como panel interno.
- Login correcto redirige con `window.location.href = "/dashboard"`.
- Logout vuelve a `/`.
- `/dashboard` redirige a `/` si no hay sesion demo en `sessionStorage`.

Verificacion:

- `/` responde HTTP 200.
- `/dashboard` responde HTTP 200.
- Tests: `7 passed`.

## Navegacion premium sin sidebar

Problema detectado: el dashboard con barra lateral se sentia anticuado y pesado visualmente.

Decision: reemplazar sidebar por navegacion superior tipo app premium.

Cambios:

- `dashboard.html` ahora usa `app-navbar` sticky superior.
- Las secciones `Pacientes`, `Diagnostico` e `Historial` pasan a pestañas horizontales centrales.
- Estado del modelo, perfil demo y logout quedan a la derecha.
- Se agrega hero interno con resumen del modulo y paciente activo.
- Se eliminaron referencias al sidebar anterior.

Verificacion:

- `/dashboard` responde HTTP 200.
- IDs requeridos por `app.js` presentes en `dashboard.html`.
- Tests: `7 passed`.

## Rediseño del bloque de resultado

Problema detectado: el resultado se leia como una lista tecnica larga. Ademas, el texto `Analizando imagen...` podia quedar visible junto al resultado porque el CSS de `.placeholder` sobreescribia el atributo `hidden`.

Cambios:

- Se agrego `[hidden] { display: none !important; }`.
- El resultado ahora muestra primero una tarjeta resumen con decision, archivo y razones.
- Se agregaron chips de causa: confianza baja, prediccion ambigua, calidad baja, etc.
- Se convirtieron grado, confianza y margen top-2 en tarjetas compactas.
- La accion sugerida queda destacada en un bloque propio.
- Calidad de imagen y probabilidades pasan a secciones secundarias mas limpias.
- Probabilidades se muestran con barras compactas y etiquetas cortas.

Verificacion:

- `node --check webapp/static/app.js`: OK.
- `/static/styles.css` responde HTTP 200.
- `/predict` probado con imagen local y devuelve campos necesarios.
- Tests: `7 passed`.

## Pulido de resultado tras revision visual

Problema detectado en pantalla:

- Se mostraba una razon tecnica cruda: `image_quality_rejected`.
- El resultado seguia siendo largo para una lectura rapida.
- La distribucion por grado ocupaba demasiado espacio inicial.

Cambios:

- Se mapeo `image_quality_rejected` a `Calidad de imagen insuficiente`.
- Se filtran razones con `_` para evitar exponer codigos internos.
- Se eliminan duplicados y se limitan las razones visibles a 3 chips.
- La distribucion por grado queda cerrada por defecto en `details`.
- Las tarjetas de grado/confianza/margen se compactaron en una columna.
- Se mejoraron textos: confianza, margen y alternativa top-2.

Verificacion:

- `node --check webapp/static/app.js`: OK.
- `/static/app.js` responde HTTP 200.
- Tests: `7 passed`.

## Retiro de metricas internas para usuario final

Problema detectado: el dashboard mostraba `Accuracy checkpoint`, `Accuracy aceptada`, `Cobertura accept` y `Error grande accept`. Esas metricas son utiles para desarrollo, pero no para el usuario final optometrico.

Decision: ocultar metricas internas del modelo en la interfaz final.

Cambios:

- Se elimino la fila `metrics-row` del dashboard.
- Se ajusto el hero a lenguaje de optometria: `Panel optometrico` y `Evaluacion retinal asistida`.
- Se cambio el bloque de diagnostico de `Inferencia / Analisis retinal` a `Evaluacion / Imagen retinal`.
- La interfaz se enfoca en paciente, calidad de imagen, posible grado y accion recomendada.

Verificacion:

- Busqueda sin resultados para `Accuracy`, `metrics-row`, `checkpoint`, `Cobertura accept` y `Error grande` en archivos de interfaz.
- `/dashboard` responde HTTP 200.
- `node --check webapp/static/app.js`: OK.
- Tests: `7 passed`.

## Compactacion del flujo de carga de imagen

Problema detectado: el bloque `Evaluacion / Imagen retinal` y el upload ocupaban espacio adicional antes del panel de vista previa.

Decision: integrar la carga de imagen dentro del panel `Imagen retinal`.

Cambios:

- Se elimino el panel superior de instrucciones de carga.
- El area de placeholder dentro de `Imagen retinal` ahora es el control de seleccion de archivo.
- El texto visible queda reducido a `Seleccionar imagen retinal` y una linea de formato/calidad.
- Se agregaron estilos `image-upload` y `panel-heading compact`.

Verificacion:

- `imageInput`, `previewImage`, `imagePlaceholder`, `resultEmpty` y `resultContent` siguen presentes.
- `/dashboard` responde HTTP 200.
- `node --check webapp/static/app.js`: OK.
- Tests: `7 passed`.

## Implementacion Grad-CAM e informe actualizado

Decision: implementar Grad-CAM en el proyecto y actualizar el informe con la arquitectura real.

Cambios tecnicos:

- `src/inference.py` ahora incluye generacion Grad-CAM.
- Se usa la ultima capa convolucional de EfficientNet: `model.features[-1]`.
- Se calcula el gradiente de la clase predicha.
- Se genera heatmap y overlay sobre la imagen retinal preprocesada.
- El overlay se devuelve como PNG base64 dentro de `result.explanation`.
- `/predict` ahora llama `predict(..., include_explanation=True)`.
- La interfaz agrega tabs de imagen: `Original` y `Mapa IA`.
- El resultado muestra un aviso: Grad-CAM es ayuda visual, no explicacion clinica definitiva.

Cambios en informe:

- Se creo `construccion_actualizada.md` como version limpia y alineada con el sistema real.
- Se agrego una nota al inicio de `construcción.md` indicando que la version actualizada esta en `construccion_actualizada.md`.
- La version nueva documenta PyTorch, FastAPI, Render, politica `accept/review/reject`, metricas actuales y Grad-CAM implementado.

Verificacion:

- `src/inference.py` y `webapp/main.py` compilan correctamente.
- `node --check webapp/static/app.js`: OK.
- Grad-CAM probado con checkpoint real: devuelve `data:image/png;base64,...`.
- `/predict` probado por HTTP: devuelve `result.explanation.overlay`.
- Tests: `7 passed`.

## Comparacion visual Original vs Mapa IA

Decision: mostrar Grad-CAM al costado de la imagen original para facilitar comparacion visual.

Cambios:

- Se eliminaron los tabs `Original` / `Mapa IA`.
- `dashboard.html` ahora incluye `imageComparison`, `previewImage`, `gradcamFigure` y `gradcamImage`.
- Al cargar imagen se muestra solo la original.
- Al terminar `/predict`, si existe Grad-CAM, se muestra el mapa IA al costado.
- En pantallas pequenas la comparacion se apila verticalmente.

Verificacion:

- No quedan referencias a `imageModeTabs` ni `.image-mode`.
- IDs nuevos presentes en `dashboard.html`.
- `node --check webapp/static/app.js`: OK.
- `/predict` devuelve overlay Grad-CAM.
- `/dashboard` responde HTTP 200.
- Tests: `7 passed`.

## Limpieza de lenguaje para usuario final optometrico

Problema detectado: el resultado seguia usando terminos tecnicos o de desarrollo, como `modelo`, `Grad-CAM`, `Descargar JSON`, `confianza del modelo`, diferencia top-2 y detalles numericos visibles.

Decision: dejar la pantalla principal orientada a uso optometrico y mover detalles tecnicos a una seccion plegable.

Cambios:

- `Modelo verificando...` cambio a `Verificando sistema...`.
- `Modelo listo` cambio a `Sistema listo`.
- `Dr. RetinaAI / Sesion demo` cambio a `Profesional / Sesion activa`.
- `HCE retinal` cambio a `Evaluacion retinal`.
- `Mapa IA / Mapa Grad-CAM` cambio a `Mapa de atencion`.
- `Confianza del modelo` cambio a `Seguridad del resultado`.
- `Grado predicho` cambio a `Posible grado`.
- Se oculto el nombre de archivo del resumen principal.
- Se elimino el bloque explicativo redundante de Grad-CAM.
- Calidad numerica, segunda opcion, diferencia y distribucion por grado pasan a `Ver detalle tecnico`.
- `Guardar en historial` cambio a `Guardar resultado`.
- `Descargar JSON` cambio a `Descargar reporte`.

Verificacion:

- Busqueda sin resultados visibles para `Modelo verificando`, `Accuracy`, `checkpoint`, `Descargar JSON` y `Grad-CAM`.
- `node --check webapp/static/app.js`: OK.
- `/dashboard` responde HTTP 200.
- Tests: `7 passed`.
