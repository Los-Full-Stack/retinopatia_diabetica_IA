# RetinaAI

Proyecto de clasificacion de retinopatia diabetica con PyTorch, EfficientNetB0, politica clinica de abstencion y demo web lista para Render.

La version principal de producto usa FastAPI + HTML/CSS/JS puro. Streamlit queda como demo alternativa.

## Modelo actual

Checkpoint protegido:

```text
models/best_retina_model.pth
```

Tamano aproximado: `71.26 MB`.

Metricas principales del checkpoint:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.7909 |
| Balanced accuracy | 0.6733 |
| F1 macro | 0.6504 |
| F1 weighted | 0.7957 |
| Kappa | 0.8616 |
| ROC AUC multiclass | 0.9201 |
| Error grande >=2 grados | 0.0455 |

Politica de inferencia:

- `accept`: prediccion suficientemente confiable.
- `review`: requiere revision profesional.
- `reject`: repetir imagen o derivar.

Auditoria de politica sobre test local:

| Estado | Casos | Cobertura | Accuracy si se usa | Error grande |
|---|---:|---:|---:|---:|
| `accept` | 219 | 0.3982 | 0.9772 | 0.0091 |
| `review` | 8 | 0.0145 | 0.8750 | 0.1250 |
| `reject` | 323 | 0.5873 | 0.6563 | 0.1176 |

## App web principal

La app vive en `webapp/`.

Funciones:

- Login demo sin base de datos.
- Creacion de pacientes en el navegador.
- Carga de imagen retinal.
- Inferencia con `RetinaPredictor`.
- Decision visual `accept/review/reject`.
- Calidad de imagen y probabilidades por grado.
- Historial local por paciente.
- Descarga del resultado en JSON.

Credenciales demo:

```text
Usuario: admin
Clave: admin123
```

Ejecutar localmente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Abrir:

```text
http://127.0.0.1:8000
```

Endpoints:

- `/`: interfaz web.
- `/health`: estado del backend y modelo.
- `/predict`: diagnostico por imagen con `multipart/form-data`.

## Despliegue en Render

Render usa `render.yaml`:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
```

El repo ignora otros checkpoints, pero permite subir:

```text
models/best_retina_model.pth
```

Como pesa menos de 100 MB, puede subirse en un repo normal de GitHub sin Git LFS.

Guia completa:

```text
RENDER_DEPLOYMENT.md
```

## Instalacion en Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Comprobacion recomendada de CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sin GPU')"
```

## Flujo de entrenamiento

```powershell
python prepare_data.py
python train.py
python evaluate.py
python calibrate_model.py
python evaluate.py
python clinical_audit.py
python report_results.py
python predict.py --image "ruta\imagen.png"
```

Para entrenamiento completo cambia en `config/config.py`:

```python
TEST_MODE = False
```

## App Streamlit alternativa

La demo Streamlit vive en `streamlit_app.py`. Puede ejecutarse localmente, pero no es la interfaz principal para Render:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

No subas `.streamlit/secrets.toml` a GitHub. Usa `.streamlit/secrets.example.toml` como plantilla.

## Archivos importantes

- `src/inference.py`: inferencia y politica clinica.
- `config/config.py`: umbrales de confianza y margen.
- `audit_inference_policy.py`: auditoria de `accept/review/reject`.
- `DEPLOYMENT_POLICY.md`: politica de despliegue.
- `CLINICAL_READINESS.md`: limites clinicos.
- `EXPERIMENT_LOG.md`: decisiones y resultados historicos.
- `RENDER_DEPLOYMENT.md`: guia para subir a Render.

## Pruebas

```powershell
pytest
```

Ultima verificacion local:

- `7 passed`.
- `/health` responde HTTP 200 y detecta `models/best_retina_model.pth`.
- `/` responde HTTP 200.
- `/predict` probado con una imagen de `data/processed_512`.

## Nota medica

Este proyecto es un prototipo de investigacion/demo. No es un dispositivo medico aprobado y no debe usarse como decision clinica automatica sin validacion profesional.
