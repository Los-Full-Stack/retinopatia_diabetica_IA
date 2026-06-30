# Despliegue en Render

Esta es la ruta simple: repo normal en GitHub, modelo incluido y app servida por FastAPI.

## Que se sube

- `webapp/main.py`: backend FastAPI.
- `webapp/static/`: interfaz HTML/CSS/JS.
- `models/best_retina_model.pth`: checkpoint bueno congelado.
- `requirements.txt`: dependencias Python.
- `render.yaml`: configuracion de Render.
- `packages.txt`: librerias del sistema para OpenCV/Pillow en Linux.

El archivo del modelo pesa aprox. 71.26 MB, asi que cabe en GitHub sin Git LFS mientras no supere 100 MB.

## Comandos locales

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Abrir:

```text
http://127.0.0.1:8000
```

Credenciales demo:

```text
Usuario: admin
Clave: admin123
```

## Crear repo normal

Esta carpeta ya fue inicializada con Git. Para crear el primer commit y subir a GitHub:

```powershell
git status --short
git add .
git commit -m "Initial RetinaAI web deployment"
git branch -M main
git remote add origin <URL_DEL_REPO>
git push -u origin main
```

Antes de hacer commit, confirma que el checkpoint permitido aparezca:

```powershell
git status --short
```

Debe aparecer `models/best_retina_model.pth`. Otros checkpoints, datasets, zips, resultados y logs quedan ignorados.

## Configuracion en Render

1. Crear un nuevo `Web Service`.
2. Conectar el repo de GitHub.
3. Render detectara `render.yaml`.
4. Confirmar:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
Python: 3.11.9
```

## Endpoints

- `/`: interfaz web.
- `/health`: estado basico del backend y ruta/tamano del modelo.
- `/predict`: diagnostico por imagen retinal mediante `multipart/form-data`.

## Decision clinica

La app no fuerza todas las predicciones. Usa la politica:

- `accept`: prediccion suficientemente confiable.
- `review`: requiere revision profesional.
- `reject`: repetir imagen o derivar, porque la confianza o calidad no es suficiente.

Esto protege el modelo en produccion demo: el accuracy general del checkpoint no cambia, pero las predicciones aceptadas son mucho mas confiables.
