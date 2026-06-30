from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config.config import MODELS_DIR
from src.inference import RetinaPredictor


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = MODELS_DIR / "best_retina_model.pth"

app = FastAPI(
    title="RetinaAI",
    description="RetinaAI web demo with FastAPI and static HTML/CSS/JS.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_predictor: RetinaPredictor | None = None


def get_predictor() -> RetinaPredictor:
    global _predictor
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No se encontro el modelo requerido: {MODEL_PATH}",
        )
    if _predictor is None:
        _predictor = RetinaPredictor(checkpoint_path=MODEL_PATH)
    return _predictor


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_PATH.exists() else "missing_model",
        "model_path": str(MODEL_PATH),
        "model_size_mb": round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if MODEL_PATH.exists() else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Solo se aceptan imagenes JPG o PNG.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {exc}") from exc

    predictor = get_predictor()
    result = predictor.predict(image, include_explanation=True)
    return {
        "filename": file.filename,
        "result": result,
    }
