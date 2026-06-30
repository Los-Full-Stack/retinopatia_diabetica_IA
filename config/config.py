import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "train.csv"
ZIP_PATH = PROJECT_ROOT / "train_images.zip"

EXTRACT_DIR = PROJECT_ROOT / "data" / "extracted"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if not value:
        return default
    return [float(item.strip()) for item in value.split(",")]


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value else default


IMAGE_SIZE = _env_int("RETINAAI_IMAGE_SIZE", 300)
PROCESSED_DIR = PROJECT_ROOT / "data" / f"processed_{IMAGE_SIZE}"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPORT_DIR = PROJECT_ROOT / "export"

BATCH_SIZE = _env_int("RETINAAI_BATCH_SIZE", 4)
NUM_CLASSES = 5
NUM_WORKERS = 2
RANDOM_SEED = _env_int("RETINAAI_RANDOM_SEED", 42)

INITIAL_EPOCHS = _env_int("RETINAAI_INITIAL_EPOCHS", 10)
FINE_TUNE_EPOCHS = _env_int("RETINAAI_FINE_TUNE_EPOCHS", 10)

LEARNING_RATE = _env_float("RETINAAI_LEARNING_RATE", 1e-3)
FINE_TUNE_LEARNING_RATE = _env_float("RETINAAI_FINE_TUNE_LEARNING_RATE", 2e-5)

EARLY_STOPPING_PATIENCE = 6
GRADIENT_CLIP_VALUE = 1.0

MODEL_NAME = _env_str("RETINAAI_MODEL_NAME", "efficientnet_b1")

TEST_MODE = False
TEST_SAMPLE_SIZE = 500

FORCE_REEXTRACT = False
CHECK_ALL_IMAGES = True
REUSE_SPLITS = False

USE_RETINAL_PREPROCESSING = True
USE_PROCESSED_IMAGES = True
USE_RAW_SPLITS = _env_bool("RETINAAI_USE_RAW_SPLITS", False)
PROCESSED_IMAGE_QUALITY = 92
USE_WEIGHTED_SAMPLER = _env_bool("RETINAAI_USE_WEIGHTED_SAMPLER", True)
USE_EXTERNAL_TRAINING_DATA = _env_bool("RETINAAI_USE_EXTERNAL_TRAINING_DATA", False)
EXTERNAL_TRAINING_CSV = Path(os.environ.get("RETINAAI_EXTERNAL_TRAINING_CSV", SPLITS_DIR / "train_with_external_processed_300_split.csv"))
SAMPLER_WEIGHT_POWER = _env_float("RETINAAI_SAMPLER_WEIGHT_POWER", 0.5)
CLASS_WEIGHT_MULTIPLIERS = _env_float_list("RETINAAI_CLASS_WEIGHT_MULTIPLIERS", [1.0, 1.0, 1.15, 1.35, 1.25])
FINE_TUNE_BLOCKS = _env_int("RETINAAI_FINE_TUNE_BLOCKS", 4)
FOCAL_LOSS_ENABLED = _env_bool("RETINAAI_FOCAL_LOSS_ENABLED", False)
FOCAL_LOSS_GAMMA = _env_float("RETINAAI_FOCAL_LOSS_GAMMA", 2.0)
ORDINAL_LOSS_ENABLED = _env_bool("RETINAAI_ORDINAL_LOSS_ENABLED", True)
ORDINAL_LOSS_ALPHA = _env_float("RETINAAI_ORDINAL_LOSS_ALPHA", 0.35)
CHECKPOINT_MONITOR = "composite"
USE_TTA = True
TRAIN_RANDOM_RESIZED_CROP_MIN_SCALE = _env_float("RETINAAI_TRAIN_CROP_MIN_SCALE", 0.85)
TRAIN_RANDOM_RESIZED_CROP_MAX_SCALE = _env_float("RETINAAI_TRAIN_CROP_MAX_SCALE", 1.0)
TRAIN_ROTATION_DEGREES = _env_float("RETINAAI_TRAIN_ROTATION_DEGREES", 10.0)
TRAIN_COLOR_JITTER = _env_float("RETINAAI_TRAIN_COLOR_JITTER", 0.12)
TRAIN_GAUSSIAN_BLUR_PROB = _env_float("RETINAAI_TRAIN_GAUSSIAN_BLUR_PROB", 0.12)
TRAIN_SHARPNESS_PROB = _env_float("RETINAAI_TRAIN_SHARPNESS_PROB", 0.15)
CALIBRATION_PATH = RESULTS_DIR / "calibration.json"
MIN_PREDICTION_CONFIDENCE = 0.70
AUTO_ACCEPT_CONFIDENCE = _env_float("RETINAAI_AUTO_ACCEPT_CONFIDENCE", 0.80)
REVIEW_MIN_CONFIDENCE = _env_float("RETINAAI_REVIEW_MIN_CONFIDENCE", 0.70)
MIN_TOP2_MARGIN = 0.18
REVIEW_MIN_TOP2_MARGIN = _env_float("RETINAAI_REVIEW_MIN_TOP2_MARGIN", 0.10)
MIN_CONFIDENCE_BY_CLASS = [0.78, 0.70, 0.70, 0.72, 0.74]
ENABLE_TTA_INFERENCE = True
MIN_RETINA_QUALITY_SCORE = 0.55
MIN_RETINA_COVERAGE = 0.18
MIN_IMAGE_SHARPNESS = 18.0
MIN_IMAGE_CONTRAST = 18.0
MIN_IMAGE_BRIGHTNESS = 18.0
MAX_IMAGE_BRIGHTNESS = 238.0

SCORE_F1_MACRO_WEIGHT = 0.30
SCORE_BALANCED_ACCURACY_WEIGHT = 0.25
SCORE_KAPPA_WEIGHT = 0.25
SCORE_SEVERE_RECALL_WEIGHT = 0.20

CLASS_NAMES = [
    "Sin retinopatía diabética",
    "Retinopatía diabética leve",
    "Retinopatía diabética moderada",
    "Retinopatía diabética severa",
    "Retinopatía diabética proliferativa",
]

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
