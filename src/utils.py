import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config.config import LOGS_DIR, RANDOM_SEED


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_dirs(LOGS_DIR)
    logger = logging.getLogger("retina_ai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / "retina_ai.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_cuda_report(logger=None) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"PyTorch: {torch.__version__}",
        f"CUDA detectada por PyTorch: {torch.version.cuda}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.extend(
            [
                f"GPU: {torch.cuda.get_device_name(0)}",
                f"VRAM total: {props.total_memory / (1024 ** 3):.2f} GB",
            ]
        )
    else:
        lines.append("ADVERTENCIA: CUDA no está disponible. El entrenamiento en CPU será mucho más lento.")
    lines.append(f"Dispositivo utilizado: {device}")
    for line in lines:
        print(line)
        if logger:
            logger.info(line)
    return device


def persistent_workers_enabled(num_workers: int) -> bool:
    return num_workers > 0
