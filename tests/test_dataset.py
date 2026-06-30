from pathlib import Path

import pandas as pd
from PIL import Image

from config.config import IMAGE_SIZE, NUM_CLASSES
from src.dataset import RetinaDataset
from src.preprocessing import get_eval_transforms


def test_dataset_loads_image(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 20, 40)).save(image_path)
    df = pd.DataFrame([{"id_code": "sample", "diagnosis": 2, "image_path": str(image_path)}])
    dataset = RetinaDataset(df, transform=get_eval_transforms())
    image, label, image_id, path = dataset[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert 0 <= label < NUM_CLASSES
    assert image_id == "sample"
    assert Path(path).exists()


def test_dataset_requires_columns(tmp_path):
    df = pd.DataFrame([{"id_code": "x", "diagnosis": 0}])
    try:
        RetinaDataset(df)
    except ValueError as exc:
        assert "image_path" in str(exc)
    else:
        raise AssertionError("Debió fallar por columna faltante.")
