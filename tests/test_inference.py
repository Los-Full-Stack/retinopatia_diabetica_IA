import pytest
import torch
from PIL import Image

from config.config import CLASS_NAMES, IMAGE_SIZE, MODEL_NAME, NORMALIZATION_MEAN, NORMALIZATION_STD
from src.inference import RetinaPredictor
from src.model import build_model


def make_checkpoint(path):
    model = build_model(num_classes=5, freeze_backbone=False, pretrained=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "model_name": MODEL_NAME,
            "num_classes": 5,
            "normalization_mean": NORMALIZATION_MEAN,
            "normalization_std": NORMALIZATION_STD,
        },
        path,
    )


def test_predictor_returns_probabilities(tmp_path):
    ckpt = tmp_path / "model.pth"
    make_checkpoint(ckpt)
    predictor = RetinaPredictor(checkpoint_path=ckpt, device=torch.device("cpu"))
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    result = predictor.predict(image)
    assert len(result["probabilities"]) == 5
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-5
    assert 0 <= result["class_id"] <= 4
    assert "image_quality" in result
    assert "prediction_accepted" in result
    assert "clinical_action" in result
    assert "second_class_id" in result
    assert "top2_margin" in result
    assert "required_confidence" in result
    assert "tta_enabled" in result


def test_predictor_missing_file_error(tmp_path):
    ckpt = tmp_path / "model.pth"
    make_checkpoint(ckpt)
    predictor = RetinaPredictor(checkpoint_path=ckpt, device=torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        predictor.predict(tmp_path / "missing.png")
