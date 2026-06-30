import numpy as np
from PIL import Image

from src.quality import assess_retina_image_quality


def test_black_image_is_not_acceptable():
    image = Image.new("RGB", (128, 128), color=(0, 0, 0))
    quality = assess_retina_image_quality(image)
    assert not quality.acceptable
    assert quality.retina_coverage == 0.0
    assert quality.warnings


def test_textured_image_returns_quality_metrics():
    data = np.random.default_rng(42).integers(30, 220, size=(128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(data, mode="RGB")
    quality = assess_retina_image_quality(image)
    assert quality.brightness > 0
    assert quality.contrast > 0
    assert quality.sharpness > 0
    assert 0 <= quality.quality_score <= 1
