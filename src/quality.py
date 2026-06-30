from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from config.config import (
    MAX_IMAGE_BRIGHTNESS,
    MIN_IMAGE_BRIGHTNESS,
    MIN_IMAGE_CONTRAST,
    MIN_IMAGE_SHARPNESS,
    MIN_RETINA_COVERAGE,
    MIN_RETINA_QUALITY_SCORE,
)


@dataclass
class ImageQuality:
    brightness: float
    contrast: float
    sharpness: float
    retina_coverage: float
    quality_score: float
    acceptable: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _score_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def assess_retina_image_quality(image: Image.Image) -> ImageQuality:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    non_black = gray > 12
    retina_coverage = float(non_black.mean())

    brightness_score = 1.0
    if brightness < MIN_IMAGE_BRIGHTNESS:
        brightness_score = _score_range(brightness, 0.0, MIN_IMAGE_BRIGHTNESS)
    elif brightness > MAX_IMAGE_BRIGHTNESS:
        brightness_score = _score_range(255.0 - brightness, 0.0, 255.0 - MAX_IMAGE_BRIGHTNESS)

    contrast_score = _score_range(contrast, MIN_IMAGE_CONTRAST * 0.35, MIN_IMAGE_CONTRAST)
    sharpness_score = _score_range(sharpness, MIN_IMAGE_SHARPNESS * 0.35, MIN_IMAGE_SHARPNESS)
    coverage_score = _score_range(retina_coverage, MIN_RETINA_COVERAGE * 0.35, MIN_RETINA_COVERAGE)
    quality_score = float(
        0.25 * brightness_score
        + 0.25 * contrast_score
        + 0.25 * sharpness_score
        + 0.25 * coverage_score
    )

    warnings = []
    if brightness < MIN_IMAGE_BRIGHTNESS:
        warnings.append("Imagen demasiado oscura.")
    if brightness > MAX_IMAGE_BRIGHTNESS:
        warnings.append("Imagen demasiado clara o sobreexpuesta.")
    if contrast < MIN_IMAGE_CONTRAST:
        warnings.append("Contraste bajo.")
    if sharpness < MIN_IMAGE_SHARPNESS:
        warnings.append("Imagen borrosa.")
    if retina_coverage < MIN_RETINA_COVERAGE:
        warnings.append("Poca retina visible o encuadre insuficiente.")

    acceptable = quality_score >= MIN_RETINA_QUALITY_SCORE and not warnings
    return ImageQuality(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        retina_coverage=retina_coverage,
        quality_score=quality_score,
        acceptable=acceptable,
        warnings=warnings,
    )
