import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from config.config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    TRAIN_COLOR_JITTER,
    TRAIN_GAUSSIAN_BLUR_PROB,
    TRAIN_RANDOM_RESIZED_CROP_MAX_SCALE,
    TRAIN_RANDOM_RESIZED_CROP_MIN_SCALE,
    TRAIN_ROTATION_DEGREES,
    TRAIN_SHARPNESS_PROB,
    USE_RETINAL_PREPROCESSING,
)


class RetinalPreprocess:
    def __init__(self, enabled: bool = USE_RETINAL_PREPROCESSING):
        self.enabled = enabled

    def __call__(self, image):
        if not self.enabled:
            return image.convert("RGB")
        rgb = np.array(image.convert("RGB"))
        cropped = self._crop_black_borders(rgb)
        enhanced = self._apply_clahe(cropped)
        return Image.fromarray(enhanced)

    @staticmethod
    def _crop_black_borders(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > 10
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = image[y0:y1, x0:x1]
        if cropped.shape[0] < 32 or cropped.shape[1] < 32:
            return image
        return cropped

    @staticmethod
    def _apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_retina_image(image, output_size: int = IMAGE_SIZE):
    image = RetinalPreprocess(enabled=True)(image)
    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def get_train_transforms(retinal_preprocess: bool = USE_RETINAL_PREPROCESSING):
    return transforms.Compose(
        [
            RetinalPreprocess(enabled=retinal_preprocess),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(TRAIN_RANDOM_RESIZED_CROP_MIN_SCALE, TRAIN_RANDOM_RESIZED_CROP_MAX_SCALE),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=TRAIN_ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=TRAIN_COLOR_JITTER,
                contrast=TRAIN_COLOR_JITTER,
                saturation=min(TRAIN_COLOR_JITTER, 0.10),
                hue=min(TRAIN_COLOR_JITTER / 4, 0.03),
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=TRAIN_GAUSSIAN_BLUR_PROB),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=TRAIN_SHARPNESS_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


def get_eval_transforms(retinal_preprocess: bool = USE_RETINAL_PREPROCESSING):
    return transforms.Compose(
        [
            RetinalPreprocess(enabled=retinal_preprocess),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )
