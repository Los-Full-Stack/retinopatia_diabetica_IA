import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config.config import (
    AUTO_ACCEPT_CONFIDENCE,
    CALIBRATION_PATH,
    CLASS_NAMES,
    ENABLE_TTA_INFERENCE,
    IMAGE_SIZE,
    MIN_CONFIDENCE_BY_CLASS,
    MIN_PREDICTION_CONFIDENCE,
    MIN_TOP2_MARGIN,
    MODEL_NAME,
    MODELS_DIR,
    NUM_CLASSES,
    REVIEW_MIN_CONFIDENCE,
    REVIEW_MIN_TOP2_MARGIN,
)
from src.model import build_model
from src.preprocessing import RetinalPreprocess, get_eval_transforms
from src.quality import assess_retina_image_quality
from src.utils import load_json


class RetinaPredictor:
    def __init__(self, checkpoint_path=None, device=None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else MODELS_DIR / "best_retina_model.pth"
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {self.checkpoint_path}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", CLASS_NAMES)
        self.use_processed_preprocess = bool(checkpoint.get("use_processed_images", False))
        self.model = build_model(
            num_classes=checkpoint.get("num_classes", NUM_CLASSES),
            freeze_backbone=False,
            pretrained=False,
            model_name=checkpoint.get("model_name", MODEL_NAME),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transforms = get_eval_transforms(retinal_preprocess=self.use_processed_preprocess)
        self.temperature = self._load_temperature()

    def _load_temperature(self) -> float:
        if not CALIBRATION_PATH.exists():
            return 1.0
        try:
            data = load_json(CALIBRATION_PATH)
            temperature = float(data.get("temperature", 1.0))
        except Exception:
            return 1.0
        return max(temperature, 1e-3)

    def _load_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"No existe la imagen: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _gradcam(self, pil_image: Image.Image, target_class: int | None = None) -> dict:
        display_image = RetinalPreprocess(enabled=self.use_processed_preprocess)(pil_image)
        display_image = display_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).convert("RGB")
        tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

        activations = None
        gradients = None

        def forward_hook(_module, _inputs, output):
            nonlocal activations
            activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        target_layer = self.model.features[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[:, target_class].sum()
            score.backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        if activations is None or gradients is None:
            raise RuntimeError("No se pudo calcular Grad-CAM para la imagen.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        if float(cam.max()) > 0:
            cam = cam / float(cam.max())
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        base = np.array(display_image).astype(np.float32)
        overlay = np.clip((0.58 * base) + (0.42 * heatmap_color.astype(np.float32)), 0, 255).astype(np.uint8)

        return {
            "method": "Grad-CAM",
            "target_class_id": int(target_class),
            "target_layer": "model.features[-1]",
            "overlay": self._image_to_data_url(Image.fromarray(overlay)),
            "heatmap": self._image_to_data_url(Image.fromarray(heatmap_color)),
            "note": "Mapa de atencion del modelo; no equivale a una explicacion clinica definitiva.",
        }

    def predict(self, image, include_explanation: bool = False) -> dict:
        pil_image = self._load_image(image)
        quality = assess_retina_image_quality(pil_image)
        tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if ENABLE_TTA_INFERENCE:
                logits_flip = self.model(torch.flip(tensor, dims=[3]))
                logits = (logits + logits_flip) / 2
            probs = torch.softmax(logits / self.temperature, dim=1).squeeze(0).detach().cpu()

        top_probs, top_indices = torch.topk(probs, k=2)
        class_id = int(top_indices[0].item())
        second_class_id = int(top_indices[1].item())
        confidence = float(probs[class_id].item())
        second_confidence = float(top_probs[1].item())
        top2_margin = float(top_probs[0].item() - top_probs[1].item())

        class_threshold = (
            float(MIN_CONFIDENCE_BY_CLASS[class_id])
            if class_id < len(MIN_CONFIDENCE_BY_CLASS)
            else float(MIN_PREDICTION_CONFIDENCE)
        )
        review_confidence = max(float(REVIEW_MIN_CONFIDENCE), float(MIN_PREDICTION_CONFIDENCE))
        required_confidence = max(float(AUTO_ACCEPT_CONFIDENCE), class_threshold, review_confidence)

        warnings = list(quality.warnings)
        policy_reasons = []
        if not quality.acceptable:
            policy_reasons.append("image_quality_rejected")
        if confidence < review_confidence:
            policy_reasons.append("confidence_below_review_threshold")
        elif confidence < required_confidence:
            policy_reasons.append("confidence_requires_review")
        if top2_margin < REVIEW_MIN_TOP2_MARGIN:
            policy_reasons.append("top2_margin_too_low")
        elif top2_margin < MIN_TOP2_MARGIN:
            policy_reasons.append("top2_margin_requires_review")

        if not quality.acceptable or confidence < review_confidence or top2_margin < REVIEW_MIN_TOP2_MARGIN:
            policy_decision = "reject"
            clinical_action = "repeat_image_or_refer"
        elif confidence >= required_confidence and top2_margin >= MIN_TOP2_MARGIN:
            policy_decision = "accept"
            clinical_action = "auto_prediction_allowed"
        else:
            policy_decision = "review"
            clinical_action = "specialist_review_recommended"

        prediction_accepted = policy_decision == "accept"
        if confidence < required_confidence:
            warnings.append("Confianza insuficiente para aceptacion automatica.")
        if top2_margin < MIN_TOP2_MARGIN:
            warnings.append("Prediccion ambigua: las dos clases mas probables estan muy cerca.")
        if policy_decision == "reject":
            warnings.append("No usar como decision automatica; repetir captura o derivar.")
        elif policy_decision == "review":
            warnings.append("Prediccion orientativa; requiere revision.")

        result = {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": confidence,
            "second_class_id": second_class_id,
            "second_class_name": self.class_names[second_class_id],
            "second_confidence": second_confidence,
            "top2_margin": top2_margin,
            "required_confidence": required_confidence,
            "review_confidence": review_confidence,
            "auto_accept_confidence": float(AUTO_ACCEPT_CONFIDENCE),
            "required_top2_margin": float(MIN_TOP2_MARGIN),
            "review_top2_margin": float(REVIEW_MIN_TOP2_MARGIN),
            "probabilities": [float(x) for x in probs.tolist()],
            "temperature": self.temperature,
            "tta_enabled": ENABLE_TTA_INFERENCE,
            "image_quality": quality.to_dict(),
            "policy_decision": policy_decision,
            "policy_reasons": policy_reasons,
            "prediction_accepted": prediction_accepted,
            "clinical_action": clinical_action,
            "warnings": warnings,
        }
        if include_explanation:
            try:
                result["explanation"] = self._gradcam(pil_image, target_class=class_id)
            except Exception as exc:
                result["explanation"] = {
                    "method": "Grad-CAM",
                    "available": False,
                    "error": str(exc),
                }
        return result
