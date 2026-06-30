import torch
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
)

from config.config import MODEL_NAME, NUM_CLASSES


def _build_backbone(model_name: str, pretrained: bool):
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    if model_name == "efficientnet_b1":
        return efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None)
    if model_name == "efficientnet_b2":
        return efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None)
    if model_name == "efficientnet_b3":
        return efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
    raise ValueError(f"Modelo no soportado: {model_name}")


def build_model(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    model_name: str = MODEL_NAME,
) -> torch.nn.Module:
    model = _build_backbone(model_name, pretrained)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.35, inplace=True),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_last_blocks(model: torch.nn.Module, num_blocks: int = 2) -> None:
    for param in model.features.parameters():
        param.requires_grad = False
    for block in list(model.features.children())[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
