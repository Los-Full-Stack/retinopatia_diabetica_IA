import torch

from src.model import build_model


def test_model_output_and_frozen_backbone():
    model = build_model(num_classes=5, freeze_backbone=True, pretrained=False)
    frozen = [p.requires_grad for p in model.features.parameters()]
    assert not any(frozen)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 5)
