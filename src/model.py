"""Both backbones use ImageNet pretrained weights and a five-class head."""
import timm
import torch
from .config import Config


def build_model(name, pretrained=True):
    if name not in ("resnet18", "efficientnet_b0"):
        raise ValueError(f"Unsupported model: {name}")
    return timm.create_model(name, pretrained=pretrained, num_classes=5)


def load_checkpoint(path, device="cpu"):
    # Only tensor/primitive checkpoints are written by this project.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    model = build_model(config.model_name, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint
