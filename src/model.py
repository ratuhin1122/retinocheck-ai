"""Backbone factory for classification (5-head) and regression (1-head) modes.

Supported architectures
-----------------------
- ``resnet18``       – ResNet-18, lightweight baseline.
- ``efficientnet_b0``– EfficientNet-B0, stronger baseline at similar speed.

Modes
-----
- ``"classification"`` – softmax head with ``num_classes`` outputs (default 5).
- ``"regression"``     – single scalar output for ordinal grading; grades are
  recovered at inference time by rounding to the nearest integer in [0, 4].
"""
import timm
import torch
import torch.nn as nn
from .config import Config

_SUPPORTED = ("resnet18", "efficientnet_b0")


def build_model(name: str, num_classes: int = 5, mode: str = "classification",
                pretrained: bool = True) -> nn.Module:
    """Return a timm backbone with a task-appropriate head.

    Parameters
    ----------
    name:
        One of ``"resnet18"`` or ``"efficientnet_b0"``.
    num_classes:
        Number of output neurons for classification mode (ignored in regression
        mode, which always uses a single output).
    mode:
        ``"classification"`` for cross-entropy training or ``"regression"`` for
        MSE / smooth-L1 ordinal training.
    pretrained:
        Load ImageNet weights when ``True`` (default).  Pass ``False`` when
        restoring a saved checkpoint.

    Returns
    -------
    nn.Module
        The backbone with the requested head already attached.
    """
    if name not in _SUPPORTED:
        raise ValueError(f"Unsupported model: {name!r}.  Choose from {_SUPPORTED}.")
    if mode not in ("classification", "regression"):
        raise ValueError(f"Unknown mode: {mode!r}.  Use 'classification' or 'regression'.")

    out_features = 1 if mode == "regression" else num_classes
    model = timm.create_model(name, pretrained=pretrained, num_classes=out_features)
    return model


def load_checkpoint(path, device="cpu"):
    """Restore a model, its ``Config``, and the raw checkpoint dict from *path*.

    Supports checkpoints saved by both the original training loop (classification
    only, no ``mode`` key) and the updated loop that stores ``mode``.
    """
    # Only tensor/primitive checkpoints are written by this project.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    settings = dict(checkpoint["config"])
    # Older project checkpoints used crop/resize without enhancement.
    settings.setdefault("ben_graham", False)
    # Older checkpoints did not record mode; default to classification.
    mode = settings.pop("mode", "classification")
    config = Config(**settings)
    num_classes = settings.get("num_classes", 5)
    model = build_model(config.model_name, num_classes=num_classes,
                        mode=mode, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint
