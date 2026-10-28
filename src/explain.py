"""Grad-CAM highlights model sensitivity; it does not validate retinal findings."""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .preprocess import preprocess_image, preprocessing_options, build_transform


def explain(model, image, config, target_class=None):
    model.eval()
    device = next(model.parameters()).device
    tensor = build_transform(config)(image).unsqueeze(0).to(device)
    if target_class is None:
        with torch.no_grad():
            target_class = int(model(tensor).argmax(1).item())
    if target_class not in range(5):
        raise ValueError("Target class must be 0-4")
    layer = model.layer4[-1] if config.model_name == "resnet18" else model.conv_head
    # Do not wrap Grad-CAM in no_grad: its gradients are essential.
    with GradCAM(model=model, target_layers=[layer]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_class)])[0]
    rgb = np.asarray(preprocess_image(image, **preprocessing_options(config))) / 255.0
    return show_cam_on_image(rgb.astype(np.float32), heatmap, use_rgb=True)
