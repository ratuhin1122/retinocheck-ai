"""Grad-CAM explainability for fundus image classification and ordinal grading.

Supported Backbones & Target Layers:
- ResNet-18: `model.layer4[-1]` (final residual block in stage 4)
- EfficientNet-B0: `model.conv_head` (final 1x1 convolution before global pooling)

Important Clinical Note:
    Grad-CAM highlights gradient-weighted convolutional feature activation sensitivity
    (i.e., where in the image the network is looking). It DOES NOT represent clinical
    reasoning, causal diagnostic logic, or validated lesion segmentation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .config import CLASS_NAMES, Config
from .preprocess import build_transform, preprocess_image, preprocessing_options

# Clinical audit template for assessing Grad-CAM saliency against retinal pathology vs artifacts
ANALYSIS_TEMPLATE = """### Grad-CAM Saliency Audit: Pathology vs. Artifacts

> ⚠️ **Important Clinical Note**: Grad-CAM visualizes model attention (where convolutional
> gradients peak), NOT human clinical reasoning or validated medical evidence.
> Heatmaps should be audited carefully against true ophthalmological findings.

#### 1. Lesion Localization vs. Spurious Artifacts Checklist

| Audit Dimension | Legitimate Clinical Focus | Spurious Shortcut / Artifact |
| :--- | :--- | :--- |
| **Microvascular Lesions** | Heatmap focuses on focal microaneurysms, blot hemorrhages, or hard exudates in the macular/temporal arcades. | Heatmap focuses diffusely on uninformative, featureless retinal background. |
| **Optic Disc & Cup** | Physiological optic disc should not trigger severity activations unless Neovascularization of the Disc (NVD, Grade 4) is present. | Model over-attends to the high-contrast bright perimeter of the normal optic nerve head. |
| **Vascular Arcades** | Activations follow venous beading, loops, or Intraretinal Microvascular Abnormalities (IRMA, Grade 3). | Activations arbitrarily track normal, healthy arteriolar bifurcations. |
| **Perimeter / Lens Halo** | Circular camera aperture borders and dark outer padding should exhibit zero activation. | ⚠️ **Shortcut Artifact**: Saliency concentrates on the black border, camera lens halo, or dust artifacts. |
| **Illumination Vignetting** | Model accounts for uniform macular illumination. | Model exploits uneven lighting, flash glare, or peripheral sensor falloff. |

#### 2. Diagnostic Failure Mode Analysis (Audit of Incorrect Predictions)

1. **False Negatives on Early Disease (True: Mild / Grade 1 → Predicted: No DR / Grade 0)**:
   - *Mechanism*: Subtle microaneurysms (tiny punctate red lesions <125 µm) are easily lost or blurred during 224×224 downsampling.
   - *Audit Question*: Does Grad-CAM show zero activation, or did it attend to the lesion but fail to cross the decision threshold?
2. **Under-Grading Severe NPDR (True: Severe / Grade 3 → Predicted: Moderate / Grade 2)**:
   - *Mechanism*: Severe NPDR requires detecting the 4-2-1 rule (diffuse blot hemorrhages in all 4 quadrants, venous beading in 2+, or IRMA in 1+).
   - *Audit Question*: Is the model localizing lesions in only a single focal quadrant rather than recognizing multi-quadrant severity?
3. **Over-Calling Artifacts (True: No DR / Grade 0 → Predicted: Mild or Moderate)**:
   - *Mechanism*: High-frequency sensor noise, uneven illumination, or prominent choroidal vasculature mistaken for hemorrhages.
   - *Audit Question*: Do peak hotspots coincide with camera artifacts or non-retinal edge boundaries?
"""


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Return the final convolutional layer for Grad-CAM for ResNet-18 or EfficientNet-B0.

    - ResNet-18: `model.layer4[-1]` (the last BasicBlock of the 4th residual stage).
    - EfficientNet-B0: `model.conv_head` (the final 1x1 conv layer before pooling).

    Parameters
    ----------
    model:
        PyTorch model instance.
    model_name:
        Architecture identifier ('resnet18' or 'efficientnet_b0').

    Returns
    -------
    torch.nn.Module
        The target convolutional layer.
    """
    if model_name == "resnet18":
        if hasattr(model, "layer4"):
            return model.layer4[-1]
    elif model_name == "efficientnet_b0":
        if hasattr(model, "conv_head"):
            return model.conv_head
        elif hasattr(model, "blocks"):
            return model.blocks[-1]

    # Dynamic fallback inspection if model_name does not match directly
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    raise ValueError(f"Cannot identify target layer for model architecture: {model_name}")


def explain(
    model: torch.nn.Module,
    image: Image.Image,
    config: Config,
    target_class: Optional[int] = None,
    overlay_on: str = "original",
) -> np.ndarray:
    """Compute Grad-CAM activation heatmap and overlay it onto the fundus photograph.

    Note on Interpretation:
        Grad-CAM visualizes model attention (where convolutional activations respond most
        strongly to the chosen class/score). It does NOT validate causal clinical pathology
        or substitute for expert diagnostic examination.

    Parameters
    ----------
    model:
        Trained PyTorch model in eval mode.
    image:
        Input PIL Image (fundus photograph).
    config:
        Configuration object containing model_name, image_size, and mode.
    target_class:
        Target grade (0-4). In classification mode, defaults to the model's predicted
        class. In regression mode, target 0 represents the continuous severity output.
    overlay_on:
        Which base image to overlay the heatmap onto:
        - "original": overlays onto the original fundus photo resized to `image_size`.
        - "preprocessed": overlays onto the Ben Graham enhanced RGB image.
        - "raw": scales heatmap up to the input image's native unresized resolution.

    Returns
    -------
    np.ndarray
        RGB image with Grad-CAM heatmap overlaid (uint8 array, values in [0, 255]).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    device = next(model.parameters()).device
    tensor = build_transform(config)(image).unsqueeze(0).to(device)
    mode = getattr(config, "mode", "classification")

    if mode == "regression":
        # In regression mode, output is a single scalar; target index 0 computes gradients of that score
        targets = [ClassifierOutputTarget(0)]
    else:
        if target_class is None:
            with torch.no_grad():
                target_class = int(model(tensor).argmax(1).item())
        if target_class not in range(5):
            raise ValueError(f"Target class must be in 0-4, got: {target_class}")
        targets = [ClassifierOutputTarget(target_class)]

    target_layer = get_target_layer(model, config.model_name)

    # Note: Grad-CAM requires gradients, so do not wrap this block in torch.no_grad()
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=tensor, targets=targets)[0]  # shape: (H, W), values in [0, 1]

    # Select base image for the overlay
    if overlay_on == "preprocessed":
        base_rgb = np.asarray(preprocess_image(image, **preprocessing_options(config)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(base_rgb, heatmap, use_rgb=True)
    elif overlay_on == "raw":
        # Upscale heatmap to native uncropped/unresized image dimensions
        heatmap_full = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        base_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(base_rgb, heatmap_full, use_rgb=True)
    else:
        # Default: Original image resized to match the model's spatial resolution
        resized_orig = image.convert("RGB").resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
        base_rgb = np.asarray(resized_orig, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(base_rgb, heatmap, use_rgb=True)

    return overlay


def generate_gradcam_grid(
    model: torch.nn.Module,
    samples: List[Tuple[Image.Image, int, int]],
    config: Config,
    output_path: str = "reports/gradcam_examples.png",
) -> Path:
    """Generate a clean visual comparison grid of 8 correct and 8 incorrect predictions.

    Parameters
    ----------
    model:
        Trained model.
    samples:
        List of 16 tuples: (image, true_label, pred_label).
        First 8 should be correct predictions; next 8 should be incorrect predictions.
    config:
        Config instance.
    output_path:
        Destination file path for the saved composite PNG.

    Returns
    -------
    Path
        Path to the saved PNG image.
    """
    import matplotlib.pyplot as plt

    if len(samples) < 16:
        raise ValueError(f"Expected at least 16 samples (8 correct, 8 wrong), got {len(samples)}")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.patch.set_facecolor("white")

    # Titles for subsections
    fig.text(0.5, 0.985, "Grad-CAM Saliency: 8 Correct vs. 8 Incorrect Predictions", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.965, "Top 2 Rows: Correct Predictions (True == Pred)  |  Bottom 2 Rows: False Positives / False Negatives (True != Pred)", ha="center", fontsize=11, color="#444444")
    fig.text(0.5, 0.950, "Notice: Grad-CAM visualizes model attention, NOT clinical reasoning.", ha="center", fontsize=10, style="italic", color="#777777")

    for idx, (img, true_lbl, pred_lbl) in enumerate(samples[:16]):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        cam_overlay = explain(model, img, config)
        ax.imshow(cam_overlay)

        is_correct = (true_lbl == pred_lbl)
        color = "#1b7837" if is_correct else "#b2182b"
        status_tag = "✓ CORRECT" if is_correct else "✗ WRONG"

        title_text = (
            f"[{status_tag}]\n"
            f"True: {CLASS_NAMES[true_lbl].split(' - ')[-1]} ({true_lbl})\n"
            f"Pred: {CLASS_NAMES[pred_lbl].split(' - ')[-1]} ({pred_lbl})"
        )

        ax.set_title(title_text, fontsize=9.5, fontweight="bold", color=color, pad=6)
        ax.axis("off")

        # Visual divider between correct (rows 0, 1) and wrong (rows 2, 3)
        if row == 2 and col == 0:
            ax.figure.patches.extend([
                plt.Line2D([0.08, 0.92], [0.505, 0.505], transform=fig.transFigure, color="#888888", linestyle="--", linewidth=1.5)
            ])

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    return out_file
