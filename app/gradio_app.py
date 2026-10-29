"""Gradio web application for RetinoCheck AI: Diabetic Retinopathy Assessment & Explainability.

Dual-mode deployment:
1. Google Colab: Run `python -m app.gradio_app --share`
2. Hugging Face Spaces: Run directly via root `app.py` (CPU-only, weight loading from Hub)

IMPORTANT CLINICAL DISCLAIMER:
    RetinoCheck AI is an experimental research demonstrator developed exclusively
    for educational and machine learning benchmarking purposes. It is NOT a certified
    medical device (FDA, CE, or equivalent). It must NEVER be used for clinical
    diagnosis, patient triage, screening, treatment planning, or any medical decision-making.
    Grad-CAM highlights convolutional gradient sensitivity (where the network attends),
    NOT validated clinical lesions or diagnostic rationale.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import CLASS_NAMES, Config

# Human-readable descriptive grade names
GRADE_SHORT_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}

# Clinical guidance mapped to international clinical diabetic retinopathy disease severity scales
# (For research demo illustration only - NOT valid medical advice)
FOLLOW_UP_RECOMMENDATIONS = {
    0: (
        "Routine annual diabetic retinal rescreening (12 months).\n"
        "Maintain optimal glycemic (HbA1c), blood pressure, and serum lipid control."
    ),
    1: (
        "Follow-up comprehensive dilated eye examination in 6–12 months.\n"
        "Review glycemic control, renal status, and cardiovascular risk factors with primary care."
    ),
    2: (
        "Specialist dilated retinal evaluation in 3–6 months.\n"
        "Timely referral to an ophthalmologist or retina specialist for closer surveillance."
    ),
    3: (
        "Urgent ophthalmology consultation in 2–4 months.\n"
        "High risk of progression to proliferative DR (4-2-1 rule). Evaluate for early panretinal "
        "photocoagulation (PRP) or anti-VEGF therapy."
    ),
    4: (
        "URGENT retina specialist referral within 24–48 hours.\n"
        "Immediate vision-threatening risk of vitreous hemorrhage, neovascular glaucoma, or tractional "
        "retinal detachment. Immediate prompt PRP and/or anti-VEGF evaluation indicated."
    ),
}

DISCLAIMER_BANNER = """
### ⚠️ IMPORTANT CLINICAL & REGULATORY DISCLAIMER
> **RESEARCH DEMO ONLY — NOT A MEDICAL DEVICE**
>
> This application is an experimental machine learning research demonstrator developed for benchmarking
> and educational purposes. It is **NOT** cleared or certified by the FDA, CE, or any health authority.
> It must **NEVER** be used for clinical diagnosis, patient triage, screening, treatment decisions, or medical advice.
>
> **Grad-CAM Attention**: The saliency overlay highlights convolutional feature sensitivity (where pixel gradients peak).
> It does **NOT** represent clinical reasoning, validated lesion segmentation, or proof of pathology.
> Always consult a board-certified ophthalmologist or licensed healthcare professional for retinal care.
"""


def is_running_in_colab() -> bool:
    """Detect whether code is running inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None


def resolve_checkpoint_path(
    checkpoint_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
) -> Optional[str]:
    """Find a local checkpoint or attempt downloading from the Hugging Face Hub."""
    # 1. Explicit path passed
    if checkpoint_path and Path(checkpoint_path).is_file():
        return str(Path(checkpoint_path).resolve())

    # 2. Environment variable
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path and Path(env_path).is_file():
        return str(Path(env_path).resolve())

    # 3. Known local candidate locations (best models prioritized)
    candidates = [
        "checkpoints/efficientnet_b0_regression/best.pt",
        "checkpoints/resnet18_regression/best.pt",
        "checkpoints/efficientnet_b0/best.pt",
        "checkpoints/resnet18/best.pt",
        "checkpoints/best.pt",
        "/content/drive/MyDrive/RetinoCheckAI/checkpoints/efficientnet_b0_regression/best.pt",
        "/content/drive/MyDrive/RetinoCheckAI/checkpoints/resnet18_regression/best.pt",
        "/content/drive/MyDrive/RetinoCheckAI/checkpoints/best.pt",
    ]
    for c in candidates:
        if Path(c).is_file():
            return str(Path(c).resolve())

    # 4. Attempt download from Hugging Face Hub if repository specified
    repo_to_try = hf_repo or os.environ.get("HF_MODEL_REPO")
    if repo_to_try:
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading checkpoint 'best.pt' from Hugging Face Hub: {repo_to_try}...")
            downloaded = hf_hub_download(repo_id=repo_to_try, filename="best.pt")
            if Path(downloaded).is_file():
                return downloaded
        except Exception as exc:
            print(f"Warning: Could not download from Hugging Face Hub ({exc}).")

    return None


def load_retinocheck_model(
    checkpoint_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Config, Optional[np.ndarray], bool]:
    """Load model, config, decision thresholds, and fallback status.

    Returns
    -------
    tuple of (model, config, thresholds, is_fallback)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.model import build_model, load_checkpoint

    resolved_path = resolve_checkpoint_path(checkpoint_path, hf_repo)

    if resolved_path is not None:
        try:
            print(f"Loading checkpoint from: {resolved_path}")
            model, config, checkpoint = load_checkpoint(resolved_path, device=device)
            # Extract saved optimal decision thresholds if regression mode
            thresholds = checkpoint.get("best_thresholds") or checkpoint.get("thresholds")
            if thresholds is not None:
                thresholds = np.array(thresholds, dtype=float)
            elif getattr(config, "mode", "classification") == "regression":
                thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=float)
            return model.to(device).eval(), config, thresholds, False
        except Exception as exc:
            print(f"Warning: Failed to load checkpoint {resolved_path}: {exc}. Using fallback model.")

    # Graceful Fallback: Pretrained EfficientNet-B0 with ImageNet weights in regression mode
    print("Initializing fallback model: efficientnet_b0 (mode=regression, ImageNet initialization).")
    config = Config(model_name="efficientnet_b0", mode="regression", ben_graham=True)
    model = build_model("efficientnet_b0", num_classes=5, mode="regression", pretrained=True)
    thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=float)
    return model.to(device).eval(), config, thresholds, True


def continuous_score_to_probabilities(
    score: float,
    sigma: float = 0.65,
    num_classes: int = 5,
) -> Dict[str, float]:
    """Convert a continuous regression severity score into a smooth 5-class confidence distribution.

    Uses a Gaussian proximity kernel: w_k = exp(-(score - k)^2 / (2 * sigma^2)),
    normalized so sum(w_k) == 1.0.
    """
    classes = np.arange(num_classes, dtype=float)
    weights = np.exp(-((score - classes) ** 2) / (2.0 * (sigma ** 2)))
    probs = weights / np.sum(weights)
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(num_classes)}


def create_app(
    checkpoint_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    device: Optional[str] = None,
):
    """Build and configure the Gradio Blocks application."""
    import gradio as gr
    from src.preprocess import build_transform
    from src.explain import explain, ANALYSIS_TEMPLATE

    model, config, thresholds, is_fallback = load_retinocheck_model(
        checkpoint_path=checkpoint_path, hf_repo=hf_repo, device=device
    )
    actual_device = next(model.parameters()).device

    mode = getattr(config, "mode", "classification")
    model_name = getattr(config, "model_name", "efficientnet_b0")

    def predict_fundus(input_image: Optional[Image.Image]):
        if input_image is None:
            raise gr.Error("Please upload a retinal fundus image.")

        # Ensure image is in RGB format
        image_rgb = input_image.convert("RGB")

        # 1. Apply standardized project preprocessing
        transform = build_transform(config)
        tensor = transform(image_rgb).unsqueeze(0).to(actual_device)

        # 2. Model inference
        with torch.no_grad():
            raw_output = model(tensor)

        if mode == "regression":
            raw_score = float(raw_output.squeeze().item())
            # Map continuous score to discrete grade via decision thresholds
            if thresholds is not None:
                pred_grade = int(np.clip(np.digitize([raw_score], thresholds)[0], 0, 4))
            else:
                pred_grade = int(np.clip(np.round(raw_score), 0, 4))

            confidence_dist = continuous_score_to_probabilities(raw_score, sigma=0.65)
            grade_summary = (
                f"Grade {pred_grade}: {GRADE_SHORT_NAMES[pred_grade]}\n"
                f"Continuous Severity Score: {raw_score:.2f} / 4.00"
            )
        else:
            probs = F.softmax(raw_output, dim=-1)[0].cpu().numpy()
            pred_grade = int(np.argmax(probs))
            confidence_dist = {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
            grade_summary = (
                f"Grade {pred_grade}: {GRADE_SHORT_NAMES[pred_grade]}\n"
                f"Class Confidence: {probs[pred_grade] * 100:.1f}%"
            )

        # 3. Grad-CAM saliency overlay
        try:
            cam_overlay = explain(model, image_rgb, config, overlay_on="original")
        except Exception as cam_err:
            print(f"Grad-CAM generation error: {cam_err}")
            # Fallback: return resized original image
            resized_orig = image_rgb.resize((config.image_size, config.image_size))
            cam_overlay = np.array(resized_orig)

        # 4. Clinical follow-up recommendation (research demo only)
        rec_text = (
            f"**Recommended Interval / Protocol (Research Simulation Only)**:\n\n"
            f"{FOLLOW_UP_RECOMMENDATIONS[pred_grade]}\n\n"
            f"> *Note: This automated suggestion reflects standard international diabetic eye guidelines for "
            f"illustrative benchmarking. It does NOT supersede clinical evaluation by an eye care specialist.*"
        )

        return grade_summary, confidence_dist, cam_overlay, rec_text

    # Gradio Blocks UI layout
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="RetinoCheck AI — Research Demo", theme=theme) as demo:
        gr.Markdown(
            """
            # 👁️ RetinoCheck AI
            ### Automated Diabetic Retinopathy Grading & Visual Attention Explainability
            """
        )

        # Prominent medical disclaimer banner
        gr.Markdown(DISCLAIMER_BANNER)

        if is_fallback:
            gr.Markdown(
                """
                > ℹ️ **Demonstration Mode Notice**: No trained model checkpoint was located on disk or Hub.
                > The app is running with an ImageNet-initialized backbone for interface demonstration.
                """
            )

        # Status badges
        status_md = (
            f"**Architecture**: `{model_name}` | **Mode**: `{mode.title()}` | "
            f"**Device**: `{actual_device.type.upper()}` | "
            f"**Resolution**: `{config.image_size}×{config.image_size}`"
        )
        if mode == "regression" and thresholds is not None:
            thr_str = ", ".join(f"{t:.2f}" for t in thresholds)
            status_md += f" | **Cut-points**: `[{thr_str}]`"
        gr.Markdown(status_md)

        with gr.Row():
            # Left column: input and controls
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Fundus Photograph (Macula / Disc Centered)",
                    sources=["upload", "clipboard"],
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(components=[image_input], value="Clear")
                    analyze_btn = gr.Button("Analyze Retinal Image", variant="primary")

                gr.Markdown(
                    """
                    **Supported Input**:
                    - Digital color fundus photographs (macula-centered or optic-disc-centered).
                    - Public or de-identified research datasets (e.g. APTOS 2019, Messidor, EyePACS).
                    - Never upload identifying patient health information (PHI).
                    """
                )

            # Right column: outputs
            with gr.Column(scale=1):
                grade_output = gr.Textbox(
                    label="Predicted DR Severity Grade",
                    interactive=False,
                    lines=2,
                )
                confidence_output = gr.Label(
                    num_top_classes=5,
                    label="Class Confidence Distribution (Uncalibrated)",
                )
                gradcam_output = gr.Image(
                    label="Grad-CAM Saliency Overlay (Model Sensitivity)",
                    interactive=False,
                )
                followup_output = gr.Markdown(
                    label="Suggested Follow-up (Research Demo Only)",
                )

        # Wire click event
        analyze_btn.click(
            fn=predict_fundus,
            inputs=[image_input],
            outputs=[grade_output, confidence_output, gradcam_output, followup_output],
        )

        # Expandable clinical audit documentation
        with gr.Accordion("📖 Clinical Audit Checklist & Grad-CAM Interpretation Guide", open=False):
            gr.Markdown(ANALYSIS_TEMPLATE)

        gr.Markdown(
            """
            ---
            **RetinoCheck AI Project** | Educational Machine Learning Benchmark |
            [Repository Documentation](https://github.com) |
            *Remember: Artificial intelligence outputs must never replace direct examination by a licensed ophthalmologist.*
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch RetinoCheck AI Gradio Application")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID to download weights from (e.g. 'username/retinocheck-ai')",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Generate a public Gradio share link (default True when running in Google Colab)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 7860)),
        help="Port to run the web server on (default 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the web server (default 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu', 'cuda', etc.)",
    )
    args = parser.parse_args()

    # Automatically set share=True if running inside Google Colab
    should_share = args.share or is_running_in_colab()

    demo = create_app(
        checkpoint_path=args.checkpoint,
        hf_repo=args.hf_repo,
        device=args.device,
    )
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=should_share,
    )


if __name__ == "__main__":
    main()
