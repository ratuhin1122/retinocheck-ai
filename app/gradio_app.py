"""Run from the repository root: python -m app.gradio_app --checkpoint ..."""
import argparse
import gradio as gr
import torch
from src.config import CLASS_NAMES
from src.model import load_checkpoint
from src.preprocess import make_transform
from src.explain import explain


def create_app(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_checkpoint(checkpoint_path, device)

    def classify(image):
        if image is None:
            raise gr.Error("Upload a fundus image for this research demo.")
        tensor = make_transform(config.image_size)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = model(tensor).softmax(1)[0].cpu().tolist()
        overlay = explain(model, image, config)
        return dict(zip(CLASS_NAMES, probabilities)), overlay

    demo = gr.Interface(fn=classify, inputs=gr.Image(type="pil", image_mode="RGB"),
        outputs=[gr.Label(num_top_classes=5, label="Model scores (uncalibrated)"),
                 gr.Image(label="Grad-CAM model sensitivity")],
        title="RetinoCheck AI — research prototype",
        description="Experimental grades 0–4 from fundus images. Never a medical device. "
                    "Not for diagnosis, screening, treatment decisions or medical advice. "
                    "Scores are uncalibrated; Grad-CAM is not evidence of pathology. "
                    "Use public or de-identified research images only.",
        flagging_mode="never")
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    create_app(args.checkpoint).launch(share=False)
