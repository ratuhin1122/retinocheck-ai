"""Root entrypoint for Hugging Face Spaces and local deployment.

When deployed to a Hugging Face Space (SDK: Gradio), Space runners execute
`python app.py` from the root of the repository. This file initializes the
RetinoCheck AI demo using CPU inference and optional model checkpoint loading
from the Hugging Face Hub.
"""
import os
from app.gradio_app import create_app

# Allow overriding model repository or checkpoint via environment variables
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", None)
PORT = int(os.environ.get("PORT", 7860))

# Create the application
demo = create_app(
    checkpoint_path=CHECKPOINT_PATH,
    hf_repo=HF_MODEL_REPO,
    device="cpu",  # Standard Hugging Face free-tier spaces run on CPU
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
    )

