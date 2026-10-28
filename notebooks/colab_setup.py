# Paste this complete file into the first code cell of a fresh Colab session.
# Select Runtime > Change runtime type > T4 GPU before running.
from google.colab import drive
import os
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # Replace this.
REPO_DIR = Path("/content/retinocheck-ai")
DRIVE_ROOT = Path("/content/drive/MyDrive/RetinoCheckAI")

if "YOUR_USERNAME" in REPO_URL or "YOUR_REPO" in REPO_URL:
    raise ValueError("Set REPO_URL to your GitHub repository URL first.")
if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError("These pins target Python 3.11–3.13. Select a matching Colab runtime.")

drive.mount("/content/drive")
for folder in ("checkpoints", "reports"):
    (DRIVE_ROOT / folder).mkdir(parents=True, exist_ok=True)

# Reuse the clone when rerunning; avoid overwriting local changes with a pull.
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
elif not (REPO_DIR / ".git").exists():
    raise RuntimeError(f"{REPO_DIR} exists but is not a Git clone. Choose another REPO_DIR.")
else:
    remote = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "remote", "get-url", "origin"], text=True
    ).strip()
    if remote != REPO_URL:
        raise RuntimeError(f"Existing clone has a different origin: {remote}")
    print("Using existing clone. Commit:", subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True).strip())
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# CUDA 12.4 wheels: the official matching PyTorch 2.6 / torchvision 0.21 pair.
subprocess.run([sys.executable, "-m", "pip", "install",
                "torch==2.6.0", "torchvision==0.21.0",
                "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# A separate Python process checks the newly installed packages, even if the
# notebook kernel already imported older versions. Allocate a GPU tensor too.
subprocess.run([sys.executable, "-c", """
import torch, torchvision, timm, datasets, sklearn, gradio, pytorch_grad_cam
print('torch:', torch.__version__, '| torchvision:', torchvision.__version__)
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU. Select a T4 GPU runtime and rerun setup.')
print('GPU:', torch.cuda.get_device_name(0))
print('GPU tensor check:', (torch.ones(1, device='cuda') + 1).item())
"""], check=True)
print("Setup complete. If packages were already imported, restart the session before training.")
