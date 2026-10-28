"""Reproducibility helpers shared by notebooks and training."""
import os
import random

# Set before CUDA initializes, including when this module is imported first.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def seed_everything(seed=42):
    """Seed Python, NumPy and PyTorch before creating models and data loaders.

    Deterministic settings improve repeatability within the same environment.
    Exact equality across GPU types or library versions is not guaranteed.
    warn_only permits operations that lack a deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
