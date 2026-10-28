"""Fixed-label metrics and saved test reports; no test-driven model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from .model import load_checkpoint
from .data import make_loaders


def metrics(y_true, y_pred):
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=list(range(5)), weights="quadratic"))
    return {"qwk": kappa if np.isfinite(kappa) else None,
            "sample_count": len(y_true),
            "class_counts": np.bincount(np.asarray(y_true, dtype=int), minlength=5).tolist(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(5)),
                                       average="macro", zero_division=0)),
            "per_class_recall": recall_score(y_true, y_pred, labels=list(range(5)),
                                             average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for images, labels in loader:
        predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
        truth.extend(labels.tolist())
    return truth, predictions


def evaluate(checkpoint_path, output_dir="reports", split="test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, checkpoint = load_checkpoint(checkpoint_path, device)
    loaders, _, _ = make_loaders(config, checkpoint["manifest"])
    truth, predictions = predict(model, loaders[split], device)
    result = metrics(truth, predictions)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / f"{split}_predictions.json").write_text(json.dumps({
        "indices": checkpoint["manifest"]["indices"][split],
        "truth": truth, "predictions": predictions}), encoding="utf-8")
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=range(5)).plot()
    plt.title(f"{config.model_name}: {split} (rows=true, columns=predicted)")
    plt.tight_layout()
    plt.savefig(root / f"{split}_confusion_matrix.png", dpi=160)
    plt.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    print(evaluate(args.checkpoint, args.output_dir))
