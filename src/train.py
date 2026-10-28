"""Mixed-precision T4 training with recoverable Drive checkpoints every epoch."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .config import Config
from .data import make_loaders
from .model import build_model
from .evaluate import metrics, predict
from .utils import seed_everything


def atomic_save(payload, path):
    """Replace only after serialization succeeds; keep the previous file on failure."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(config, resume=None):
    seed_everything(config.seed)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not resume and (root / "last.pt").exists():
        raise FileExistsError("Run already exists: resume last.pt or choose a new output_dir")
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint:
        # An extended epoch budget is allowed; optimizer/split settings must agree.
        for key, value in checkpoint["config"].items():
            if key not in ("epochs", "output_dir", "num_workers") and getattr(config, key) != value:
                raise ValueError(f"Resume configuration differs: {key}")
    loaders, train_labels, manifest = make_loaders(
        config, checkpoint["manifest"] if checkpoint else None)
    (root / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config.model_name, pretrained=checkpoint is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    weights = None
    if config.class_weights:
        counts = np.bincount(train_labels, minlength=5)
        weights = torch.tensor(len(train_labels) / (5 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    start, best, history = 0, -float("inf"), []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, history = checkpoint["epoch"] + 1, checkpoint["best_qwk"], checkpoint["history"]
        # Carry the best weights too, so a resumed run in another folder is complete.
        if checkpoint.get("best_model") is not None:
            best_payload = dict(checkpoint, model=checkpoint["best_model"])
            atomic_save(best_payload, root / "best.pt")
    best_model = checkpoint.get("best_model") if checkpoint else None
    for epoch in range(start, config.epochs):
        # Epoch-specific seeds make epoch-boundary restarts reproducible without
        # pickling NumPy RNG objects. GPU kernels may still vary across runtimes.
        seed_everything(config.seed + epoch)
        model.train()
        loss_sum, seen = 0.0, 0
        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; previous checkpoint is preserved")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        truth, predictions = predict(model, loaders["val"], device)
        result = metrics(truth, predictions)
        improved = result["qwk"] is not None and result["qwk"] > best
        if improved:
            best = result["qwk"]
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / seen, **result})
        payload = {"config": asdict(config), "manifest": manifest, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "best_qwk": best, "best_model": best_model, "history": history}
        # output_dir points directly to mounted Drive in the notebook.
        atomic_save(payload, root / "last.pt")
        if improved:
            atomic_save(payload, root / "best.pt")
        (root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(history[-1])
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(Config(model_name=args.model, output_dir=args.output_dir, epochs=args.epochs), args.resume)
