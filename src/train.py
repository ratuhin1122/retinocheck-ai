"""Mixed-precision training with cosine LR, early stopping, and CSV logging.

Modes
-----
``"classification"``
    Five-output head trained with ``CrossEntropyLoss`` (optionally
    class-weighted).  Predictions are the argmax of the softmax.

``"regression"``
    One-output head trained with ``SmoothL1Loss``.  Raw scalar outputs are
    rounded to the nearest integer in [0, num_classes-1] to produce grade
    predictions.  An optional threshold optimiser searches for cut-points on
    the validation set that maximise QWK.

Imbalance handling (mutually exclusive; set via Config)
-------------------------------------------------------
- ``config.class_weights=True, config.use_weighted_sampler=False``:
  Class-weighted loss (default).  Works for both modes.
- ``config.use_weighted_sampler=True``:
  ``WeightedRandomSampler`` replaces shuffle=True on the train loader so
  every mini-batch is approximately balanced; loss is unweighted.

Checkpoints
-----------
Two files are written atomically to ``config.output_dir`` every epoch:

- ``last.pt``  – the most recent state (for resuming).
- ``best.pt``  – snapshot at the epoch with the highest validation QWK.

Both files are self-contained: they embed the model weights, the frozen
data-split manifest, and the full training history so that a resumed run
or an evaluation script never needs a separate config file.

Early stopping
--------------
Training halts automatically when QWK has not improved for
``config.patience`` consecutive epochs (0 = disabled).

CSV logging
-----------
``history.csv`` is written to ``config.output_dir`` after every epoch with
columns: epoch, train_loss, val_loss, qwk, thresholds (regression only).

Regularisation
--------------
Two orthogonal regularisation mechanisms are supported and may be combined:

L2 via AdamW weight decay (``config.weight_decay``)
    AdamW decouples the weight-decay step from the gradient update, so the
    effective penalty is ``weight_decay * ||w||_2^2``.  Set
    ``config.weight_decay=0`` to disable.

L1 manual penalty (``l1_lambda`` argument to :func:`train`)
    At each forward pass the sum of absolute parameter values is added to the
    task loss: ``total_loss = task_loss + l1_lambda * sum(|w|)``.  Because
    the L1 ball is a diamond (rhombus) in 2-D, its corners sit on the
    coordinate axes, making it likely that the optimum lands exactly on an
    axis and drives some weights to zero — that is, L1 induces sparsity.
    L2 uses a circle / sphere whose smooth boundary never forces exact zeros,
    so L2 shrinks weights uniformly rather than zeroing them.  Set
    ``l1_lambda=0`` (default) to disable.
"""
import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import Config
from .data import make_loaders
from .evaluate import metrics, predict
from .model import build_model
from .utils import seed_everything

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atomic_save(payload: dict, path: Path) -> None:
    """Serialize *payload* to a temp file then rename it over *path* atomically."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# L1 penalty
# ---------------------------------------------------------------------------

def _l1_penalty(model: nn.Module) -> torch.Tensor:
    """Return the sum of absolute values of all trainable parameters.

    Called inside the training loop when ``l1_lambda > 0``.  The result is
    a scalar tensor on the same device as the model so autograd can
    differentiate through it.

    Only parameters that require gradients are included; frozen layers (e.g.
    a partially-frozen backbone) are excluded automatically.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            penalty = penalty + param.abs().sum()
    return penalty


# ---------------------------------------------------------------------------
# Regression utilities
# ---------------------------------------------------------------------------

def round_to_grades(raw: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Clip and round continuous outputs to integer grades."""
    return np.clip(np.round(raw).astype(int), 0, num_classes - 1)


def apply_thresholds(raw: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert regression scores to grades using *thresholds* cut-points.

    A scalar ``s`` maps to grade ``k`` if ``thresholds[k-1] <= s < thresholds[k]``
    (boundaries at -inf and +inf are implied).  There must be exactly
    ``num_classes - 1`` thresholds in ascending order.
    """
    grades = np.zeros(len(raw), dtype=int)
    for k, thr in enumerate(thresholds, start=1):
        grades[raw >= thr] = k
    return grades


def _qwk_neg(thresholds: np.ndarray, raw: np.ndarray, truth: np.ndarray,
              num_classes: int) -> float:
    """Objective for threshold optimiser (negated QWK, for minimisation)."""
    # Enforce monotonicity via cumsum of softplus-transformed differences.
    # We optimise over unconstrained real parameters and reconstruct thresholds
    # inside the objective so the solver never encounters an invalid ordering.
    thr = np.cumsum(np.log1p(np.exp(thresholds)))
    predictions = apply_thresholds(raw, thr)
    try:
        kappa = cohen_kappa_score(truth, predictions,
                                  labels=list(range(num_classes)),
                                  weights="quadratic")
    except ValueError:
        kappa = -1.0
    return -float(kappa)


def optimise_thresholds(
    raw: np.ndarray,
    truth: np.ndarray,
    num_classes: int = 5,
    n_restarts: int = 5,
) -> Tuple[np.ndarray, float]:
    """Search for QWK-maximising thresholds on the validation set.

    Parameters
    ----------
    raw:
        Continuous regression outputs (before rounding).
    truth:
        Integer ground-truth grades.
    num_classes:
        Number of grade classes (determines number of thresholds = num_classes-1).
    n_restarts:
        Number of random restarts for the Nelder-Mead solver.

    Returns
    -------
    thresholds : np.ndarray, shape (num_classes - 1,)
        Monotonically increasing cut-points in the same scale as *raw*.
    best_qwk : float
    """
    rng = np.random.default_rng(0)
    best_qwk = -np.inf
    best_thr: Optional[np.ndarray] = None
    n_thr = num_classes - 1

    for _ in range(n_restarts):
        x0 = rng.uniform(-1.0, 1.0, size=n_thr)
        result = minimize(
            _qwk_neg,
            x0,
            args=(raw, truth, num_classes),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
        )
        qwk = -result.fun
        if qwk > best_qwk:
            best_qwk = qwk
            best_thr = np.cumsum(np.log1p(np.exp(result.x)))

    return best_thr, best_qwk


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_raw(
    model: nn.Module, loader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw model outputs and ground-truth labels as NumPy arrays.

    For classification the raw outputs are logits (shape [N, C]).
    For regression the raw outputs are scalars (shape [N]).
    """
    model.eval()
    raws, truths = [], []
    for images, labels in loader:
        out = model(images.to(device))
        raws.append(out.squeeze(-1).cpu().float().numpy())
        truths.append(labels.numpy())
    return np.concatenate(raws), np.concatenate(truths)


def _val_loss_and_preds(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    mode: str,
    num_classes: int,
    thresholds: Optional[np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute validation loss and integer grade predictions."""
    raw_outputs, truth = _predict_raw(model, loader, device)
    truth_int = truth.astype(int)

    # --- loss ---
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            if mode == "regression":
                val_loss += criterion(out.squeeze(-1),
                                      labels.float()).item() * len(labels)
            else:
                val_loss += criterion(out, labels).item() * len(labels)
    val_loss /= len(loader.dataset)

    # --- predictions ---
    if mode == "regression":
        if thresholds is not None:
            preds = apply_thresholds(raw_outputs, thresholds)
        else:
            preds = round_to_grades(raw_outputs, num_classes)
    else:
        preds = raw_outputs.argmax(axis=1).astype(int)

    return val_loss, preds, truth_int


# ---------------------------------------------------------------------------
# Weighted sampler factory
# ---------------------------------------------------------------------------

def _make_weighted_sampler(
    train_labels: np.ndarray,
) -> WeightedRandomSampler:
    """Return a sampler that up-weights rare classes in each mini-batch."""
    counts = np.bincount(train_labels, minlength=5)
    class_weight = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weight[train_labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def _write_csv(path: Path, history: list) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: Config, resume: Optional[str] = None,
          l1_lambda: float = 0.0) -> List[dict]:
    """Run the full training loop and return the epoch history list.

    Parameters
    ----------
    config:
        A fully validated ``Config`` instance.
    resume:
        Path to a ``last.pt`` checkpoint to continue training from.  When
        ``None`` a fresh run is started.  An existing ``last.pt`` in
        ``config.output_dir`` without a *resume* argument raises an error to
        prevent silent overwrites.
    l1_lambda:
        Coefficient for the manual L1 weight penalty added to the task loss
        each mini-batch: ``total_loss = task_loss + l1_lambda * Σ|w|``.
        Set to ``0.0`` (default) to train without L1 regularisation.  Can
        be combined with AdamW ``weight_decay`` (L2) simultaneously.

    Returns
    -------
    list[dict]
        One entry per epoch with keys ``epoch``, ``train_loss``, ``val_loss``,
        ``qwk``, plus metric details from :func:`evaluate.metrics`.
        Each row also carries ``l1_lambda`` so the history is self-describing.
    """
    if l1_lambda < 0:
        raise ValueError("l1_lambda must be >= 0")
    seed_everything(config.seed)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # --- Guard against accidental overwrites ---
    if not resume and (root / "last.pt").exists():
        raise FileExistsError(
            "Run already exists: resume last.pt or choose a new output_dir"
        )

    # --- Load checkpoint (if resuming) ---
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    )
    if checkpoint:
        chk_cfg = checkpoint["config"]
        if chk_cfg.get("ben_graham", False) != config.ben_graham:
            raise ValueError(
                "Resume enhancement differs. Match the original pipeline or start a new run."
            )
        if chk_cfg.get("mode", "classification") != config.mode:
            raise ValueError(
                "Resume mode differs from the requested configuration."
            )
        for key, value in chk_cfg.items():
            if key not in (
                "epochs", "output_dir", "num_workers",
                "patience", "threshold_search",
            ) and getattr(config, key, value) != value:
                raise ValueError(f"Resume configuration differs: {key}")

    # --- Data loaders ---
    loaders, train_labels, manifest = make_loaders(
        config, checkpoint["manifest"] if checkpoint else None
    )
    (root / "splits.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (root / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    # Optionally replace shuffle with a weighted sampler (classification only).
    if config.use_weighted_sampler and config.mode == "classification":
        sampler = _make_weighted_sampler(train_labels)
        # Rebuild train loader with the sampler (shuffle must be False).
        train_dataset = loaders["train"].dataset
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    # --- Device, model, optimiser, scheduler ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        config.model_name,
        num_classes=config.num_classes,
        mode=config.mode,
        pretrained=(checkpoint is None),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Cosine annealing over the full training budget; warm restarts are not
    # used so the schedule is predictable and reproducible across resumes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 1e-3
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # --- Loss function ---
    if config.mode == "regression":
        criterion: nn.Module = nn.SmoothL1Loss()
    else:
        if config.class_weights and not config.use_weighted_sampler:
            counts = np.bincount(train_labels, minlength=config.num_classes)
            w = torch.tensor(
                len(train_labels) / (config.num_classes * counts),
                dtype=torch.float32,
                device=device,
            )
        else:
            w = None
        criterion = nn.CrossEntropyLoss(weight=w)

    # --- Restore state from checkpoint ---
    start, best_qwk, no_improve, history = 0, -float("inf"), 0, []
    best_model: Optional[dict] = None
    best_thresholds: Optional[np.ndarray] = None

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        best_qwk = checkpoint["best_qwk"]
        no_improve = checkpoint.get("no_improve", 0)
        history = checkpoint["history"]
        # Restore thresholds if they were saved.
        saved_thr = checkpoint.get("best_thresholds")
        best_thresholds = np.array(saved_thr) if saved_thr is not None else None
        if checkpoint.get("best_model") is not None:
            best_model = checkpoint["best_model"]
            best_payload = dict(checkpoint, model=checkpoint["best_model"])
            atomic_save(best_payload, root / "best.pt")

    # Advance scheduler to the correct step so the LR curve is continuous.
    if start > 0:
        for _ in range(start):
            scheduler.step()

    # --- Training loop ---
    for epoch in range(start, config.epochs):
        # Per-epoch seeds make epoch-boundary restarts deterministic.
        seed_everything(config.seed + epoch)

        # ---- Train phase ----
        model.train()
        loss_sum, seen = 0.0, 0
        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{config.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                out = model(images)
                if config.mode == "regression":
                    loss = criterion(out.squeeze(-1), labels.float())
                else:
                    loss = criterion(out, labels)
                # L1 penalty is added inside autocast so it participates in
                # mixed-precision scaling without a device-type mismatch.
                if l1_lambda > 0.0:
                    loss = loss + l1_lambda * _l1_penalty(model)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite training loss; previous checkpoint is preserved"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)

        train_loss = loss_sum / seen

        # ---- Validation phase ----
        # Current thresholds: use the best found so far (may be None on epoch 0).
        val_loss, preds, truth = _val_loss_and_preds(
            model, loaders["val"], criterion, device,
            config.mode, config.num_classes, best_thresholds,
        )

        # Threshold search on the validation set (regression only).
        epoch_thresholds: Optional[np.ndarray] = best_thresholds
        if config.mode == "regression" and config.threshold_search:
            raw_outputs, truth_arr = _predict_raw(model, loaders["val"], device)
            opt_thr, opt_qwk = optimise_thresholds(
                raw_outputs, truth_arr, num_classes=config.num_classes
            )
            # Only adopt the new thresholds if they improve QWK.
            round_preds = round_to_grades(raw_outputs, config.num_classes)
            round_qwk = cohen_kappa_score(
                truth_arr, round_preds,
                labels=list(range(config.num_classes)),
                weights="quadratic",
            )
            if opt_qwk >= round_qwk:
                preds = apply_thresholds(raw_outputs, opt_thr)
                epoch_thresholds = opt_thr
            else:
                preds = round_preds
                epoch_thresholds = None  # plain rounding wins this epoch

        result = metrics(truth, preds)

        # ---- Early stopping ----
        improved = result["qwk"] is not None and result["qwk"] > best_qwk
        if improved:
            best_qwk = result["qwk"]
            best_thresholds = epoch_thresholds
            no_improve = 0
            best_model = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1

        scheduler.step()

        # ---- Build history row ----
        row: dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "l1_lambda": l1_lambda,
            **result,
        }
        if config.mode == "regression":
            row["thresholds"] = (
                best_thresholds.tolist() if best_thresholds is not None else None
            )
        history.append(row)

        # ---- Persist checkpoints ----
        payload = {
            "config": asdict(config),
            "manifest": manifest,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_qwk": best_qwk,
            "best_model": best_model,
            "best_thresholds": (
                best_thresholds.tolist() if best_thresholds is not None else None
            ),
            "no_improve": no_improve,
            "history": history,
        }
        # output_dir points directly to mounted Drive in the notebook.
        atomic_save(payload, root / "last.pt")
        if improved:
            atomic_save(payload, root / "best.pt")

        # ---- Write human-readable logs ----
        (root / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        _write_csv(root / "history.csv", history)
        print(history[-1])

        # ---- Check early-stopping condition ----
        if config.patience > 0 and no_improve >= config.patience:
            print(
                f"Early stopping: QWK did not improve for {no_improve} consecutive epochs."
            )
            break

    return history


# ---------------------------------------------------------------------------
# Regularisation comparison helper
# ---------------------------------------------------------------------------

def train_one_config(
    base_config: Config,
    output_subdir: str,
    weight_decay: float,
    l1_lambda: float,
    label: str,
) -> List[dict]:
    """Train one arm of a regularisation experiment and return its history.

    Builds a fresh ``Config`` from *base_config* with ``weight_decay`` and
    ``output_dir`` overridden, then calls :func:`train` with the given
    *l1_lambda*.  Using a distinct *output_subdir* per arm keeps checkpoints,
    CSV logs, and JSON histories isolated and re-readable after the kernel
    restarts.

    Parameters
    ----------
    base_config:
        Template ``Config`` with all shared hyperparameters already set.
    output_subdir:
        Directory name appended to ``base_config.output_dir``.  The final
        output path will be ``<base_config.output_dir>/<output_subdir>``.
    weight_decay:
        AdamW L2 weight-decay coefficient for this arm.
    l1_lambda:
        L1 penalty coefficient for this arm (0.0 = disabled).
    label:
        Human-readable arm label stored in each history row so results can
        be assembled into a comparison table without looking at paths.

    Returns
    -------
    list[dict]
        Epoch history with an extra ``"arm"`` key set to *label*.
    """
    from dataclasses import replace as _dc_replace  # stdlib, always available

    arm_config = _dc_replace(
        base_config,
        weight_decay=weight_decay,
        output_dir=str(Path(base_config.output_dir) / output_subdir),
    )
    history = train(arm_config, l1_lambda=l1_lambda)
    for row in history:
        row["arm"] = label
    return history


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetinoCheck backbone")
    parser.add_argument(
        "--model", choices=["resnet18", "efficientnet_b0"], default="resnet18"
    )
    parser.add_argument(
        "--mode", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience in epochs (0 = disabled)."
    )
    parser.add_argument(
        "--use-weighted-sampler", action="store_true",
        help="Use WeightedRandomSampler instead of class-weighted loss."
    )
    parser.add_argument(
        "--no-threshold-search", action="store_true",
        help="Disable threshold optimiser (regression mode only)."
    )
    parser.add_argument(
        "--l1-lambda", type=float, default=0.0,
        help="L1 penalty coefficient (0 = disabled)."
    )
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT")
    args = parser.parse_args()

    train(
        Config(
            model_name=args.model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            mode=args.mode,
            patience=args.patience,
            use_weighted_sampler=args.use_weighted_sampler,
            threshold_search=not args.no_threshold_search,
        ),
        args.resume,
        l1_lambda=args.l1_lambda,
    )
