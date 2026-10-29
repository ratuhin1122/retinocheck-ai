"""Comparison runner: ResNet-18 vs EfficientNet-B0 in Classification and Regression modes.

This module automates training/evaluation of both architectures under:
1. 5-class classification (CrossEntropy with optional class weighting)
2. Ordinal regression (SmoothL1 / MSE loss with 4 validation-optimized thresholds)

Outputs are logged to `reports/results.csv` with columns:
`model, mode, regularization, accuracy, macro_f1, qwk`
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config, CHECKPOINT_ROOT
from .evaluate import evaluate, save_results_csv
from .train import train


EXPERIMENT_CONFIGS = [
    {
        "model": "resnet18",
        "mode": "classification",
        "regularization": "L2 (wd=0.01)",
        "weight_decay": 1e-2,
        "l1_lambda": 0.0,
        "regression_loss": "smooth_l1",
        "output_subdir": "resnet18_classification",
    },
    {
        "model": "resnet18",
        "mode": "regression",
        "regularization": "L2 (wd=0.01)",
        "weight_decay": 1e-2,
        "l1_lambda": 0.0,
        "regression_loss": "smooth_l1",
        "output_subdir": "resnet18_regression",
    },
    {
        "model": "efficientnet_b0",
        "mode": "classification",
        "regularization": "L2 (wd=0.01)",
        "weight_decay": 1e-2,
        "l1_lambda": 0.0,
        "regression_loss": "smooth_l1",
        "output_subdir": "efficientnet_b0_classification",
    },
    {
        "model": "efficientnet_b0",
        "mode": "regression",
        "regularization": "L2 (wd=0.01)",
        "weight_decay": 1e-2,
        "l1_lambda": 0.0,
        "regression_loss": "smooth_l1",
        "output_subdir": "efficientnet_b0_regression",
    },
]


def run_experiment(
    exp_def: dict,
    base_config: Optional[Config] = None,
    split: str = "test",
    skip_training_if_done: bool = True,
) -> dict:
    """Train or load a checkpoint, evaluate on `split`, and return a result row.

    Parameters
    ----------
    exp_def:
        Dictionary defining the experiment parameters.
    base_config:
        Optional base config template.
    split:
        Evaluation split ('test' or 'val').
    skip_training_if_done:
        If True and best.pt already exists, skips training and evaluates directly.

    Returns
    -------
    dict:
        Keys: 'model', 'mode', 'regularization', 'accuracy', 'macro_f1', 'qwk'
    """
    if base_config is None:
        base_config = Config()

    out_dir = Path(base_config.output_dir) / exp_def["output_subdir"]
    best_ckpt = out_dir / "best.pt"

    config = replace(
        base_config,
        model_name=exp_def["model"],
        mode=exp_def["mode"],
        weight_decay=exp_def["weight_decay"],
        regression_loss=exp_def.get("regression_loss", "smooth_l1"),
        output_dir=str(out_dir),
    )

    if not (skip_training_if_done and best_ckpt.exists()):
        print(f"\nTraining {exp_def['model']} in {exp_def['mode']} mode -> {out_dir}")
        train(config, l1_lambda=exp_def.get("l1_lambda", 0.0))

    eval_results = evaluate(str(best_ckpt), output_dir=str(out_dir / "reports"), split=split)

    row = {
        "model": exp_def["model"],
        "mode": exp_def["mode"],
        "regularization": exp_def["regularization"],
        "accuracy": eval_results["accuracy"],
        "macro_f1": eval_results["macro_f1"],
        "qwk": eval_results["qwk"],
    }
    return row


def compare_all(
    configs: Optional[List[dict]] = None,
    output_csv: str = "reports/results.csv",
    split: str = "test",
    base_config: Optional[Config] = None,
) -> List[dict]:
    """Execute all comparisons and save the compiled results to CSV.

    Parameters
    ----------
    configs:
        List of experiment definitions (defaults to EXPERIMENT_CONFIGS).
    output_csv:
        Path for reports/results.csv.
    split:
        'val' or 'test'.
    base_config:
        Optional Config instance to inherit settings from.

    Returns
    -------
    list of dicts containing the result rows.
    """
    if configs is None:
        configs = EXPERIMENT_CONFIGS

    results = []
    print("=" * 70)
    print("Starting Model & Mode Comparison (ResNet18 vs EfficientNet-B0)")
    print("=" * 70)

    for exp in configs:
        row = run_experiment(exp, base_config=base_config, split=split)
        results.append(row)
        print(f"[{row['model']} | {row['mode']}] Acc: {row['accuracy']:.4f} | F1: {row['macro_f1']:.4f} | QWK: {row['qwk']:.4f}")

    save_results_csv(results, output_csv)
    print(f"\nAll results saved to {output_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression vs classification comparison")
    parser.add_argument("--output-csv", default="reports/results.csv", help="Destination results CSV path")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluation split")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per arm")
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    compare_all(output_csv=args.output_csv, split=args.split, base_config=cfg)

