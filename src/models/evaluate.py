"""
Model evaluation utilities: per-epoch metrics and full test-set evaluation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES: List[str] = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Compute loss and macro-AUC over a DataLoader.

    Returns:
        Dict with keys 'loss' and 'auc'.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    model.eval()
    running_loss = 0.0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_labels, axis=0)

    try:
        auc = roc_auc_score(targets, probs, average="macro")
    except ValueError:
        # Not all classes may be present in a small validation split
        auc = float("nan")

    return {"loss": running_loss / len(loader), "auc": auc}


@torch.inference_mode()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Full evaluation with per-class and aggregate metrics.

    Per-class AUC is accompanied by a 95% bootstrap confidence interval
    (``auc_{cls}_ci95_lo`` / ``auc_{cls}_ci95_hi``) when ``n_bootstrap > 0``.
    The CI is computed via percentile bootstrap with ``n_bootstrap`` resamples
    (matching the methodology described in Wang et al. 2021).

    Args:
        model: Trained model in eval mode.
        loader: DataLoader for the evaluation split.
        device: Torch device.
        threshold: Decision threshold for binary predictions (default 0.5).
        n_bootstrap: Number of bootstrap resamples for CI. Set to 0 to skip.

    Returns:
        Dict of metric names to float values.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_labels, axis=0).astype(int)
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    try:
        metrics["macro_auc"] = roc_auc_score(targets, probs, average="macro")
    except ValueError:
        metrics["macro_auc"] = float("nan")
    metrics["macro_f1"] = f1_score(targets, preds, average="macro", zero_division=0)
    metrics["subset_accuracy"] = accuracy_score(targets, preds)

    # Per-class AUC
    for i, cls_name in enumerate(HEMORRHAGE_TYPES):
        try:
            metrics[f"auc_{cls_name}"] = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError:
            metrics[f"auc_{cls_name}"] = float("nan")

        tp = int(((preds[:, i] == 1) & (targets[:, i] == 1)).sum())
        fn = int(((preds[:, i] == 0) & (targets[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (targets[:, i] == 0)).sum())
        tn = int(((preds[:, i] == 0) & (targets[:, i] == 0)).sum())

        metrics[f"sensitivity_{cls_name}"] = tp / (tp + fn + 1e-8)
        metrics[f"specificity_{cls_name}"] = tn / (tn + fp + 1e-8)

    # ── Bootstrap 95% CI for per-class AUC ───────────────────────────────────
    if n_bootstrap > 0:
        rng = np.random.default_rng(42)
        n = len(targets)
        for i, cls_name in enumerate(HEMORRHAGE_TYPES):
            y_true = targets[:, i]
            y_score = probs[:, i]
            boot_aucs: List[float] = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                try:
                    boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
                except ValueError:
                    pass  # skip resamples where only one class is present
            if len(boot_aucs) >= 2:
                metrics[f"auc_{cls_name}_ci95_lo"] = float(np.percentile(boot_aucs, 2.5))
                metrics[f"auc_{cls_name}_ci95_hi"] = float(np.percentile(boot_aucs, 97.5))

    return metrics
