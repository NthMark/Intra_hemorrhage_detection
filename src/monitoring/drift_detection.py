"""
Production-serving monitoring: input data drift and prediction-score drift.

Intended to be called periodically against logged inference batches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES: List[str] = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def compute_prediction_drift(
    reference_probs: np.ndarray,
    current_probs: np.ndarray,
) -> Dict[str, float]:
    """Compute Population Stability Index (PSI) per output class.

    PSI < 0.1  → no significant drift
    PSI 0.1-0.2 → moderate drift (monitor closely)
    PSI > 0.2  → significant drift (retrain recommended)

    Args:
        reference_probs: Baseline probability array, shape (N_ref, C).
        current_probs: Current probability array, shape (N_cur, C).

    Returns:
        Dict mapping class name to PSI score.
    """
    num_classes = reference_probs.shape[1]
    psi_scores: Dict[str, float] = {}

    for i, cls_name in enumerate(HEMORRHAGE_TYPES[:num_classes]):
        psi = _compute_psi(reference_probs[:, i], current_probs[:, i])
        psi_scores[cls_name] = psi
        if psi > 0.2:
            logger.warning(
                "Significant prediction drift detected for '%s': PSI=%.4f. "
                "Consider retraining.",
                cls_name, psi,
            )

    return psi_scores


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute Population Stability Index between two 1-D distributions."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / (ref_counts.sum() + epsilon)
    cur_pct = cur_counts / (cur_counts.sum() + epsilon)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_pixel_statistics(
    images: np.ndarray,
) -> Dict[str, float]:
    """Compute basic pixel-level statistics for input drift detection.

    Args:
        images: Array of shape (N, H, W, C) or (N, C, H, W), float32.

    Returns:
        Dict of mean, std, min, max per channel.
    """
    if images.ndim == 4 and images.shape[1] in (1, 3):
        # (N, C, H, W) → (N, H, W, C)
        images = images.transpose(0, 2, 3, 1)

    stats: Dict[str, float] = {}
    for c in range(images.shape[-1]):
        channel = images[..., c]
        stats[f"ch{c}_mean"] = float(channel.mean())
        stats[f"ch{c}_std"] = float(channel.std())
        stats[f"ch{c}_min"] = float(channel.min())
        stats[f"ch{c}_max"] = float(channel.max())
    return stats


def save_reference_stats(
    reference_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Persist reference prediction statistics for future drift checks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "per_class_mean": reference_probs.mean(axis=0).tolist(),
        "per_class_std": reference_probs.std(axis=0).tolist(),
        "n_samples": len(reference_probs),
    }
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Reference statistics saved to %s", output_path)
