"""
Inference utilities: single-image and batch prediction with optional TTA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import clip_hu, hu_to_3channel, load_dicom_slice

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES: List[str] = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def load_model(checkpoint_path: Path, model: nn.Module, device: torch.device) -> nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint saved by train.py.
        model: Uninitialised model instance with matching architecture.
        device: Target device.

    Returns:
        Model with loaded weights in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_single(
    model: nn.Module,
    dcm_path: Path,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run inference on a single DICOM file.

    Args:
        model: Loaded model in eval mode.
        dcm_path: Path to .dcm file.
        transform: Validation transform (albumentations Compose).
        device: Compute device.
        threshold: Binary classification threshold.

    Returns:
        Dict mapping hemorrhage type → probability and binary prediction.
    """
    hu_array, metadata = load_dicom_slice(dcm_path)
    hu_array = clip_hu(hu_array)
    image = hu_to_3channel(hu_array)

    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    result: Dict[str, float] = {}
    for cls_name, prob in zip(HEMORRHAGE_TYPES, probs):
        result[f"{cls_name}_prob"] = float(prob)
        result[f"{cls_name}_pred"] = int(prob >= threshold)

    result["metadata"] = metadata  # type: ignore[assignment]
    return result


@torch.inference_mode()
def predict_batch(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Run inference over a DataLoader.

    Returns:
        Dict with 'probs' (N, C) and 'preds' (N, C) arrays.
    """
    model.eval()
    all_probs: List[np.ndarray] = []

    for images, *_ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    return {"probs": probs, "preds": (probs >= threshold).astype(int)}
