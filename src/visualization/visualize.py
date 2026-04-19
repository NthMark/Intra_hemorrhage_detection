"""
Visualisation utilities for ICH detection:
  - CT window display
  - GradCAM saliency maps
  - ROC curves and confusion matrices
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def plot_ct_windows(
    hu_array: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Display a CT slice in three standard windows side by side.

    Args:
        hu_array: 2-D HU pixel array.
        save_path: If provided, saves figure instead of displaying.
    """
    import matplotlib.pyplot as plt

    from src.data.preprocessing import apply_window, CT_WINDOWS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = list(CT_WINDOWS.keys())

    for ax, (name, (center, width)) in zip(axes, CT_WINDOWS.items()):
        windowed = apply_window(hu_array, center, width)
        ax.imshow(windowed, cmap="gray")
        ax.set_title(f"{name.capitalize()} window\n(C={center}, W={width})")
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved CT window figure → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Plot per-class ROC curves.

    Args:
        targets: Ground-truth binary array, shape (N, C).
        probs: Predicted probability array, shape (N, C).
        save_path: If provided, saves figure to disk.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, (cls_name, ax) in enumerate(zip(HEMORRHAGE_TYPES, axes_flat)):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(cls_name.replace("_", " ").title())
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("ROC Curves per Hemorrhage Type", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ROC curves → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_gradcam(
    model,
    image_tensor,
    target_class: int,
    device,
    original_image: Optional[np.ndarray] = None,
    class_name: Optional[str] = None,
    save_path: Optional[Path] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Compute Grad-CAM and optionally overlay it on the original CT image.

    Uses hooks on the last Conv2d layer of model.backbone to capture
    gradients and feature maps.

    Args:
        model: Trained EfficientNetICH or ResNetICH instance.
        image_tensor: Preprocessed tensor, shape (1, C, H, W).
        target_class: Class index to explain (0-5 for ICH subtypes).
        device: Compute device.
        original_image: Optional (H, W, 3) float image in [0,1] for overlay.
            If None, only the raw heatmap is shown.
        class_name: Label shown in the figure title (e.g. "subarachnoid").
        save_path: If given, saves figure to this path; else shows interactively.
        alpha: Opacity of heatmap overlay (0=invisible, 1=opaque).

    Returns:
        Grad-CAM heatmap as (H, W) float32 array, values in [0, 1].
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    model.eval()
    image_tensor = image_tensor.to(device)

    gradients: List[np.ndarray] = []
    activations: List[np.ndarray] = []

    def _save_grad(grad: torch.Tensor) -> None:
        gradients.append(grad.detach().cpu().numpy())

    def _forward_hook(module, input, output: torch.Tensor) -> None:
        activations.append(output.detach().cpu().numpy())
        output.register_hook(_save_grad)

    # Hook the last Conv2d in the backbone
    target_layer = None
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in model.backbone.")

    handle = target_layer.register_forward_hook(_forward_hook)

    logits = model(image_tensor)
    model.zero_grad()
    logits[0, target_class].backward()
    handle.remove()

    grad = gradients[0][0]   # (C, H, W)
    act = activations[0][0]  # (C, H, W)

    # Global average pool gradients → channel weights
    weights = grad.mean(axis=(1, 2), keepdims=True)
    cam = (weights * act).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cam.astype(np.float32)

    # ── Figure ────────────────────────────────────────────────────────────────
    prob = torch.sigmoid(logits[0, target_class]).item()
    title_label = class_name or HEMORRHAGE_TYPES[target_class]
    title = f"Grad-CAM — {title_label.replace('_', ' ').title()}  (p={prob:.3f})"

    if original_image is not None:
        H, W = original_image.shape[:2]
        cam_up = F.interpolate(
            torch.from_numpy(cam)[None, None],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        heatmap = cm.jet(cam_up)[..., :3]          # (H, W, 3) RGB
        overlay = (1 - alpha) * original_image[..., :3] + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image[..., 0], cmap="gray")
        axes[0].set_title("Input (brain window)")
        axes[0].axis("off")
        axes[1].imshow(cam_up, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        plt.suptitle(title, fontsize=13, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(cam, cmap="jet")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Grad-CAM figure → %s", save_path)
    else:
        plt.show()
    plt.close(fig)

    return cam
