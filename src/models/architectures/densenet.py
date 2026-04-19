"""DenseNet-based models for ICH detection.

Implements DenseNet121_change_avg and DenseNet169_change_avg from the
1st-place RSNA 2019 Intracranial Hemorrhage Detection solution.

Reference:
    Wang et al., "A Deep Learning Algorithm for Automatic Detection and
    Classification of Acute Intracranial Hemorrhages in Head CT Scans",
    NeuroImage: Clinical, 2021.  https://doi.org/10.1016/j.nicl.2021.102785
    GitHub: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseNet121ICH(nn.Module):
    """DenseNet-121 with adaptive avg-pool head for ICH multi-label detection.

    Architecture matches ``DenseNet121_change_avg`` from the 1st-place solution:
        features → ReLU → AdaptiveAvgPool2d(1) → flatten → Linear(1024, 6)

    Args:
        num_classes: Number of output logits (default 6).
        pretrained: Load ImageNet-1K weights (default True).
    """

    feature_dim: int = 1024

    def __init__(self, num_classes: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import torchvision.models as tv
        except ImportError as exc:
            raise ImportError("torchvision is required: pip install torchvision") from exc

        weights = tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = tv.densenet121(weights=weights)
        self.features = densenet.features       # [B, 1024, H', W']
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim) penultimate features (before classifier)."""
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class DenseNet169ICH(nn.Module):
    """DenseNet-169 with adaptive avg-pool head for ICH multi-label detection.

    Architecture matches ``DenseNet169_change_avg`` from the 1st-place solution:
        features → ReLU → AdaptiveAvgPool2d(1) → flatten → Linear(1664, 6)

    Args:
        num_classes: Number of output logits (default 6).
        pretrained: Load ImageNet-1K weights (default True).
    """

    feature_dim: int = 1664

    def __init__(self, num_classes: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import torchvision.models as tv
        except ImportError as exc:
            raise ImportError("torchvision is required: pip install torchvision") from exc

        weights = tv.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = tv.densenet169(weights=weights)
        self.features = densenet.features       # [B, 1664, H', W']
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim) penultimate features (before classifier)."""
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


def build_densenet(
    model_name: str = "densenet121",
    num_classes: int = 6,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory for DenseNet ICH models.

    Args:
        model_name: ``'densenet121'`` or ``'densenet169'``.
        num_classes: Output logit count.
        pretrained: Load ImageNet weights.

    Returns:
        DenseNet121ICH or DenseNet169ICH instance.
    """
    name = model_name.lower()
    if "121" in name:
        return DenseNet121ICH(num_classes=num_classes, pretrained=pretrained)
    if "169" in name:
        return DenseNet169ICH(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unsupported DenseNet variant: '{model_name}'. "
        "Choose 'densenet121' or 'densenet169'."
    )
