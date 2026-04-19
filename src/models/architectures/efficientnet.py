"""
EfficientNet-based model for multi-label ICH classification.

Uses timm for backbone, adds a custom classification head.
Adapted with AdaptiveConcatPool2d from the 1st-place RSNA ICH solution
(https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    """Concatenate adaptive average and max pooling outputs.

    Yields a richer global descriptor (2 * feature_dim) than avg-pool alone.
    Borrowed from the 1st-place RSNA ICH Detection solution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.avg(x), self.max(x)], dim=1)


class EfficientNetICH(nn.Module):
    """EfficientNet backbone + classification head for ICH detection.

    Args:
        model_name: timm model name (e.g. 'efficientnet_b4').
        num_classes: Number of output classes (default 6 for ICH subtypes).
        num_input_channels: Number of input channels (3 for RGB-window CT).
        pretrained: Whether to load ImageNet-pretrained weights.
        dropout: Dropout rate before the final linear layer.
        hidden_dim: Hidden dimension of the intermediate FC layer.
        concat_pool: If True, use AdaptiveConcatPool2d (avg+max) instead of
            avg-pool only, doubling the feature dimension going into the head.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = 6,
        num_input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        concat_pool: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required: pip install timm") from exc

        self.concat_pool = concat_pool
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove default head
            in_chans=num_input_channels,
            global_pool="" if concat_pool else "avg",  # disable internal pool
        )
        feature_dim = self.backbone.num_features
        pool_dim = feature_dim * 2 if concat_pool else feature_dim

        self.pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        features = self.backbone(x)          # (B, feature_dim, H', W')
        pooled = self.pool(features)         # (B, pool_dim, 1, 1)
        logits = self.head(pooled)           # (B, num_classes)
        return logits


def build_efficientnet(
    model_name: str = "efficientnet_b4",
    num_classes: int = 6,
    num_input_channels: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    hidden_dim: int = 512,
) -> EfficientNetICH:
    """Factory helper to instantiate EfficientNetICH."""
    return EfficientNetICH(
        model_name=model_name,
        num_classes=num_classes,
        num_input_channels=num_input_channels,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )
