"""
ResNet-based model for multi-label ICH classification.

Adapted with AdaptiveConcatPool2d from the 1st-place RSNA ICH solution
(https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.architectures.efficientnet import AdaptiveConcatPool2d


class ResNetICH(nn.Module):
    """ResNet backbone + classification head for ICH detection.

    Args:
        model_name: timm model name (e.g. 'resnet50d').
        num_classes: Number of output classes.
        num_input_channels: Number of input channels.
        pretrained: Whether to load ImageNet-pretrained weights.
        dropout: Dropout rate before final layer.
        hidden_dim: Dimension of intermediate linear layer.
        concat_pool: If True, use AdaptiveConcatPool2d (avg+max) instead of
            avg-pool only, doubling the feature dimension into the head.
    """

    def __init__(
        self,
        model_name: str = "resnet50d",
        num_classes: int = 6,
        num_input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.4,
        hidden_dim: int = 256,
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
            num_classes=0,
            in_chans=num_input_channels,
            global_pool="" if concat_pool else "avg",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, feature_dim, H', W')
        pooled = self.pool(features)  # (B, pool_dim, 1, 1)
        logits = self.head(pooled)    # (B, num_classes)
        return logits


def build_resnet(
    model_name: str = "resnet50d",
    num_classes: int = 6,
    num_input_channels: int = 3,
    pretrained: bool = True,
    dropout: float = 0.4,
    hidden_dim: int = 256,
    concat_pool: bool = True,
) -> ResNetICH:
    """Factory helper to instantiate ResNetICH."""
    return ResNetICH(
        model_name=model_name,
        num_classes=num_classes,
        num_input_channels=num_input_channels,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
        concat_pool=concat_pool,
    )
