"""SE-ResNeXt-101-32x4d backbone for ICH detection.

Implements the third backbone in the paper's 3-model ensemble.  The paper
(Wang et al. 2021, Table 2) trains DenseNet121, DenseNet169, and
SE-ResNeXt101 separately, then averages their per-slice logits before
feeding them into the Stage-2 BiGRU sequence model.

Reference:
    Wang et al., "A Deep Learning Algorithm for Automatic Detection and
    Classification of Acute Intracranial Hemorrhages in Head CT Scans",
    NeuroImage: Clinical, 2021.  https://doi.org/10.1016/j.nicl.2021.102785

    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    timm model key: ``se_resnext101_32x4d``
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.architectures.efficientnet import AdaptiveConcatPool2d


class SEResNeXt101ICH(nn.Module):
    """SE-ResNeXt-101-32×4d with AdaptiveConcatPool2d head for ICH detection.

    Architecture mirrors the ResNetICH wrapper but is fixed to the
    ``se_resnext101_32x4d`` backbone used in the paper's ensemble.

    The classification head:
        AdaptiveConcatPool2d(1) → Flatten → BN → Dropout(0.5) →
        Linear(feature_dim*2, 512) → ReLU → BN → Dropout(0.5) →
        Linear(512, num_classes)

    Args:
        num_classes: Number of output logits (default 6).
        pretrained: Load ImageNet-1K weights via timm (default True).
        dropout: Dropout rate before each linear layer (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for SE-ResNeXt101: pip install timm"
            ) from exc

        self.backbone = timm.create_model(
            "seresnext101_32x4d",
            pretrained=pretrained,
            num_classes=0,         # remove timm's head
            global_pool="",        # disable built-in pooling; we use our own
        )
        feature_dim = self.backbone.num_features   # 2048

        self.pool = AdaptiveConcatPool2d()         # avg + max → 2×feature_dim
        pool_dim = feature_dim * 2                 # 4096

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)     # [B, 2048, H', W']
        x = self.pool(x)         # [B, 4096, 1, 1]
        return self.head(x)      # [B, num_classes]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 4096) pooled features (before the classification head)."""
        x = self.backbone(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)
