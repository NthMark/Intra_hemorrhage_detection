"""Sequence model (Stage 2) for ICH detection.

Takes per-slice logits from the 2D CNN (Stage 1) ordered within each CT study
and refines them using a bidirectional GRU + 1D-CNN with skip connections.

Architecture matches the SequenceModel from the 1st-place RSNA 2019 solution:
  - Sequence Model 1: FC → BiGRU  (operating on CNN logits per study)
  - Sequence Model 2: Conv1D → BiGRU (operating on per-class logit sequences)
  Both models output class logits per slice; final output is their sum.

Reference:
    Wang et al., "A Deep Learning Algorithm for Automatic Detection and
    Classification of Acute Intracranial Hemorrhages in Head CT Scans",
    NeuroImage: Clinical, 2021.  https://doi.org/10.1016/j.nicl.2021.102785
    GitHub: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection

Input (forward):
    logits : Tensor [B, seq_len, num_classes]  — 2D CNN logit sequences
             B = batch (studies packed together)

Output:
    Tensor [B, seq_len, num_classes] — refined logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceModel(nn.Module):
    """Bidirectional GRU sequence model for refining per-slice ICH predictions.

    Args:
        num_classes: Number of ICH classes (default 6).
        hidden: GRU hidden size (paper uses 96).
        lstm_layers: Number of BiGRU layers (paper uses 2).
        dropout: Dropout probability (paper uses 0.5).
    """

    def __init__(
        self,
        num_classes: int = 6,
        hidden: int = 96,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        use_slice_thickness: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.use_slice_thickness = use_slice_thickness

        # ── Sequence Model 1: FC compression + BiGRU ─────────────────────────
        # Input: [B, seq_len, num_classes] logits
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_classes, 64),
            nn.BatchNorm1d(64),   # applied per-token inside forward
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.bigru1 = nn.GRU(
            input_size=32,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Project BiGRU output → per-class logit per slice
        self.head1 = nn.Linear(hidden * 2, num_classes)
        # Skip connection: FC output → logits (bypasses BiGRU, matches s1.png ③)
        self.skip1 = nn.Linear(32, num_classes)

        # ── Sequence Model 2: Conv1D + BiGRU on per-class sequences ──────────
        # Operates on [B, num_classes, seq_len] (transposed from input)
        ratio = 4
        conv_channels = 64 * ratio       # 256

        # Conv1D input: logits[C] + model1_sigmoid[C] + optional slice_thickness[1]
        conv_in_ch = num_classes * 2 + (1 if use_slice_thickness else 0)
        self.conv_first = nn.Sequential(
            nn.Conv1d(conv_in_ch, conv_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(conv_channels // 2),
            nn.ReLU(),
        )
        mid_ch = conv_channels // 2      # 128

        self.conv_res = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
        )
        self.conv_final = nn.Conv1d(mid_ch, num_classes, kernel_size=3, padding=1, bias=False)

        self.bigru2 = nn.GRU(
            input_size=mid_ch,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head2 = nn.Linear(hidden * 2, num_classes)

    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        slice_thickness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, seq_len, num_classes]  raw 2D CNN logits
            slice_thickness: [B, seq_len]  SliceThickness in mm from DICOM header
                (paper: fed as extra metadata to Sequence Model 2).  Pass None
                to skip (also skipped when use_slice_thickness=False).

        Returns:
            [B, seq_len, num_classes] refined logits
        """
        B, T, C = logits.shape

        # ── Model 1 ──────────────────────────────────────────────────────────
        # FC: [B, T, C] → [B, T, 32]
        x1 = logits.view(B * T, C)
        x1 = self.fc1[0](x1)     # Dropout
        x1 = self.fc1[1](x1)     # Linear → 64
        # BatchNorm1d expects (N, C) or (N, C, L); use functional BN
        x1 = F.batch_norm(
            x1,
            self.fc1[2].running_mean, self.fc1[2].running_var,
            self.fc1[2].weight, self.fc1[2].bias,
            self.fc1[2].training,
        )
        x1 = self.fc1[3](x1)     # ReLU
        x1 = self.fc1[4](x1)     # Dropout
        x1 = self.fc1[5](x1)     # Linear → 32
        x1 = self.fc1[6](x1)     # ReLU
        x1 = x1.view(B, T, 32)

        # BiGRU: [B, T, 32] → [B, T, 2*hidden]
        gru1_out, _ = self.bigru1(x1)
        out1 = self.head1(gru1_out)         # [B, T, C]
        out1 = out1 + self.skip1(x1)        # skip connection ③ from s1.png

        # ── Model 2 ──────────────────────────────────────────────────────────
        # Concatenate raw logits + Model-1 sigmoid predictions
        sig1 = torch.sigmoid(out1)          # [B, T, C]
        x2 = torch.cat([logits, sig1], dim=-1)  # [B, T, 2C]
        # Optionally append normalised SliceThickness as extra channel
        if self.use_slice_thickness and slice_thickness is not None:
            # Normalise: divide by 10mm (typical max for brain CT) → roughly [0, 1]
            st = (slice_thickness.float() / 10.0).unsqueeze(-1)  # [B, T, 1]
            x2 = torch.cat([x2, st], dim=-1)                     # [B, T, 2C+1]
        x2 = x2.permute(0, 2, 1)           # [B, 2C(+1), T]

        x2 = self.conv_first(x2)            # [B, mid_ch, T]
        x2 = x2 + self.conv_res(x2)         # residual
        conv_out = self.conv_final(x2)       # [B, C, T]

        # BiGRU on conv features: [B, T, mid_ch]
        x_gru = x2.permute(0, 2, 1)
        gru2_out, _ = self.bigru2(x_gru)    # [B, T, 2*hidden]
        out2 = self.head2(gru2_out)          # [B, T, C]

        # Skip-connection sum (matching paper's element-wise sum)
        out2 = out2 + conv_out.permute(0, 2, 1)

        return out1 + out2                   # [B, T, C]
