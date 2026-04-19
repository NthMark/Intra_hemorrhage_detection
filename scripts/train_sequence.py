#!/usr/bin/env python
"""
train_sequence.py — Train the Stage-2 BiGRU sequence model.

Requires Stage-1 features extracted by scripts/extract_features.py.
The sequence model takes all per-slice logits of a CT study (in slice order)
and refines the predictions using bidirectional GRU context.

Usage:
    python scripts/train_sequence.py --params params.yaml
    python scripts/train_sequence.py --epochs 40 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Study-level dataset for sequence training ─────────────────────────────────

class StudySequenceDataset(Dataset):
    """Dataset of per-study logit sequences for sequence model training.

    Each item is:
        logits : [seq_len, num_classes]  — 2D CNN logits for one CT study
        labels : [seq_len, num_classes]  — ground-truth multi-hot labels

    Studies shorter than ``seq_len`` are zero-padded on the right.
    Studies longer than ``seq_len`` have a random window sampled at training
    time (matching the paper's approach).

    Args:
        features: Dict loaded from features_{split}.pt
        seq_len: Maximum sequence length (paper uses 24).
        mode: 'train' randomly samples a window; 'val' takes from start.
    """

    def __init__(self, features: dict, seq_len: int = 24, mode: str = "train") -> None:
        self.seq_len = seq_len
        self.mode = mode
        self.num_classes = features["logits"].shape[1]
        self.has_slice_thickness = "slice_thickness" in features

        # Group rows by study
        study_logits: Dict[str, List[torch.Tensor]] = defaultdict(list)
        study_labels: Dict[str, List[torch.Tensor]] = defaultdict(list)
        study_st: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for i, sid in enumerate(features["study_id"]):
            study_logits[sid].append(features["logits"][i])
            study_labels[sid].append(features["labels"][i])
            if self.has_slice_thickness:
                study_st[sid].append(features["slice_thickness"][i])

        self.studies: List[str] = sorted(study_logits.keys())
        self.study_logits = {
            sid: torch.stack(study_logits[sid], dim=0) for sid in self.studies
        }
        self.study_labels = {
            sid: torch.stack(study_labels[sid], dim=0) for sid in self.studies
        }
        if self.has_slice_thickness:
            self.study_st = {
                sid: torch.stack(study_st[sid], dim=0) for sid in self.studies
            }
        logger.info(
            "StudySequenceDataset (%s): %d studies, slice_thickness=%s",
            mode, len(self.studies), self.has_slice_thickness,
        )

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int):
        sid = self.studies[idx]
        logits = self.study_logits[sid]  # [N, C]
        labels = self.study_labels[sid]  # [N, C]
        N = logits.shape[0]

        # Sample or clip to seq_len
        start = 0
        if N > self.seq_len and self.mode == "train":
            import random
            start = random.randint(0, N - self.seq_len)

        logits = logits[start : start + self.seq_len]
        labels = labels[start : start + self.seq_len]

        st = None
        if self.has_slice_thickness:
            st = self.study_st[sid][start : start + self.seq_len]  # [T]

        # Zero-pad if shorter than seq_len
        pad = self.seq_len - logits.shape[0]
        if pad > 0:
            logits = F.pad(logits, (0, 0, 0, pad))
            labels = F.pad(labels, (0, 0, 0, pad))
            if st is not None:
                st = F.pad(st, (0, pad))

        return logits, labels, st  # [seq_len, C], [seq_len, C], [seq_len] or None


def collate_fn(batch):
    logits = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    st_list = [b[2] for b in batch]
    st = torch.stack(st_list, dim=0) if st_list[0] is not None else None
    return logits, labels, st


# ── Training utilities ────────────────────────────────────────────────────────

def compute_auc(preds: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(targets, preds, average="macro")
    except Exception:
        return float("nan")


def run_epoch(model, loader, optimizer, device, is_train: bool) -> Dict[str, float]:
    model.train(is_train)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for logits, labels, st in loader:
            logits = logits.to(device)
            labels = labels.to(device)
            st = st.to(device) if st is not None else None

            out = model(logits, slice_thickness=st)  # [B, seq_len, C]
            loss = F.binary_cross_entropy_with_logits(out, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(out).detach().cpu())
            all_labels.append(labels.detach().cpu())

    preds  = torch.cat(all_preds, dim=0).view(-1, all_preds[0].shape[-1]).numpy()
    labels = torch.cat(all_labels, dim=0).view(-1, all_labels[0].shape[-1]).numpy()
    auc = compute_auc(preds, labels)

    return {"loss": total_loss / len(loader), "auc": auc}


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-2 BiGRU sequence model")
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--features-dir", default="data/processed/",
                   help="Directory containing features_{train,val}.pt")
    p.add_argument("--checkpoint-dir", default="models/checkpoints/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=24,
                   help="Sequence length per study window (paper uses 24)")
    p.add_argument("--hidden", type=int, default=96,
                   help="GRU hidden size (paper uses 96)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.params) as f:
        params = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    feat_dir = Path(args.features_dir)
    train_feat_path = feat_dir / "features_train.pt"
    val_feat_path   = feat_dir / "features_val.pt"

    if not train_feat_path.exists():
        logger.error(
            "Training features not found at %s.\n"
            "Run: python scripts/extract_features.py --checkpoint models/checkpoints/best_model.pt",
            train_feat_path,
        )
        raise SystemExit(1)

    train_feat = torch.load(train_feat_path, map_location="cpu")
    val_feat   = torch.load(val_feat_path,   map_location="cpu")

    seq_len = args.seq_len
    train_ds = StudySequenceDataset(train_feat, seq_len=seq_len, mode="train")
    val_ds   = StudySequenceDataset(val_feat,   seq_len=seq_len, mode="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    num_classes = params["data"]["num_classes"]
    use_slice_thickness = train_feat.get("slice_thickness") is not None
    from src.models.sequence_model import SequenceModel
    model = SequenceModel(
        num_classes=num_classes,
        hidden=args.hidden,
        lstm_layers=2,
        dropout=0.5,
        use_slice_thickness=use_slice_thickness,
    ).to(device)
    logger.info("SequenceModel use_slice_thickness=%s", use_slice_thickness)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("SequenceModel parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_sequence_model.pt"

    best_auc = -float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, device, is_train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | %.0fs",
            epoch, args.epochs,
            train_m["loss"], val_m["loss"], val_m["auc"], elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "val_auc": val_m["auc"],
        })

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "best_auc": best_auc},
                best_path,
            )
            logger.info("  ✓ New best sequence model saved (val_auc=%.4f)", best_auc)

    history_path = ckpt_dir / "sequence_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History saved → %s", history_path)
    logger.info("Best val AUC: %.4f", best_auc)


if __name__ == "__main__":
    main()
