"""Training loop for ICH detection model with MLflow experiment tracking."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Disable MLflow telemetry before the first import so start_run() never blocks
os.environ.setdefault("MLFLOW_DO_NOT_TRACK", "true")
import mlflow  # noqa: E402

from src.models.evaluate import evaluate_epoch

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Numerically stable binary focal loss for multi-label classification.

    Uses the log-sum-exp formulation from the 1st-place RSNA ICH solution
    (https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)
    to avoid sigmoid overflow on large logit magnitudes.

    Args:
        gamma: Focusing parameter (default 2).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Numerically stable BCE via log-sum-exp (avoids exp overflow)
        max_val = (-logits).clamp(min=0)
        loss = logits - logits * targets + max_val + (
            (-max_val).exp() + (-logits - max_val).exp()
        ).log()
        # Focal weighting: down-weight easy examples
        invprobs = nn.functional.logsigmoid(-logits * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Instantiate optimizer from config dict.

    Paper uses: Adam(lr=0.0005, betas=(0.9,0.999), eps=1e-8, weight_decay=2e-5)
    """
    lr = config.get("lr", 5e-4)
    weight_decay = config.get("weight_decay", 2e-5)
    eps = config.get("eps", 1e-8)
    name = config.get("name", "adam").lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.999),
            eps=eps, weight_decay=weight_decay,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> Optional[object]:
    """Instantiate learning rate scheduler from config dict.

    Paper uses: WarmRestart(T_max=5, T_mult=1, eta_min=1e-5)
    """
    name = config.get("name", "warm_restart").lower()

    if name == "warm_restart":
        from src.models.schedulers import WarmRestart
        return WarmRestart(
            optimizer,
            T_max=config.get("T_max", 5),
            T_mult=config.get("T_mult", 1),
            eta_min=config.get("eta_min", 1e-5),
        )
    elif name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("eta_min", 1e-7),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Run a single training epoch.

    Returns:
        Dict with 'loss' key.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    log_interval = max(1, min(100, num_batches // 10))  # every 100 batches or 10% whichever is smaller
    logger.info("  Epoch started — %d batches", num_batches)

    for batch_idx, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        running_loss += loss.item()

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            pct = 100 * batch_idx / num_batches
            logger.info(
                "  [%5.1f%%] Batch %d/%d | avg_loss=%.4f",
                pct, batch_idx, num_batches, running_loss / batch_idx,
            )

    return {"loss": running_loss / num_batches}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_dir: Path,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Full training loop with early stopping and MLflow logging.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Flat config dict (merged from params.yaml).
        checkpoint_dir: Directory to save model checkpoints.
        device: Compute device. Defaults to CUDA if available.

    Returns:
        Best model (loaded from checkpoint).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Moving model to device %s...", device)
    model = model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building criterion / optimizer / scheduler...")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_name = config.get("loss_name", "bce")
    if loss_name == "focal":
        criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0))
    else:
        # Paper uses BCEWithLogitsLoss with uniform pos_weight
        pos_weight = torch.ones(config.get("num_classes", 6)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer (paper: Adam lr=5e-4, wd=2e-5) ─────────────────────────────
    optimizer = build_optimizer(model, config.get("optimizer", {}))

    # ── Scheduler (paper: WarmRestart T_max=5, then warm_restart T_mult=2) ───
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))

    scaler = GradScaler("cuda") if config.get("mixed_precision", True) else None
    logger.info("Setup complete. Starting epoch loop (epochs=%d)...", config.get("epochs", 50))

    epochs = config.get("epochs", 50)
    patience = config.get("early_stopping_patience", 10)
    best_metric = -np.inf
    epochs_without_improvement = 0
    best_checkpoint = checkpoint_dir / "best_model.pt"
    history = []

    with mlflow.start_run():
        mlflow.log_params(
            {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Paper scheduler rule ──────────────────────────────────────────
            # Epochs 1-10: constant lr (warmup plateau, no scheduler step)
            # Epoch 11+  : step() + warm_restart(T_mult=2) every epoch
            if epoch > 10 and scheduler is not None:
                scheduler.step()
                if hasattr(scheduler, "T_max"):           # WarmRestart instance
                    from src.models.schedulers import warm_restart
                    scheduler = warm_restart(scheduler, T_mult=2)

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                gradient_clip=config.get("gradient_clip", 1.0),
            )

            # ── Validate every N epochs (paper: every 5) ──────────────────────
            val_interval = config.get("val_interval", 5)
            do_val = (epoch % val_interval == 0) or (epoch == 1) or (epoch == epochs)
            if do_val:
                val_metrics = evaluate_epoch(model, val_loader, criterion, device)

            if scheduler is not None and epoch <= 10:
                pass  # warmup plateau — scheduler steps begin at epoch 11

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            if do_val:
                monitor = val_metrics["auc"]
                if np.isnan(monitor):
                    monitor = 0.0
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_auc=%.4f | lr=%.2e | %.0fs",
                    epoch, epochs,
                    train_metrics["loss"],
                    val_metrics["auc"],
                    current_lr,
                    elapsed,
                )
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "val_auc": val_metrics["auc"],
                        "lr": current_lr,
                    },
                    step=epoch,
                )
                history.append({
                    "epoch": epoch,
                    "lr": current_lr,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })
            else:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | (val skipped) | lr=%.2e | %.0fs",
                    epoch, epochs,
                    train_metrics["loss"],
                    current_lr,
                    elapsed,
                )
                mlflow.log_metrics(
                    {"train_loss": train_metrics["loss"], "lr": current_lr},
                    step=epoch,
                )
                history.append({
                    "epoch": epoch,
                    "lr": current_lr,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                })

            if do_val and monitor > best_metric:
                best_metric = monitor
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "config": config,
                    },
                    best_checkpoint,
                )
                logger.info("  ✓ New best checkpoint saved (metric=%.4f)", best_metric)
            elif do_val:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        patience,
                    )
                    break

        mlflow.log_metric("best_val_metric", best_metric)
        if best_checkpoint.exists():
            mlflow.log_artifact(str(best_checkpoint))

    # Save training history to JSON
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)

    # Reload best weights
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model
