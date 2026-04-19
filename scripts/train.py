#!/usr/bin/env python
"""
train.py – Train ICH detection model.

Usage:
    python scripts/train.py --params params.yaml --model efficientnet_b4
    python scripts/train.py --config configs/training/fast_dev.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ICH detection model")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--config", default=None, help="Training config override YAML")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints/")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    if args.config:
        with open(args.config) as f:
            override = yaml.safe_load(f)
        params["training"].update(override)

    # CLI overrides
    if args.model:
        params["model"]["name"] = args.model
    if args.epochs:
        params["training"]["epochs"] = args.epochs
    if args.batch_size:
        params["training"]["batch_size"] = args.batch_size
    if args.lr:
        params["optimizer"]["lr"] = args.lr

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    import pandas as pd

    processed_dir = Path(params["data"]["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    image_size = tuple(params["preprocessing"]["image_size"])  # (H, W) from config
    image_size_int = image_size[0]  # transforms expect a single int (square)
    dataset_mode = params["training"].get("dataset_mode", "study")
    logger.info("Dataset mode: %s", dataset_mode)

    if dataset_mode == "study":
        # Paper method: study-based sampling with adjacent-slice context
        from src.data.augmentation import build_paper_train_transforms, build_paper_val_transforms
        from src.data.dataset import build_study_dataloaders

        train_loader, val_loader, _ = build_study_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=val_df,
            train_transform=build_paper_train_transforms(image_size=image_size_int),
            val_transform=build_paper_val_transforms(image_size=image_size_int),
            batch_size=params["training"]["batch_size"],
            num_workers=params["training"]["num_workers"],
            pin_memory=params["training"]["pin_memory"],
            image_size=None,
        )
    else:
        # Original slice-based mode
        from src.data.augmentation import build_train_transforms, build_val_transforms
        from src.data.dataset import build_dataloaders

        train_loader, val_loader, _ = build_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=val_df,
            train_transform=build_train_transforms(image_size=image_size_int),
            val_transform=build_val_transforms(image_size=image_size_int),
            batch_size=params["training"]["batch_size"],
            num_workers=params["training"]["num_workers"],
            pin_memory=params["training"]["pin_memory"],
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = params["model"]["name"]
    num_classes = params["data"]["num_classes"]

    if "densenet" in model_name:
        from src.models.architectures.densenet import build_densenet
        model = build_densenet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=params["model"]["pretrained"],
        )
    elif "efficientnet" in model_name:
        from src.models.architectures.efficientnet import build_efficientnet
        model = build_efficientnet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=params["model"]["pretrained"],
            dropout=params["model"]["dropout"],
        )
    else:
        from src.models.architectures.resnet import build_resnet
        model = build_resnet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=params["model"]["pretrained"],
            dropout=params["model"]["dropout"],
        )

    logger.info("Model: %s | Parameters: %s", model_name, f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    flat_config = {
        **params["training"],
        "optimizer": params["optimizer"],
        "scheduler": params["scheduler"],
        "loss_name": params["loss"].get("name", "bce"),
        "focal_gamma": params["loss"]["gamma"],
        "num_classes": num_classes,
        "mixed_precision": params["training"]["mixed_precision"],
        "gradient_clip": params["training"]["gradient_clip"],
    }

    import os, mlflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    logger.info("Starting training loop...")
    from src.models.train import train
    best_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=flat_config,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=device,
    )
    logger.info("Training complete. Best model saved to %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
