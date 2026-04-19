#!/usr/bin/env python
"""
evaluate.py – Evaluate a trained checkpoint on the held-out test set.

Outputs a JSON metrics file to reports/metrics/.

Usage:
    python scripts/evaluate.py \
        --checkpoint models/checkpoints/best_model.pt \
        --params params.yaml
"""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate ICH detection model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--metrics-out", default="reports/metrics/test_metrics.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ── Data ──────────────────────────────────────────────────────────────────
    import pandas as pd

    processed_dir = Path(params["data"]["processed_dir"])
    test_df = pd.read_csv(processed_dir / "test.csv")

    from src.data.augmentation import build_val_transforms
    from src.data.dataset import build_dataloaders

    image_size = params["preprocessing"]["image_size"][0]
    _, _, test_loader = build_dataloaders(
        train_df=test_df,
        val_df=test_df,
        test_df=test_df,
        train_transform=build_val_transforms(image_size=image_size),
        val_transform=build_val_transforms(image_size=image_size),
        batch_size=params["inference"]["batch_size"],
        num_workers=params["training"]["num_workers"],
        pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = params["model"]["name"]
    num_classes = params["data"]["num_classes"]

    if "efficientnet" in model_name:
        from src.models.architectures.efficientnet import build_efficientnet
        model = build_efficientnet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=params["model"]["dropout"],
        )
    else:
        from src.models.architectures.resnet import build_resnet
        model = build_resnet(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=params["model"]["dropout"],
        )

    from src.models.predict import load_model
    model = load_model(Path(args.checkpoint), model, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from src.models.evaluate import evaluate_full
    metrics = evaluate_full(model, test_loader, device)

    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info("  %-35s %.4f", k, v)

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", out_path)

    # Log to MLflow if a run is active
    try:
        import mlflow
        mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
