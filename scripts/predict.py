#!/usr/bin/env python
"""
predict.py – Run inference on a DICOM file or directory.

Usage:
    python scripts/predict.py \
        --input path/to/scan.dcm \
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
    parser = argparse.ArgumentParser(description="Run ICH inference on DICOM file(s)")
    parser.add_argument("--input", required=True, help="Path to .dcm file or directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_name = params["model"]["name"]
    num_classes = params["data"]["num_classes"]

    if "efficientnet" in model_name:
        from src.models.architectures.efficientnet import build_efficientnet
        model = build_efficientnet(
            model_name=model_name, num_classes=num_classes, pretrained=False,
            dropout=params["model"]["dropout"],
        )
    else:
        from src.models.architectures.resnet import build_resnet
        model = build_resnet(
            model_name=model_name, num_classes=num_classes, pretrained=False,
            dropout=params["model"]["dropout"],
        )

    from src.models.predict import load_model, predict_single
    model = load_model(Path(args.checkpoint), model, device)

    from src.data.augmentation import build_val_transforms
    transform = build_val_transforms(image_size=params["preprocessing"]["image_size"][0])

    input_path = Path(args.input)
    dcm_files = (
        [input_path] if input_path.is_file()
        else sorted(input_path.rglob("*.dcm"))
    )

    if not dcm_files:
        logger.error("No DICOM files found at: %s", input_path)
        sys.exit(1)

    results = []
    for dcm_path in dcm_files:
        prediction = predict_single(model, dcm_path, transform, device, args.threshold)
        results.append(prediction)
        logger.info(
            "%s → any_hemorrhage_prob=%.3f",
            dcm_path.name,
            1.0 - prediction.get("no_hemorrhage_prob", 1.0),
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # metadata key is not JSON-serialisable as-is
        serialisable = [{k: v for k, v in r.items() if k != "metadata"} for r in results]
        with open(out, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Predictions saved → %s", out)


if __name__ == "__main__":
    main()
