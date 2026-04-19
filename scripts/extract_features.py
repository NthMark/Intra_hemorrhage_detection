#!/usr/bin/env python
"""
extract_features.py — Run the trained 2D CNN over the full dataset and save
per-slice logits in study order for the Stage-2 sequence model.

Output (one .pt file per split):
    data/processed/features_train.pt
    data/processed/features_val.pt
    data/processed/features_test.pt

Each .pt file is a dict:
    {
      "study_id"       : List[str],       # study ID per row
      "image_path"     : List[str],       # image path per row
      "logits"         : Tensor[N, 6],    # (ensemble-averaged) CNN logits
      "labels"         : Tensor[N, 6],    # ground-truth labels (float32)
      "slice_thickness": Tensor[N],       # SliceThickness in mm (from CSV)
    }

Single-backbone usage:
    python scripts/extract_features.py \\
        --checkpoint models/checkpoints/best_model.pt

3-backbone ensemble (paper Section 3.2):
    python scripts/extract_features.py \\
        --checkpoints models/checkpoints/densenet121.pt \\
                      models/checkpoints/densenet169.pt \\
                      models/checkpoints/seresnext101.pt \\
        --model-names densenet121 densenet169 se_resnext101
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 2D CNN features for sequence model")
    # Single-checkpoint (kept for backward compatibility)
    p.add_argument("--checkpoint", default=None,
                   help="Path to a single checkpoint (use --checkpoints for ensemble)")
    # Multi-checkpoint ensemble
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="Paths to multiple checkpoints for ensemble averaging")
    p.add_argument("--model-names", nargs="+", default=None,
                   help="Model name for each checkpoint in --checkpoints "
                        "(e.g. densenet121 densenet169 se_resnext101). "
                        "If omitted, params.yaml model.name is used for all.")
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--out-dir", default="data/processed/")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _build_model(model_name: str, num_classes: int, pretrained: bool = False):
    """Instantiate the correct backbone from a model name string."""
    name = model_name.lower()
    if "densenet121" in name:
        from src.models.architectures.densenet import DenseNet121ICH
        return DenseNet121ICH(num_classes=num_classes, pretrained=pretrained)
    if "densenet169" in name:
        from src.models.architectures.densenet import DenseNet169ICH
        return DenseNet169ICH(num_classes=num_classes, pretrained=pretrained)
    if "se_resnext101" in name or "seresnext101" in name:
        from src.models.architectures.seresnext import SEResNeXt101ICH
        return SEResNeXt101ICH(num_classes=num_classes, pretrained=pretrained)
    if "efficientnet" in name:
        from src.models.architectures.efficientnet import build_efficientnet
        return build_efficientnet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    from src.models.architectures.resnet import build_resnet
    return build_resnet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)


def _load_checkpoint(ckpt_path: str, model_name: str, num_classes: int,
                     device: torch.device):
    """Load a single backbone checkpoint and return the model in eval mode."""
    model = _build_model(model_name, num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()
    logger.info("Loaded %s from %s", model_name, ckpt_path)
    return model


def extract_split(
    models: List,
    df,
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Run inference on one split; average logits across all models in the list.

    Args:
        models: List of backbone models (ensemble when len > 1).
        df: DataFrame for this split, must include ``image_path`` and label columns.
            If a ``slice_thickness`` column is present it is included in the output.

    Returns:
        Dict with keys ``study_id``, ``image_path``, ``logits``, ``labels``,
        and optionally ``slice_thickness``.
    """
    from torch.utils.data import DataLoader
    from src.data.dataset import ICHStudyValDataset, _extract_study_id

    ds = ICHStudyValDataset(df, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Run every model in the ensemble and collect logits
    all_logits_per_model = [[] for _ in models]
    all_labels: List[torch.Tensor] = []

    for m in models:
        m.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)

            for m_idx, m in enumerate(models):
                all_logits_per_model[m_idx].append(m(images).cpu())

            all_labels.append(labels.cpu())

            if batch_idx % 50 == 0:
                logger.info("  Batch %d / %d", batch_idx, len(loader))

    # Stack and average across ensemble members
    per_model_tensors = [
        torch.cat(logit_list, dim=0) for logit_list in all_logits_per_model
    ]
    # [N, 6] averaged across all backbones
    logits_tensor = torch.stack(per_model_tensors, dim=0).mean(dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    study_ids = [_extract_study_id(p) for p in df["image_path"].tolist()]

    result = {
        "study_id":   study_ids,
        "image_path": df["image_path"].tolist(),
        "logits":     logits_tensor,
        "labels":     labels_tensor,
    }

    # Preserve SliceThickness if the CSV has it (added by build_features.py)
    if "slice_thickness" in df.columns:
        result["slice_thickness"] = torch.tensor(
            df["slice_thickness"].fillna(5.0).values, dtype=torch.float32
        )

    return result


def main() -> None:
    args = parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    num_classes = params["data"]["num_classes"]
    default_model_name = params["model"]["name"]

    # ── Resolve checkpoint list ───────────────────────────────────────────────
    if args.checkpoints:
        ckpt_paths = args.checkpoints
        model_names = args.model_names or [default_model_name] * len(ckpt_paths)
        if len(model_names) != len(ckpt_paths):
            raise ValueError(
                f"--model-names has {len(model_names)} entries but "
                f"--checkpoints has {len(ckpt_paths)}"
            )
    elif args.checkpoint:
        ckpt_paths = [args.checkpoint]
        model_names = [default_model_name]
    else:
        raise SystemExit("Provide --checkpoint or --checkpoints")

    # ── Load all models ───────────────────────────────────────────────────────
    models = [
        _load_checkpoint(ckpt_path, mname, num_classes, device)
        for ckpt_path, mname in zip(ckpt_paths, model_names)
    ]
    if len(models) > 1:
        logger.info("Ensemble of %d backbones — logits will be averaged", len(models))

    # ── Build transform ───────────────────────────────────────────────────────
    import pandas as pd
    from src.data.augmentation import build_paper_val_transforms

    image_size = params["preprocessing"]["image_size"][0]
    val_transform = build_paper_val_transforms(image_size=image_size)

    processed_dir = Path(params["data"]["processed_dir"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning("Skipping %s — %s not found", split, csv_path)
            continue

        logger.info("Extracting features for split='%s' …", split)
        df = pd.read_csv(csv_path)
        result = extract_split(
            models, df, val_transform, device,
            batch_size=params["inference"]["batch_size"],
            num_workers=params["training"]["num_workers"],
        )

        out_path = out_dir / f"features_{split}.pt"
        torch.save(result, out_path)
        has_st = "slice_thickness" in result
        logger.info(
            "Saved %d logits → %s  (shape %s, slice_thickness=%s)",
            result["logits"].shape[0], out_path,
            tuple(result["logits"].shape), has_st,
        )


if __name__ == "__main__":
    main()
