#!/usr/bin/env python
"""
prepare_data.py – Extract CQ500 archives and build train/val/test CSV splits.

Usage:
    python scripts/prepare_data.py \
        --dataset-dir dataset/ \
        --raw-dir data/raw/ \
        --processed-dir data/processed/ \
        [--labels-csv path/to/labels.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ICH dataset")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--labels-csv", default=None)
    return parser.parse_args()


def extract_archives(dataset_dir: Path, raw_dir: Path) -> None:
    """Unzip all .zip archives from dataset_dir into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(dataset_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.testzip()  # Check for corrupt files
        except zipfile.BadZipFile:
            logger.error("Corrupt ZIP file skipped: %s", zip_path)
            continue
        logger.info("Extracting %s → %s", zip_path.name, raw_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)


def main() -> None:
    args = parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    dataset_dir = Path(args.dataset_dir or params["data"]["dataset_dir"])
    raw_dir = Path(args.raw_dir or params["data"]["raw_dir"])
    processed_dir = Path(args.processed_dir or params["data"]["processed_dir"])
    labels_csv = Path(args.labels_csv) if args.labels_csv else None

    # Step 1: Extract
    extract_archives(dataset_dir, raw_dir)

    # Step 2: Build metadata CSV
    from src.features.build_features import build_metadata_csv, split_dataframe

    meta_csv = processed_dir / "metadata.csv"
    df = build_metadata_csv(raw_dir, meta_csv, labels_csv=labels_csv)

    # Step 3: Split
    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=params["data"]["train_split"],
        val_ratio=params["data"]["val_split"],
        seed=params["base"]["random_seed"],
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    logger.info(
        "Split sizes → train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )


if __name__ == "__main__":
    main()
