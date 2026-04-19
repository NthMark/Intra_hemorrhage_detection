"""
Data preparation: extract CQ500 DICOM archives and build metadata CSV.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

# Mapping from reads.csv column prefixes to our label names
# Each type has 3 reader columns (R1, R2, R3); majority vote (>=2) is used
_READS_COL_MAP = {
    "ICH": None,           # used to derive no_hemorrhage (inverse)
    "EDH": "epidural",
    "IPH": "intraparenchymal",
    "IVH": "intraventricular",
    "SAH": "subarachnoid",
    "SDH": "subdural",
}


def _normalize_study_name(name: str) -> str:
    """Convert a study name to a consistent key for matching.

    Handles two formats:
    - reads.csv:  'CQ500-CT-1'      → 'CQ500CT1'
    - directory:  'CQ500CT1 CQ500CT1' (repeated) → 'CQ500CT1'

    Strategy: take the first whitespace-delimited token, then strip hyphens.
    """
    first_token = name.strip().split()[0]
    return re.sub(r"-", "", first_token).upper()


def _parse_reads_csv(labels_csv: Path) -> pd.DataFrame:
    """Parse reads.csv into a study-level DataFrame with majority-vote labels.

    For each hemorrhage type, three radiologist readings (R1, R2, R3) are
    combined via majority vote (label=1 if at least 2 out of 3 readers agree).

    Returns:
        DataFrame with columns: study_key, no_hemorrhage, epidural,
        intraparenchymal, intraventricular, subarachnoid, subdural
    """
    raw = pd.read_csv(labels_csv)

    records = []
    for _, row in raw.iterrows():
        study_key = _normalize_study_name(str(row["name"]))
        entry: dict = {"study_key": study_key}

        for col_suffix, label_name in _READS_COL_MAP.items():
            if label_name is None:
                continue
            votes = [
                int(row.get(f"R{r}:{col_suffix}", 0) or 0)
                for r in (1, 2, 3)
            ]
            entry[label_name] = 1 if sum(votes) >= 2 else 0

        # no_hemorrhage = 1 if majority (>=2/3) of readers agree there is NO ICH
        ich_votes = [
            int(row.get(f"R{r}:ICH", 0) or 0)
            for r in (1, 2, 3)
        ]
        entry["no_hemorrhage"] = 1 if sum(ich_votes) < 2 else 0

        records.append(entry)

    return pd.DataFrame(records)


def build_metadata_csv(
    raw_dir: Path,
    output_path: Path,
    labels_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Walk raw_dir for DICOM files and optionally merge with a labels CSV.

    Args:
        raw_dir: Directory containing extracted DICOM studies.
        output_path: Destination path for the output metadata CSV.
        labels_csv: Optional path to reads.csv (CQ500 format) with per-study labels.

    Returns:
        Metadata DataFrame saved to output_path.
    """
    records = []
    for dcm_path in sorted(raw_dir.rglob("*.dcm")):
        # Extract study key from directory name (e.g. 'CQ500CT1 CQ500CT1' → 'CQ500CT1')
        study_dir_name = dcm_path.parts[len(raw_dir.parts)]
        study_key = _normalize_study_name(study_dir_name)

        # Extract SliceThickness from DICOM header (paper uses it as sequence-model input)
        slice_thickness = 5.0
        try:
            import pydicom  # optional; install with:  pip install pydicom
            dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            raw_st = getattr(dcm, "SliceThickness", None)
            if raw_st is not None:
                slice_thickness = float(raw_st)
        except Exception:
            pass  # fall back to 5.0 mm default

        records.append({
            "image_path": str(dcm_path),
            "study_key": study_key,
            "slice_thickness": slice_thickness,
        })

    df = pd.DataFrame(records)

    # Initialise label columns with zeros
    for col in HEMORRHAGE_TYPES:
        df[col] = 0

    if labels_csv is not None and Path(labels_csv).exists():
        study_labels = _parse_reads_csv(Path(labels_csv))
        df = df.merge(study_labels, on="study_key", how="left", suffixes=("", "_lbl"))
        for col in HEMORRHAGE_TYPES:
            if f"{col}_lbl" in df.columns:
                df[col] = df[f"{col}_lbl"].fillna(0).astype(int)
                df.drop(columns=[f"{col}_lbl"], inplace=True)
        logger.info(
            "Labels merged: %d/%d slices matched a study",
            df[HEMORRHAGE_TYPES[0]].notna().sum(), len(df),
        )

    df.drop(columns=["study_key"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved metadata CSV with %d rows → %s", len(df), output_path)
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Study-level stratified split to prevent data leakage.

    Splits on unique CT studies (not individual slices), so no study appears
    in more than one of train/val/test.  Stratification is by whether the
    study contains any hemorrhage (majority-vote label).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    import re
    from sklearn.model_selection import train_test_split

    df = df.copy()
    df["any_hemorrhage"] = (df[HEMORRHAGE_TYPES[1:]].sum(axis=1) > 0).astype(int)

    # ── Build one row per study ───────────────────────────────────────────────
    def _study_id(path: str) -> str:
        for part in Path(path).parts:
            m = re.match(r"(CQ500CT\d+)", part)
            if m:
                return m.group(1)
        return Path(path).parent.name

    df["study_id"] = df["image_path"].apply(_study_id)

    # One label per study (any slice positive → study is positive)
    study_labels = (
        df.groupby("study_id")["any_hemorrhage"].max().reset_index()
    )

    # ── Study-level split ─────────────────────────────────────────────────────
    test_ratio = 1.0 - train_ratio - val_ratio
    train_studies, valtest_studies = train_test_split(
        study_labels,
        test_size=(val_ratio + test_ratio),
        stratify=study_labels["any_hemorrhage"],
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_studies, test_studies = train_test_split(
        valtest_studies,
        test_size=(1.0 - relative_val),
        stratify=valtest_studies["any_hemorrhage"],
        random_state=seed,
    )

    # ── Map back to slice rows ────────────────────────────────────────────────
    train_ids = set(train_studies["study_id"])
    val_ids = set(val_studies["study_id"])
    test_ids = set(test_studies["study_id"])

    train_df = df[df["study_id"].isin(train_ids)].drop(columns=["study_id"])
    val_df = df[df["study_id"].isin(val_ids)].drop(columns=["study_id"])
    test_df = df[df["study_id"].isin(test_ids)].drop(columns=["study_id"])

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
