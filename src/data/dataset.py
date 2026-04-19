"""
PyTorch Dataset classes for intracranial hemorrhage detection.

Supports two training modes:
  1. Slice-based (ICHDataset) — one sample per CSV row.
  2. Study-based (ICHStudyDataset / ICHStudyValDataset) — matches the
     1st-place RSNA 2019 solution's RSNA_Dataset_train_by_study_context
     strategy: sample a random slice from a random study, apply three CT
     window settings (brain, subdural, bone) as the three RGB channels.

Reference:
    Wang et al., 1st-place RSNA 2019 ICH Detection.
    https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES: List[str] = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


class ICHDataset:
    """Dataset for intracranial hemorrhage CT slices.

    Expects a CSV with columns:
        image_path, no_hemorrhage, epidural, intraparenchymal,
        intraventricular, subarachnoid, subdural

    Args:
        df: DataFrame with image paths and multi-hot labels.
        transform: Image transform callable (albumentations Compose).
        preload: Whether to preload all images into RAM.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        preload: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = df[HEMORRHAGE_TYPES].values.astype(np.float32)
        self._cache: Dict[int, np.ndarray] = {}

        if preload:
            logger.info("Pre-loading %d images into RAM…", len(self.df))
            for idx in range(len(self.df)):
                self._cache[idx] = self._load_image(idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """Return (image_tensor, label_tensor) for an index."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required: pip install torch") from exc

        image = self._cache.get(idx) or self._load_image(idx)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def _load_image(self, idx: int) -> np.ndarray:
        from src.data.preprocessing import clip_hu, hu_to_3channel, load_dicom_slice

        image_path = Path(self.df.loc[idx, "image_path"])

        if image_path.suffix.lower() == ".dcm":
            hu_array, _ = load_dicom_slice(image_path)
            hu_array = clip_hu(hu_array)
            image = hu_to_3channel(hu_array)
        else:
            # Assume pre-processed PNG/JPG
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            image = np.array(pil_img, dtype=np.float32) / 255.0

        return image


# ── Helper utilities ──────────────────────────────────────────────────────────

def _extract_study_id(image_path: str) -> str:
    """Extract the study identifier from a CQ500 image path.

    E.g. 'data/raw/CQ500CT252 CQ500CT252/Unknown Study/.../CT000088.dcm'
         → 'CQ500CT252'
    """
    parts = Path(image_path).parts
    # First folder component after 'raw/' contains 'CQ500CTxxx CQ500CTxxx'
    for part in parts:
        m = re.match(r"(CQ500CT\d+)", part)
        if m:
            return m.group(1)
    # Fallback: use parent folder name
    return Path(image_path).parent.name


def _extract_slice_index(image_path: str) -> int:
    """Extract numeric slice index from filename, e.g. CT000088.dcm → 88."""
    stem = Path(image_path).stem          # 'CT000088'
    digits = re.search(r"(\d+)$", stem)
    return int(digits.group(1)) if digits else 0


def _build_weighted_index_list(length: int) -> List[int]:
    """Return a list with slice indices weighted toward the centre.

    Slices near the first/last position are under-represented so the model
    sees more brain content than skull-cap/base.  Matches the paper's
    ``generate_random_list``.
    """
    result: List[int] = []
    for i in range(length):
        if i <= length / 2:
            weight = max(1, i // 4)
        else:
            weight = max(1, (length - i) // 4)
        result.extend([i] * weight)
    return result


# ── Study-based datasets (1st-place RSNA solution strategy) ──────────────────


class ICHStudyDataset:
    """Training dataset that samples one random slice per study each epoch.

    Matches ``RSNA_Dataset_train_by_study_context`` from the 1st-place
    RSNA 2019 ICH Detection solution.  For each call to ``__getitem__``:
      1. A study is selected by index (modulo study count).
      2. A weighted-random slice is chosen (middle slices more likely).
      3. The slice is loaded and converted to a 3-channel image by applying
         three CT window settings (brain, subdural, bone), each as one
         channel — exactly as described in Wang et al. 2021.

    The dataset length is ``num_studies × 4`` so each study is seen ~4 times
    per epoch (same as the reference implementation).

    Args:
        df: DataFrame with columns ``image_path`` + 6 label columns.
        transform: Albumentations Compose applied to the final HxWx3 array.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

        # Build study → ordered list of row-dicts (sorted by slice index)
        study_rows: Dict[str, List[dict]] = defaultdict(list)
        for _, row in self.df.iterrows():
            sid = _extract_study_id(row["image_path"])
            study_rows[sid].append(row.to_dict())

        self.studies: List[str] = sorted(study_rows.keys())
        self.study_rows: Dict[str, List[dict]] = {
            sid: sorted(rows, key=lambda r: _extract_slice_index(r["image_path"]))
            for sid, rows in study_rows.items()
        }
        logger.info(
            "ICHStudyDataset: %d studies, %d slices total (len=%d)",
            len(self.studies), len(self.df), len(self),
        )

    def __len__(self) -> int:
        return len(self.studies) * 4

    def __getitem__(self, idx: int) -> Tuple:
        import torch
        from src.data.preprocessing import clip_hu, hu_to_3channel, load_dicom_slice

        study_id = self.studies[idx % len(self.studies)]
        slices = self.study_rows[study_id]
        n = len(slices)

        # Weighted-random slice index (favours centre)
        weighted = _build_weighted_index_list(n - 1) if n > 1 else [0]
        s_idx = random.choice(weighted)

        row_curr = slices[s_idx]

        # Load the single slice and apply three CT window settings to produce
        # a 3-channel RGB image, exactly as described in Wang et al. 2021:
        # "applying the three window settings and then converting each result
        # to an 8-bit grayscale image [...] assembled as the three channels
        # of an RGB image."
        p = Path(row_curr["image_path"])
        if p.suffix.lower() == ".dcm":
            hu, _ = load_dicom_slice(p)
            hu = clip_hu(hu)
            image = hu_to_3channel(hu)          # (H, W, 3): brain/subdural/bone
        else:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(p).convert("RGB"), dtype=np.float32) / 255.0
            image = img                          # (H, W, 3) fallback

        # Only resize here if no transform is set and image_size is specified
        if self.transform is None and self.image_size is not None:
            h, w = self.image_size
            if image.shape[:2] != (h, w):
                import cv2
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(
            [row_curr[c] for c in HEMORRHAGE_TYPES], dtype=torch.float32
        )
        return image, label


class ICHStudyValDataset:
    """Validation/test dataset — iterates every slice in every study.

    Matches ``RSNA_Dataset_val_by_study_context`` from the 1st-place solution.
    Every slice is evaluated exactly once, converted to a 3-channel image
    using the brain, subdural, and bone CT windows (paper method).

    Args:
        df: DataFrame with columns ``image_path`` + 6 label columns.
        transform: Albumentations Compose applied to the final HxWx3 array.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

        # Build study → ordered slice list for adjacency lookup
        study_rows: Dict[str, List[dict]] = defaultdict(list)
        for _, row in self.df.iterrows():
            sid = _extract_study_id(row["image_path"])
            study_rows[sid].append(row.to_dict())

        self.study_rows: Dict[str, List[dict]] = {
            sid: sorted(rows, key=lambda r: _extract_slice_index(r["image_path"]))
            for sid, rows in study_rows.items()
        }

        # Flat ordered list of (study_id, local_index) for __getitem__
        self._index: List[Tuple[str, int]] = []
        for sid, rows in self.study_rows.items():
            for i in range(len(rows)):
                self._index.append((sid, i))

        logger.info(
            "ICHStudyValDataset: %d slices across %d studies",
            len(self._index), len(self.study_rows),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple:
        import torch
        from src.data.preprocessing import clip_hu, hu_to_3channel, load_dicom_slice

        study_id, s_idx = self._index[idx]
        slices = self.study_rows[study_id]

        row_curr = slices[s_idx]

        # Load single slice → three CT windows → 3-channel image (paper method)
        p = Path(row_curr["image_path"])
        if p.suffix.lower() == ".dcm":
            hu, _ = load_dicom_slice(p)
            hu = clip_hu(hu)
            image = hu_to_3channel(hu)          # (H, W, 3): brain/subdural/bone
        else:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(p).convert("RGB"), dtype=np.float32) / 255.0
            image = img

        # Only resize here if no transform is set and image_size is specified
        if self.transform is None and self.image_size is not None:
            import cv2
            target_h, target_w = self.image_size
            img_h, img_w = image.shape[:2]
            if (img_h, img_w) != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(
            [row_curr[c] for c in HEMORRHAGE_TYPES], dtype=torch.float32
        )
        return image, label


def build_study_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Optional[Callable],
    val_transform: Optional[Callable],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple:
    """Build study-based train / val / test DataLoaders (paper method).

    Uses ICHStudyDataset for training and ICHStudyValDataset for val/test.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    train_ds = ICHStudyDataset(train_df, transform=train_transform, image_size=image_size)
    val_ds = ICHStudyValDataset(val_df, transform=val_transform, image_size=image_size)
    test_ds = ICHStudyValDataset(test_df, transform=val_transform, image_size=image_size)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    return train_loader, val_loader, test_loader


# ── Original slice-based dataloaders (kept for compatibility) ─────────────────

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Optional[Callable],
    val_transform: Optional[Callable],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple:
    """Build train / val / test DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required: pip install torch") from exc

    train_ds = ICHDataset(train_df, transform=train_transform)
    val_ds = ICHDataset(val_df, transform=val_transform)
    test_ds = ICHDataset(test_df, transform=val_transform)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    return train_loader, val_loader, test_loader
