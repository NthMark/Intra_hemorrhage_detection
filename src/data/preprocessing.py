"""
CT scan preprocessing utilities for intracranial hemorrhage detection.

Supports DICOM ingestion, HU windowing, and multi-channel slice generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CQ500 / RSNA hemorrhage subtypes
HEMORRHAGE_TYPES: List[str] = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

# Standard CT windowing presets: (center, width)
CT_WINDOWS = {
    "brain": (40, 80),
    "subdural": (75, 215),
    "bone": (600, 2800),
}


def apply_window(
    hu_array: np.ndarray,
    center: int,
    width: int,
) -> np.ndarray:
    """Apply CT windowing and scale to [0, 1].

    Args:
        hu_array: CT slice in Hounsfield Units.
        center: Window center (level).
        width: Window width.

    Returns:
        Windowed image normalised to [0, 1].
    """
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(hu_array, low, high)
    return (windowed - low) / (high - low)


def hu_to_3channel(hu_array: np.ndarray) -> np.ndarray:
    """Convert a single HU slice to a 3-channel image using standard CT windows.

    Channels: [brain window, subdural window, bone window].

    Args:
        hu_array: 2-D CT slice in Hounsfield Units, shape (H, W).

    Returns:
        3-channel float32 image, shape (H, W, 3), values in [0, 1].
    """
    channels = [
        apply_window(hu_array, *CT_WINDOWS["brain"]),
        apply_window(hu_array, *CT_WINDOWS["subdural"]),
        apply_window(hu_array, *CT_WINDOWS["bone"]),
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def load_dicom_slice(dcm_path: Path) -> Tuple[np.ndarray, dict]:
    """Load a DICOM file and return the HU pixel array plus metadata.

    Args:
        dcm_path: Path to a .dcm file.

    Returns:
        Tuple of (hu_array [H, W], metadata dict).

    Raises:
        ImportError: If pydicom is not installed.
        FileNotFoundError: If the DICOM file does not exist.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required: pip install pydicom") from exc

    if not dcm_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dcm_path}")

    dcm = pydicom.dcmread(str(dcm_path))
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Apply rescale slope / intercept to get HU
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    hu_array = pixel_array * slope + intercept

    metadata = {
        "patient_id": getattr(dcm, "PatientID", ""),
        "study_uid": getattr(dcm, "StudyInstanceUID", ""),
        "series_uid": getattr(dcm, "SeriesInstanceUID", ""),
        "sop_uid": getattr(dcm, "SOPInstanceUID", ""),
        "slice_location": float(getattr(dcm, "SliceLocation", 0)),
        "pixel_spacing": list(getattr(dcm, "PixelSpacing", [1.0, 1.0])),
        "image_path": str(dcm_path),
    }
    return hu_array, metadata


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """ImageNet-style normalisation for a (H, W, 3) float32 image in [0, 1].

    Args:
        image: Input image array, shape (H, W, 3).
        mean: Per-channel means.
        std: Per-channel standard deviations.

    Returns:
        Normalised float32 image.
    """
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    return (image - mean_arr) / std_arr


def clip_hu(
    hu_array: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 3000.0,
) -> np.ndarray:
    """Clip HU values to a physiologically meaningful range."""
    return np.clip(hu_array, hu_min, hu_max)


def adjacent_slices_to_3channel(
    hu_prev: Optional[np.ndarray],
    hu_curr: np.ndarray,
    hu_next: Optional[np.ndarray],
    window_center: int = 40,
    window_width: int = 80,
) -> np.ndarray:
    """Stack three adjacent CT slices as a single 3-channel image.

    This is the input strategy used by the 1st-place RSNA ICH Detection
    solution (https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection).
    Instead of applying three different windowing presets to one slice, it
    uses the *same window* on the previous, current, and next slice, giving
    the model implicit 3-D context without a volumetric backbone.

    Boundary slices (no previous / next neighbour) repeat the existing edge
    slice so the channel count stays constant.

    Args:
        hu_prev: HU array of the slice immediately before ``hu_curr``.
            Pass ``None`` if ``hu_curr`` is the first slice in the series.
        hu_curr: HU array of the target slice, shape (H, W).
        hu_next: HU array of the slice immediately after ``hu_curr``.
            Pass ``None`` if ``hu_curr`` is the last slice in the series.
        window_center: CT window center in HU (default: 40 — brain window).
        window_width: CT window width in HU  (default: 80 — brain window).

    Returns:
        Float32 image of shape (H, W, 3) with values in [0, 1].
        Channel 0 = previous slice, channel 1 = current, channel 2 = next.
    """
    if hu_prev is None:
        hu_prev = hu_curr  # pad with edge slice
    if hu_next is None:
        hu_next = hu_curr  # pad with edge slice

    channels = [
        apply_window(hu_prev, window_center, window_width),
        apply_window(hu_curr, window_center, window_width),
        apply_window(hu_next, window_center, window_width),
    ]
    return np.stack(channels, axis=-1).astype(np.float32)
