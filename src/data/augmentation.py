"""
Albumentations-based augmentation pipeline for CT scan slices.

Two augmentation modes are provided:
  1. Paper-faithful (build_paper_train_transforms / build_paper_val_transforms)
     Matches the 1st-place RSNA 2019 ICH Detection solution augmentation:
       - Horizontal flip, ShiftScaleRotate, random erasing, random crop
       - Normalize: mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224)
     Reference: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection

  2. Original (build_train_transforms / build_val_transforms)
     Kept for backward compatibility.

Note on normalization: the paper's PNGs were in uint8 [0, 255] with
  max_pixel_value=255 .  Our DICOM pipeline produces float32 [0, 1], so we
  normalise with max_pixel_value=1.0 and the same mean/std values.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np


# ── Paper-faithful augmentation (1st-place RSNA 2019 solution) ────────────────

def _random_shift_scale_rotate(
    image: np.ndarray,
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    aspect_limit: float = 0.1,
    rotate_limit: float = 30.0,
    p: float = 0.5,
) -> np.ndarray:
    """Geometric augmentation matching the paper's randomShiftScaleRotate."""
    import cv2

    if random.random() >= p:
        return image

    height, width = image.shape[:2]
    angle = random.uniform(-rotate_limit, rotate_limit)
    scale = random.uniform(1 - scale_limit, 1 + scale_limit)
    aspect = random.uniform(1 - aspect_limit, 1 + aspect_limit)
    sx = scale * aspect / (aspect ** 0.5)
    sy = scale / (aspect ** 0.5)
    dx = round(random.uniform(-shift_limit, shift_limit) * width)
    dy = round(random.uniform(-shift_limit, shift_limit) * height)

    cc = math.cos(math.radians(angle)) * sx
    ss = math.sin(math.radians(angle)) * sy
    rotate_matrix = np.array([[cc, -ss, dx], [ss, cc, dy]])

    image = cv2.warpAffine(
        image,
        rotate_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image


def _random_erasing(
    image: np.ndarray,
    probability: float = 0.5,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
) -> np.ndarray:
    """Random erasing matching the paper's random_erasing."""
    if random.uniform(0, 1) > probability:
        return image

    for _ in range(100):
        area = image.shape[0] * image.shape[1]
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < image.shape[1] and h < image.shape[0]:
            x1 = random.randint(0, image.shape[0] - h)
            y1 = random.randint(0, image.shape[1] - w)
            image[x1 : x1 + h, y1 : y1 + w, :] = 0.0
            return image
    return image


def _random_crop(image: np.ndarray, ratio_min: float = 0.6, ratio_max: float = 0.99) -> np.ndarray:
    """Random crop-and-resize matching the paper's random_cropping."""
    import cv2

    ratio = random.uniform(ratio_min, ratio_max)
    height, width = image.shape[:2]
    target_h = int(height * ratio)
    target_w = int(width * ratio)
    start_x = random.randint(0, width - target_w)
    start_y = random.randint(0, height - target_h)
    crop = image[start_y : start_y + target_h, start_x : start_x + target_w, :]
    return cv2.resize(crop, (width, height))


def _center_crop(image: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Centre crop-and-resize matching the paper's cropping(code=0)."""
    import cv2

    height, width = image.shape[:2]
    target_h = int(height * ratio)
    target_w = int(width * ratio)
    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2
    crop = image[start_y : start_y + target_h, start_x : start_x + target_w, :]
    return cv2.resize(crop, (width, height))


class PaperTrainTransform:
    """Training transform matching the 1st-place RSNA 2019 solution.

    Pipeline:
      1. Resize to ``image_size × image_size``
      2. Random horizontal flip (p=0.5)
      3. ShiftScaleRotate (shift±10%, scale±10%, rotate±30°)
      4. Random erasing (p=0.5)
      5. Random crop (ratio 0.6–0.99)
      6. Normalize mean=(0.456,0.456,0.456) std=(0.224,0.224,0.224)
      7. ToTensorV2

    Input: HxWx3 float32 in [0, 1].
    Output: 3xHxW float32 tensor (normalised).
    """

    def __init__(self, image_size: int = 512) -> None:
        self.image_size = image_size
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError as exc:
            raise ImportError("albumentations is required: pip install albumentations") from exc

        # Resize + normalize are handled by albumentations; geometric augs done manually
        self._resize_norm = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray) -> dict:
        import cv2

        # 1. Resize to target
        h, w = image.shape[:2]
        if (h, w) != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))

        # 2. Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        # 3. ShiftScaleRotate
        image = _random_shift_scale_rotate(image, p=0.5)

        # 4. Random erasing
        image = _random_erasing(image, probability=0.5)

        # 5. Random crop + resize back
        image = _random_crop(image, ratio_min=0.6, ratio_max=0.99)

        # 6. Normalize + ToTensor via albumentations
        out = self._resize_norm(image=image)
        return out  # {"image": tensor}


class PaperValTransform:
    """Validation transform matching the 1st-place RSNA 2019 solution.

    Pipeline:
      1. Resize to ``image_size × image_size``
      2. Centre crop (ratio=0.8) + resize back
      3. Normalize mean=(0.456,0.456,0.456) std=(0.224,0.224,0.224)
      4. ToTensorV2

    Input: HxWx3 float32 in [0, 1].
    Output: 3xHxW float32 tensor (normalised).
    """

    def __init__(self, image_size: int = 512) -> None:
        self.image_size = image_size
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError as exc:
            raise ImportError("albumentations is required: pip install albumentations") from exc

        self._resize_norm = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray) -> dict:
        import cv2

        h, w = image.shape[:2]
        if (h, w) != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))

        # Centre crop then resize back (paper's inference augmentation)
        image = _center_crop(image, ratio=0.8)

        out = self._resize_norm(image=image)
        return out


def build_paper_train_transforms(image_size: int = 512) -> PaperTrainTransform:
    """Return the paper-faithful training transform."""
    return PaperTrainTransform(image_size=image_size)


def build_paper_val_transforms(image_size: int = 512) -> PaperValTransform:
    """Return the paper-faithful validation/test transform."""
    return PaperValTransform(image_size=image_size)


# ── Original albumentations-based augmentation (kept for compatibility) ───────

def build_train_transforms(
    image_size: int = 512,
    rotation_limit: int = 15,
    shift_limit: float = 0.05,
    scale_limit: float = 0.10,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
) -> Callable:
    """Build an albumentations Compose transform for training.

    Args:
        image_size: Target square size after resize.
        rotation_limit: Max rotation in degrees.
        shift_limit: Max fraction shift for ShiftScaleRotate.
        scale_limit: Max fraction scale for ShiftScaleRotate.
        brightness_limit: Limit for RandomBrightnessContrast.
        contrast_limit: Limit for RandomBrightnessContrast.

    Returns:
        An albumentations Compose object.

    Raises:
        ImportError: If albumentations is not installed.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError(
            "albumentations is required: pip install albumentations"
        ) from exc

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-shift_limit, shift_limit), "y": (-shift_limit, shift_limit)},
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                cval=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, image_size // 16),
                hole_width_range=(1, image_size // 16),
                fill=0,
                p=0.3,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def build_val_transforms(image_size: int = 512) -> Callable:
    """Build a deterministic albumentations Compose transform for validation/test.

    Args:
        image_size: Target square size after resize.

    Returns:
        An albumentations Compose object.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError(
            "albumentations is required: pip install albumentations"
        ) from exc

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )
