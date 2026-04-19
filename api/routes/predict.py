"""
/predict endpoint – accepts base64-encoded CT slices and returns ICH scores.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from api.schemas import (
    HemorrhageScores,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()

HEMORRHAGE_TYPES = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def _decode_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image to a float32 (H, W, 3) array in [0, 1].

    Raises:
        ValueError: If the string cannot be decoded as an image.
    """
    try:
        image_bytes = base64.b64decode(b64_string)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Run ICH detection on one or more CT slice images.

    Accepts base64-encoded images (PNG or JPEG). Each image should already
    be rendered using the appropriate CT windowing (e.g. brain + subdural +
    bone channels stacked as RGB).
    """
    app_state = request.app.state
    model = getattr(app_state, "model", None)
    transform = getattr(app_state, "transform", None)
    device = getattr(app_state, "device", None)

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    import torch

    start = time.perf_counter()

    try:
        decoded_images: List[np.ndarray] = [
            _decode_image(b64) for b64 in body.images
        ]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tensors = []
    for img in decoded_images:
        aug = transform(image=img)
        tensors.append(aug["image"])

    batch = torch.stack(tensors).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    results: List[PredictionResult] = []
    for prob_row in probs:
        prob_dict = dict(zip(HEMORRHAGE_TYPES, prob_row.tolist()))
        pred_dict = {k: float(v >= body.threshold) for k, v in prob_dict.items()}
        any_hem = any(pred_dict[k] > 0 for k in HEMORRHAGE_TYPES[1:])

        results.append(
            PredictionResult(
                probabilities=HemorrhageScores(**prob_dict),
                predictions=HemorrhageScores(**pred_dict),
                any_hemorrhage=any_hem,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        results=results,
        model_version=getattr(app_state, "model_version", "unknown"),
        processing_time_ms=round(elapsed_ms, 2),
    )
