"""
Pydantic schemas for the ICH detection REST API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for batch prediction via base64-encoded images."""

    images: List[str] = Field(
        ...,
        description="List of base64-encoded PNG/JPEG images (after CT windowing).",
        min_length=1,
        max_length=32,
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Binary classification threshold.",
    )


class HemorrhageScores(BaseModel):
    no_hemorrhage: float
    epidural: float
    intraparenchymal: float
    intraventricular: float
    subarachnoid: float
    subdural: float


class PredictionResult(BaseModel):
    """Per-image prediction output."""

    probabilities: HemorrhageScores
    predictions: HemorrhageScores
    any_hemorrhage: bool


class PredictionResponse(BaseModel):
    results: List[PredictionResult]
    model_version: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str
