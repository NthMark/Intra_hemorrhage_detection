"""
FastAPI application for ICH Detection & Classification REST API.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router as predict_router
from api.schemas import HealthResponse
from src import __version__

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; release resources on shutdown."""
    params_path = os.environ.get("PARAMS_PATH", "params.yaml")
    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT", "models/checkpoints/best_model.pt"
    )

    with open(params_path) as f:
        params = yaml.safe_load(f)

    device_str = os.environ.get("INFERENCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

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
    from src.data.augmentation import build_val_transforms

    model = load_model(Path(checkpoint_path), model, device)
    transform = build_val_transforms(image_size=params["preprocessing"]["image_size"][0])

    app.state.model = model
    app.state.transform = transform
    app.state.device = device
    app.state.model_name = model_name
    app.state.model_version = os.environ.get("MODEL_VERSION", "1.0.0")

    logger.info("Model loaded: %s on %s", model_name, device)
    yield

    # Cleanup
    del app.state.model
    torch.cuda.empty_cache()
    logger.info("Model unloaded.")


app = FastAPI(
    title="Intracranial Hemorrhage Detection API",
    description=(
        "REST API for automated detection and classification of intracranial "
        "hemorrhage subtypes from CT scan slices using deep learning."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – restrict to known origins in production via environment variable
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(predict_router, prefix="/api/v1", tags=["Inference"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness / readiness probe endpoint."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=getattr(app.state, "model_name", ""),
        device=str(getattr(app.state, "device", "cpu")),
        version=__version__,
    )
