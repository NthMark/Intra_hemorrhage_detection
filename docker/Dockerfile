# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Training image
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS trainer

LABEL maintainer="ICH Detection Team"
LABEL description="Intracranial Hemorrhage Detection – Training Image"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: API serving image (smaller, no training deps)
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS api

LABEL description="Intracranial Hemorrhage Detection – API Image"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PARAMS_PATH=/workspace/params.yaml \
    MODEL_CHECKPOINT=/workspace/models/production/best_model.pt \
    INFERENCE_DEVICE=cpu \
    ALLOWED_ORIGINS=http://localhost:3000

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY params.yaml .
COPY setup.py setup.cfg ./
RUN pip install -e . --no-deps

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
