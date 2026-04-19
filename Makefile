.PHONY: help install install-dev prepare train evaluate predict serve test lint format clean curves gradcam extract-features train-sequence

PYTHON   := python
PARAMS   := params.yaml
CKPT     := models/checkpoints/best_model.pt
METRICS  := reports/metrics/test_metrics.json

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────
install:  ## Install runtime dependencies (pip, inside active conda env)
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e . --no-deps

install-dev:  ## Install dev + runtime dependencies (pip, inside active conda env)
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e . --no-deps

conda-create:  ## Create conda environment from environment.yml
	conda env create -f environment.yml

conda-update:  ## Update conda environment after changes to environment.yml
	conda env update -f environment.yml --prune

# ── Data pipeline ─────────────────────────────────────────────────────────────
prepare:  ## Extract archives and build train/val/test CSVs
	$(PYTHON) scripts/prepare_data.py --params $(PARAMS) --labels-csv reads.csv

# ── Training ──────────────────────────────────────────────────────────────────
train:  ## Train model (uses params.yaml)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MLFLOW_DO_NOT_TRACK=true $(PYTHON) scripts/train.py --params $(PARAMS)

train-fast:  ## Quick smoke-test run (5 epochs, small images)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MLFLOW_DO_NOT_TRACK=true $(PYTHON) scripts/train.py --params $(PARAMS) --config configs/training/fast_dev.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:  ## Evaluate best checkpoint on test set
	$(PYTHON) scripts/evaluate.py --checkpoint $(CKPT) --params $(PARAMS)

# ── DVC pipeline ──────────────────────────────────────────────────────────────
pipeline:  ## Run full DVC pipeline (prepare → train → evaluate)
	dvc repro

# ── Inference ─────────────────────────────────────────────────────────────────
predict:  ## Run inference (set INPUT=path/to/scan.dcm)
	$(PYTHON) scripts/predict.py \
		--input "$(INPUT)" \
		--checkpoint "$(CKPT)" \
		--params $(PARAMS)

gradcam:  ## Grad-CAM visualisation (set INPUT=path/to/slice.dcm, CLASS=0-5, OUT=path/to/out.png)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/gradcam.py \
		--input "$(INPUT)" \
		--checkpoint "$(CKPT)" \
		--params $(PARAMS) \
		$(if $(CLASS),--class-index $(CLASS),) \
		$(if $(OUT),--output "$(OUT)",)

curves:  ## Plot train/val loss & AUC curves + print overfitting diagnosis
	$(PYTHON) scripts/plot_curves.py \
		$(if $(HISTORY),--history "$(HISTORY)",) \
		$(if $(OUT),--out "$(OUT)",)

extract-features:  ## Stage-2 prep: extract 2D CNN logits for all slices (set CKPT=)
	$(PYTHON) scripts/extract_features.py \
		--checkpoint "$(CKPT)" \
		--params $(PARAMS)

train-sequence:  ## Stage-2: train BiGRU sequence model on extracted features
	$(PYTHON) scripts/train_sequence.py \
		--params $(PARAMS) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(LR),--lr $(LR),)

# ── API ───────────────────────────────────────────────────────────────────────
serve:  ## Start FastAPI development server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-docker:  ## Build and start Docker containers
	docker compose -f docker/docker-compose.yml up --build

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow-ui:  ## Launch MLflow tracking UI
	mlflow ui --backend-store-uri mlruns/ --port 5000

# ── Testing ───────────────────────────────────────────────────────────────────
test:  ## Run all unit tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# ── Code quality ──────────────────────────────────────────────────────────────
lint:  ## Lint with ruff
	ruff check src/ tests/ scripts/ api/

format:  ## Auto-format with ruff
	ruff format src/ tests/ scripts/ api/

type-check:  ## Run mypy static type checker
	mypy src/ api/

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info
