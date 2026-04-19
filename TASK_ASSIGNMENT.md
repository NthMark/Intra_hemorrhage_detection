# Team Task Assignment

This document describes the responsibilities of each team member in the ICH Detection project, following the 1st-place RSNA 2019 pipeline (Wang et al., 2021).

---

## Overview

The project is divided into three areas that map directly onto the pipeline stages:

```
Member 1                    Member 2                    Member 3
─────────────────────       ─────────────────────       ─────────────────────
Data & Preprocessing   →    Stage 1: CNN Training   →   Stage 2: Sequence
                                                         Model + Deployment
     dataset/               src/models/                 src/models/
     src/data/              architectures/              sequence_model.py
     notebooks/             train.py                    scripts/train_sequence.py
     scripts/               schedulers.py               scripts/extract_features.py
     prepare_data.py        scripts/train.py            api/
                            scripts/evaluate.py         docker/
```

---

## Member 1 — Data & Preprocessing

### Responsibility
Owns everything from raw DICOM files to clean, model-ready tensors. Produces the CSV splits that Members 2 and 3 depend on.

### Files owned

| File / Folder | Purpose |
|---|---|
| `scripts/prepare_data.py` | Main pipeline: extract ZIPs → build train/val/test CSVs |
| `src/data/preprocessing.py` | DICOM loading, HU conversion, windowing, adjacent-slice stacking |
| `src/data/augmentation.py` | Training and validation transform pipelines |
| `src/data/dataset.py` | `ICHStudyDataset`, `ICHStudyValDataset`, `build_study_dataloaders` |
| `src/features/build_features.py` | Metadata extraction and stratified splitting logic |
| `notebooks/01_dicom_visualization.ipynb` | DICOM exploration and sanity checks |
| `notebooks/02_preprocessing_deep_dive.ipynb` | Full preprocessing walkthrough with visualisations |

### Key tasks

1. **Data extraction** — unzip CQ500 archives into `data/raw/`, verify all DICOM files are readable and complete
2. **Label generation** — parse `reads.csv`, apply majority vote (≥ 2/3 readers), broadcast study-level labels to every slice
3. **Stratified splitting** — split 491 studies into train/val/test (70/15/15) ensuring hemorrhage subtypes are represented proportionally in each split
4. **HU preprocessing pipeline**:
   - Load DICOM pixel arrays
   - Apply `RescaleSlope` / `RescaleIntercept` → Hounsfield Units
   - Clip to [−1000, 3000] HU (remove physiologically meaningless extremes)
   - Apply brain window (C=40, W=80) to three adjacent slices (s−1, s, s+1)
   - Stack as 3-channel float32 image [0, 1]
5. **Augmentation pipeline** — implement paper's `aug_image`: HFlip + ShiftScaleRotate + RandomErase + RandomCrop for training; centre crop for validation
6. **Study-based dataset** — implement `ICHStudyDataset` with centre-weighted slice sampling (paper's `generate_random_list`)
7. **Data quality audit** — run notebook section 1 to identify corrupted files, missing metadata, dimension outliers
8. **Class imbalance analysis** — compute `pos_weight` per class (notebook section 11), check label distribution across splits

### Definition of done
- `make prepare` runs without errors
- `data/processed/train.csv`, `val.csv`, `test.csv` exist with correct columns and label distributions
- Notebook 02 runs end-to-end and all visualisations render correctly
- Member 2 can call `build_study_dataloaders()` and get batches with shape `(B, 3, 384, 384)`

---

## Member 2 — Stage 1: Model Architecture & Training

### Responsibility
Owns the CNN backbone, training loop, scheduler, and evaluation metrics. Takes the dataloaders from Member 1 and produces a trained checkpoint for Member 3.

### Files owned

| File / Folder | Purpose |
|---|---|
| `src/models/architectures/densenet.py` | DenseNet121ICH and DenseNet169ICH backbones (paper method) |
| `src/models/architectures/efficientnet.py` | EfficientNet backbone (alternative) |
| `src/models/architectures/resnet.py` | ResNet backbone (alternative) |
| `src/models/schedulers.py` | WarmRestart cosine-annealing scheduler (paper's SGDR) |
| `src/models/train.py` | Full training loop: loss, optimizer, scheduler, MLflow, early stopping |
| `src/models/evaluate.py` | AUC, sensitivity, specificity metrics |
| `scripts/train.py` | CLI entry point for Stage 1 training |
| `scripts/evaluate.py` | Test-set evaluation → `reports/metrics/test_metrics.json` |
| `scripts/plot_curves.py` | Loss/AUC curve plotting and overfit diagnosis |
| `params.yaml` | All hyperparameters (shared, but Member 2 owns model/training sections) |

### Key tasks

1. **DenseNet backbone** — implement `DenseNet121ICH` with `features → ReLU → AdaptiveAvgPool2d(1) → Linear(1024, 6)` head matching paper's `DenseNet121_change_avg`; include `get_features()` method for Stage 2
2. **Loss function** — `BCEWithLogitsLoss` with configurable `pos_weight` (primary); `FocalLoss` (fallback)
3. **Optimizer** — Adam with lr=5×10⁻⁴, weight_decay=2×10⁻⁵, eps=1×10⁻⁸ (paper's exact values)
4. **WarmRestart scheduler** — epochs 1–10: constant LR (warmup plateau); epoch 11+: `scheduler.step()` + `warm_restart(T_mult=2)` each epoch
5. **Validation interval** — validate every 5 epochs (`val_interval=5` in `params.yaml`) to avoid ~75K DICOM reads per epoch
6. **Training loop** — mixed precision (AMP), gradient clipping, MLflow logging, early stopping, checkpoint saving
7. **Metrics** — per-class AUC (ROC), macro-average AUC, sensitivity and specificity at threshold 0.5
8. **Experiment tracking** — log all runs to MLflow; compare hyperparameter experiments using `mlflow ui`
9. **Curve analysis** — after training, run `make curves` to check for overfitting/underfitting

### Hyperparameters to tune (in `params.yaml`)

| Parameter | Paper value | Notes |
|---|---|---|
| `model.name` | `densenet121` | Can also try `densenet169` |
| `training.epochs` | 80 | Paper trains for 80 epochs |
| `training.batch_size` | 16 | Fits 7.65 GiB GPU at 384×384 |
| `optimizer.lr` | 5e-4 | Paper's exact value |
| `scheduler.T_max` | 5 | Cosine cycle length |
| `training.val_interval` | 5 | Validate every N epochs |

### Definition of done
- `make train` completes and saves `models/checkpoints/best_model.pt`
- `make evaluate` produces `reports/metrics/test_metrics.json` with per-class AUC values
- `make curves` shows training curves without severe overfitting
- Member 3 can load `best_model.pt` and call `model.get_features(batch)` to extract logits

---

## Member 3 — Stage 2: Sequence Model + Deployment

### Responsibility
Owns the BiGRU sequence model that refines Stage 1 predictions using full-scan context, and the Docker-based API that serves the final model to reviewers/users.

### Files owned

| File / Folder | Purpose |
|---|---|
| `src/models/sequence_model.py` | BiGRU Stage-2 model (Model 1 + Model 2 with skip connections) |
| `scripts/extract_features.py` | Run trained CNN on all splits → save per-slice logits as `.pt` files |
| `scripts/train_sequence.py` | Train BiGRU on extracted logit sequences |
| `api/main.py` | FastAPI application entrypoint + model loading |
| `api/schemas.py` | Pydantic request/response schemas |
| `api/routes/predict.py` | `/api/v1/predict` endpoint |
| `docker/Dockerfile` | Multi-stage build: trainer + api targets |
| `docker/docker-compose.yml` | Docker Compose: API + MLflow UI services |
| `scripts/gradcam.py` | Grad-CAM visualisation of model attention |
| `Makefile` (deployment targets) | `make serve`, `make serve-docker`, `make gradcam` |

### Key tasks

1. **Feature extraction** — run `make extract-features CKPT=models/checkpoints/best_model.pt` to generate `data/processed/features_{train,val,test}.pt` (per-slice logits and labels from Member 2's model)
2. **Sequence dataset** — implement `StudySequenceDataset`: group rows by study, pad/crop to `seq_len=24`, apply random windowing during training
3. **BiGRU model** — two sub-models summed together:
   - Model 1: `FC(6→64→32) + BiGRU(hidden=96) → Linear` 
   - Model 2: `Conv1D + BiGRU with skip connection → Linear`
4. **Stage 2 training** — Adam + CosineAnnealingLR for 40 epochs; save `models/checkpoints/best_sequence_model.pt`
5. **Grad-CAM** — produce saliency map overlays on CT slices showing which regions the CNN focused on; save to `reports/figures/gradcam.png`
6. **FastAPI endpoint** — `/api/v1/predict` accepts a DICOM file upload, runs preprocessing + CNN + sequence model, returns JSON with per-class probabilities
7. **Docker packaging** — build image with `make serve-docker`; verify the teacher can run it with only Docker installed (no Python/conda needed)
8. **Submission checklist**:
   - `models/checkpoints/best_model.pt` — Stage 1 checkpoint (from Member 2)
   - `models/checkpoints/best_sequence_model.pt` — Stage 2 checkpoint
   - `reports/metrics/test_metrics.json` — final test AUC scores
   - `reports/figures/loss_curves.png` — training curves
   - `reports/figures/gradcam.png` — sample Grad-CAM output
   - Docker image builds and API starts with `make serve-docker`

### Definition of done
- `make extract-features` produces `.pt` feature files for all splits
- `make train-sequence` completes and saves `best_sequence_model.pt`
- `make gradcam` produces a valid PNG with visible saliency overlay
- `make serve-docker` starts with no errors; `http://localhost:8000/docs` loads; a DICOM file can be uploaded and returns predictions
- Teacher can reproduce the full inference: clone repo → place checkpoint → `make serve-docker` → upload DICOM → get predictions

---

## Dependency Timeline

```
Week 1
  Member 1: Data audit → label generation → stratified split → CSVs ready
  Member 2: Implement DenseNet backbone, loss, scheduler (can use synthetic data)
  Member 3: Implement BiGRU model, FastAPI skeleton (no model weights yet)

Week 2
  Member 1: Augmentation pipeline, study-based dataset, class imbalance analysis
  Member 2: Full training run (depends on Member 1's dataloaders) → checkpoint
  Member 3: Feature extraction (depends on Member 2's checkpoint) → Stage 2 training

Week 3
  Member 2: Evaluate, plot curves, tune hyperparameters if needed
  Member 3: Grad-CAM, Docker packaging, submission checklist
  Member 1: Assist with preprocessing bugs, update notebooks with results
```

---

## Shared Responsibilities

| Task | Who leads | Who supports |
|---|---|---|
| `params.yaml` updates | Member 2 | All members review |
| `Makefile` targets | Member 3 | All members test locally |
| `tests/` unit tests | All members write tests for their own code | |
| `README.md` | All members | Keep in sync with actual code |
| Code reviews | Cross-review before merging to `main` | |

---

## Communication Contracts

These are the exact function signatures each member depends on from the others:

**Member 1 → Member 2**
```python
# Member 2 calls this — Member 1 must make it work
from src.data.dataset import build_study_dataloaders
train_loader, val_loader, test_loader = build_study_dataloaders(config)
# Each batch: images (B, 3, 384, 384) float32, labels (B, 6) float32
```

**Member 2 → Member 3**
```python
# Member 3 calls this — Member 2 must make it work
from src.models.architectures.densenet import build_densenet
model = build_densenet('densenet121', num_classes=6, pretrained=False)
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

logits = model(images)          # (B, 6) — raw logits
features = model.get_features(images)  # (B, 1024) — for Stage 2 feature extraction
```
