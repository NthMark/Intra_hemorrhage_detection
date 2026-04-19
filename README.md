# Intracranial Hemorrhage Detection & Classification

Automated detection and multi-label classification of intracranial hemorrhage (ICH) subtypes from CT scans using deep learning — implemented following the **1st-place RSNA 2019 Intracranial Hemorrhage Detection solution** (Wang et al., 2021), structured as a production-grade ML-DevOps project.

**Hemorrhage subtypes detected:**
`epidural` · `intraparenchymal` · `intraventricular` · `subarachnoid` · `subdural`

---

## Method Overview

This project replicates the 2-stage pipeline from the winning RSNA 2019 solution:

```
Stage 1 — 2D CNN Classifier
  DICOM slices  →  3-channel input (adjacent slices s-1, s, s+1)
               →  DenseNet121 backbone
               →  BCEWithLogitsLoss + WarmRestart scheduler
               →  Per-slice logits [N, 6]

Stage 2 — Sequence Model
  Per-slice logits (ordered within each CT study)
               →  FC → BiGRU (Sequence Model 1)
               →  Conv1D → BiGRU with skip connection (Sequence Model 2)
               →  Elementwise sum → refined per-slice predictions
```

The 3-channel input stacks three adjacent CT slices (instead of three CT window types) to give the backbone implicit depth context across the scan volume.

---

## Project Structure

```
Intra_hemorrhage_detection/
├── .github/workflows/      # CI (lint + test + Docker build) & CD (push image)
├── api/                    # FastAPI REST inference service
│   ├── main.py             # Application entrypoint + lifespan model loading
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/predict.py   # /api/v1/predict endpoint
├── configs/
│   ├── model/              # Per-architecture YAML configs
│   └── training/           # Training experiment configs (default, fast_dev)
├── data/                   # Runtime data (DVC-tracked, git-ignored)
│   ├── raw/                # Extracted DICOM files
│   ├── processed/          # train/val/test CSVs + extracted features (Stage 2)
│   ├── interim/            # Intermediate artefacts
│   └── external/           # External reference data
├── dataset/                # Source CQ500 archives (.zip)
├── docker/
│   ├── Dockerfile          # Multi-stage: trainer + api targets
│   └── docker-compose.yml  # API + MLflow tracking server
├── models/
│   ├── checkpoints/        # Training checkpoints (DVC-tracked)
│   └── production/         # Promoted production weights
├── notebooks/              # Exploratory notebooks (EDA, preprocessing, modelling)
├── paper/                  # Reference literature
├── reference_code/         # 1st-place RSNA 2019 original source code
├── reports/
│   ├── figures/            # Generated plots (loss curves, GradCAM)
│   └── metrics/            # JSON test metrics (DVC metrics)
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Extract archives → build CSVs
│   ├── train.py            # Stage-1 training run
│   ├── extract_features.py # Stage-2 prep: run CNN → save per-slice logits
│   ├── train_sequence.py   # Stage-2: train BiGRU sequence model
│   ├── evaluate.py         # Test-set evaluation → metrics JSON
│   ├── predict.py          # Single / batch inference
│   ├── gradcam.py          # Grad-CAM visualisation
│   └── plot_curves.py      # Plot train/val loss curves + overfit diagnosis
├── src/                    # Core library
│   ├── data/
│   │   ├── dataset.py      # ICHStudyDataset (study-based, paper method) + slice-based
│   │   ├── augmentation.py # Paper augmentation + original albumentations pipeline
│   │   └── preprocessing.py# DICOM loading, HU windowing, adjacent-slice stacking
│   ├── features/           # Metadata CSV builder & stratified splitting
│   ├── models/
│   │   ├── architectures/
│   │   │   ├── densenet.py     # DenseNet121ICH, DenseNet169ICH (paper backbones)
│   │   │   ├── efficientnet.py # EfficientNet + AdaptiveConcatPool2d
│   │   │   └── resnet.py       # ResNet backbone
│   │   ├── sequence_model.py   # BiGRU Stage-2 sequence model
│   │   ├── schedulers.py       # WarmRestart cosine-annealing (paper scheduler)
│   │   ├── train.py            # Training loop + MLflow + early stopping
│   │   ├── evaluate.py         # AUC / sensitivity / specificity metrics
│   │   └── predict.py          # Inference helpers
│   ├── monitoring/         # PSI-based prediction drift detection
│   └── visualization/      # CT window plots, ROC curves, GradCAM
├── tests/                  # pytest unit tests
├── dvc.yaml                # DVC pipeline (prepare → train → evaluate)
├── MLproject               # MLflow Projects definition
├── Makefile                # Developer task runner
├── params.yaml             # Single source of truth for all hyperparameters
├── requirements.txt
├── requirements-dev.txt
├── setup.py / setup.cfg    # Package installation
└── pyproject.toml          # Ruff + mypy configuration
```

---

## Quickstart

### 1. Install

**Using conda (recommended):**

```bash
# Create the environment (first time only)
make conda-create

# Activate before every session
conda activate ich-detection

# After pulling changes that update environment.yml
make conda-update
```

**Using pip only (inside an already-active environment):**

```bash
make install        # runtime dependencies
make install-dev    # runtime + dev tools (lint, test, notebooks)
```

### 2. Prepare data

Place the CQ500 `.zip` archives inside `dataset/`, then:

```bash
make prepare
```

This extracts DICOMs to `data/raw/` and writes stratified `train.csv`, `val.csv`, `test.csv` to `data/processed/`.

### 3. Stage 1 — Train 2D CNN

```bash
make train               # full run — 80 epochs, DenseNet121, paper method
make train-fast          # 5-epoch smoke test
```

All hyperparameters live in `params.yaml`. Override via CLI:

```bash
python scripts/train.py --model densenet169 --epochs 80 --lr 5e-4
```

### 4. Stage 2 — Train Sequence Model (BiGRU)

Run after Stage 1 is complete:

```bash
# Step 1: extract per-slice logits from the trained CNN
make extract-features CKPT=models/checkpoints/best_model.pt

# Step 2: train the BiGRU sequence model on those features
make train-sequence
```

### 5. Evaluate

```bash
make evaluate
# → reports/metrics/test_metrics.json
```

### 6. Visualise training curves

```bash
make curves
# → prints overfit/underfit diagnosis to terminal
# → reports/figures/loss_curves.png
```

### 7. Grad-CAM visualisation

```bash
make gradcam INPUT=data/raw/CQ500CT0.../CT000010.dcm CLASS=2
# → reports/figures/gradcam.png
```

### 8. Run the DVC pipeline end-to-end (optional)

```bash
make pipeline   # equivalent to: dvc repro
```

> **You do NOT need to run this after completing steps 1–7.** `make pipeline` is an *alternative* to running those steps manually — it is intended for reproducing the full run from a clean state (e.g. on a new machine, in CI, or when sharing the project with a collaborator). DVC will automatically skip any stage whose inputs and outputs are already up-to-date, so if you have already run `make prepare`, `make train`, and `make evaluate`, this command will do nothing new.
>
> **When it is useful:**
> - Cloning the repo on a new machine and wanting to reproduce results in one command
> - Running the pipeline in CI/CD to verify reproducibility
> - After changing `params.yaml` and wanting DVC to re-run only the affected downstream stages automatically

### 9. Serve the API

> **Recommended for project submission.** The Docker option lets a reviewer run and test the model with a single command — no Python environment, conda, or training required. Just a trained checkpoint and Docker.

```bash
make serve                    # uvicorn dev server (hot-reload, requires conda env active)
make serve-docker             # Docker Compose — API + MLflow UI, no local env needed
```

Once running, the teacher can:
- Open **http://localhost:8000/docs** — interactive API docs (Swagger UI), upload a DICOM and get predictions directly in the browser
- Open **http://localhost:5000** — MLflow UI to browse training experiments, metrics, and saved checkpoints

> Make sure `models/checkpoints/best_model.pt` exists (produced by step 3) before starting the server. The Docker image bundles all dependencies — only the checkpoint and DICOM data need to be present on the host machine.

---

## Key Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Stage-1 backbone | DenseNet121 / DenseNet169 | Paper's primary architectures (Wang et al., 2021) |
| Stage-2 model | BiGRU + Conv1D with skip connections | Captures inter-slice context within each CT study |
| 3-channel input | Brain / subdural / bone CT windows | Paper strategy (Wang et al. / Sage & Badura 2020): three window presets stacked as RGB channels |
| Training sampling | Study-based random slice sampling | Paper's `RSNA_Dataset_train_by_study_context` |
| Loss | BCEWithLogitsLoss (uniform weight) | Paper's loss; handles multi-label output |
| Optimizer | Adam lr=5×10⁻⁴, wd=2×10⁻⁵ | Paper's exact hyperparameters |
| Scheduler | WarmRestart (T_max=5, η_min=1×10⁻⁵) | Paper's SGDR with warm restarts from epoch 11 |
| Augmentation | HFlip + ShiftScaleRotate + random erasing + random crop | Paper's `aug_image` pipeline |
| Experiment tracking | MLflow (SQLite backend) | Lightweight, self-hostable |
| Pipeline reproducibility | DVC | Git-native data & model versioning |
| Serving | FastAPI + uvicorn | Async, typed, OpenAPI schema auto-generated |
| Monitoring | PSI-based score drift | Stateless, no external dependencies |

---

## Training Schedule (Paper-faithful)

| Epoch range | Behaviour |
|---|---|
| 1 – 10 | Constant LR = 5×10⁻⁴ (warmup plateau) |
| 11 – 80 | `scheduler.step()` + `warm_restart(T_mult=2)` each epoch |

The WarmRestart scheduler (SGDR) exponentially increases the cosine cycle period after each restart, allowing the model to escape local minima progressively.

---

## Traning Batches Explaineid

### What is one sample?

Each sample fed to the CNN is **one CT slice** — a single DICOM file converted to a `(3, 512, 512)` tensor by applying three CT window settings (brain / subdural / bone) as the three RGB channels.

### How many batches per epoch?

The number of batches per epoch is determined by three layers of logic:

**1. Study-based dataset length**

`ICHStudyDataset` groups all slices in `train.csv` by CT study (patient scan). Its length is:

```
dataset length = num_unique_studies × 4
```

The `×4` multiplier matches the reference implementation — each study is seen approximately 4 times per epoch, cycling through different randomly-sampled slices each time.

**2. Random slice selection per study**

Each `__getitem__` call picks a **weighted-random** slice from that study (slices near the centre of the scan are sampled more often than top/bottom slices, which contain more skull than brain). The model sees a different slice from the same study on each of the 4 visits.

**3. DataLoader with `drop_last=True`**

```
batches_per_epoch = floor(dataset_length / batch_size)
                  = floor(num_studies × 4 / 16)
                  = num_studies // 4
```

**Example with CQ500 (~491 studies, 70% training split → ~488 in train.csv):**

| Value | Calculation |
|---|---|
| Training studies | ~488 |
| Dataset length | 488 × 4 = 1,952 |
| Batch size | 16 |
| **Batches per epoch** | **floor(1952 / 16) = 122** |

> The split in `prepare_data.py` works at the **slice level** (stratified by hemorrhage presence), not the study level. This means nearly every study has at least some slices in `train.csv`, so the number of unique training studies is close to 70% × 491 ≈ 344 after a strict study-level split, but in practice approaches 488 because slices from the same study are spread across all three splits.

---

## Further Information

### Glossary of Medical Imaging Terms

| Term | Explanation |
|---|---|
| **CT scan** (Computed Tomography) | A medical imaging technique that uses X-rays taken from many angles and combines them into cross-sectional images of the body. For the brain, it produces a stack of horizontal "slice" images from the top of the head down to the neck. |
| **CT slice** | A single 2D cross-sectional image from a CT scan — like one layer in a loaf of bread. A full brain CT scan typically contains 20–400 slices depending on the slice thickness used. |
| **CT study** | The complete set of slices from a single patient's CT scan session. One study = one full brain scan = many slices stacked together. |
| **DICOM** | The standard file format used by medical scanners to store CT images. Each `.dcm` file typically contains one CT slice plus metadata (patient ID, scan settings, etc.). |
| **HU (Hounsfield Unit)** | The measurement scale used in CT imaging. Air = −1000 HU, water = 0 HU, soft tissue = 20–80 HU, bone = 400–1000 HU. Different hemorrhage types appear at different HU ranges, so "windowing" (clipping the HU range) is used to highlight specific tissues. |
| **Intracranial hemorrhage (ICH)** | Bleeding inside the skull. It is a medical emergency because blood accumulating in the brain increases pressure and can cause permanent damage or death. |
| **Epidural** | Bleeding between the skull bone and the outer brain membrane (dura mater). Often caused by head trauma. |
| **Subdural** | Bleeding between the outer and middle brain membranes. Can be acute (sudden) or chronic (slow). |
| **Subarachnoid** | Bleeding in the space between the brain surface and the inner membrane. Often caused by a ruptured aneurysm. |
| **Intraparenchymal** | Bleeding directly inside the brain tissue itself. |
| **Intraventricular** | Bleeding inside the fluid-filled cavities (ventricles) of the brain. |
| **Multi-label classification** | A task where each input can belong to multiple classes simultaneously. Here, a CT slice can show more than one hemorrhage subtype at the same time. |
| **AUC (Area Under the ROC Curve)** | A metric between 0 and 1 measuring how well the model distinguishes between positive and negative cases. AUC = 1.0 is perfect; AUC = 0.5 is random guessing. |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — a visualisation technique that highlights which regions of a CT slice the model focused on when making its prediction. |

---

### Why train Stage 1 first, then Stage 2?

It may seem unusual to train two separate models sequentially. Here is the reasoning behind the design:

**Stage 1 — CNN trains slice-by-slice**

The DenseNet looks at one CT slice at a time (three adjacent slices stacked as channels: s-1, s, s+1). It learns *"does this slice look like it contains hemorrhage?"* and outputs 6 confidence scores per slice. It has no awareness of where that slice sits within the full scan or what the surrounding slices look like.

**Stage 2 — BiGRU trains on the full sequence**

The per-slice scores from Stage 1 are stacked in anatomical order (top-to-bottom of the head) for every CT study. The BiGRU reads this sequence forward and backward, learning *"given how the scores evolve across the whole scan, is this prediction real hemorrhage or noise?"* It refines the final predictions using context from the entire study.

**Why not train them end-to-end together?**

| Reason | Explanation |
|---|---|
| Memory | Loading a full CT study (100–400 slices) through a CNN simultaneously would exceed GPU memory |
| Speed | Stage 1 trains on individual slices in parallel; Stage 2 trains on compact logit sequences (tiny data, finishes in minutes) |
| Results | The paper showed this 2-stage approach outperforms a slice-only CNN |

Think of it like a radiologist's workflow: first scan each slice individually for suspicious regions (Stage 1), then review the full scan together to make a confident final diagnosis (Stage 2).

---

## Running Tests

```bash
make test          # unit tests only
make test-cov      # with HTML coverage report
```

---

## Code Quality

```bash
make lint          # ruff
make format        # ruff format (auto-fix)
make type-check    # mypy
```

---

## Dataset

This project uses the **CQ500** dataset (Centre for Advanced Research in Imaging, Neurosciences & Genomics, New Delhi).

### Source files (`dataset/`)

| File | Description |
|---|---|
| `CQ500-CT-*.zip` | CT studies (491 total in full dataset) |
| `reads.csv` | Radiologist readings: 3 readers (R1, R2, R3) × 9 findings per study |
| `prediction_probabilities.csv` | Pre-computed AI model confidence scores (not used in training) |

### Label generation

Study-level labels are derived from `reads.csv` using **majority vote** (≥ 2 out of 3 readers must agree):

- `no_hemorrhage = 1` if fewer than 2 readers flagged ICH
- Each subtype (`epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`) `= 1` if ≥ 2 readers agree

Labels are broadcast from study level to every DICOM slice within that study.

### Train / val / test split

The split is performed at the **study level** — all slices from a given CT study go into exactly one partition. This prevents data leakage: the model cannot see slices from a patient during training and then be evaluated on a different slice from the same patient.

```
split_dataframe() in src/features/build_features.py:
  1. Assign each study a binary label (any hemorrhage present?)
  2. Stratified train_test_split on study IDs  (70 / 15 / 15)
  3. Map study IDs back to slice rows → train.csv / val.csv / test.csv
```

> A naive slice-level split would let slices from the same study appear in all three partitions simultaneously — inflating validation/test AUC by up to several percent because the model has already seen the patient's anatomy during training.

### Full dataset statistics (491 studies, ~171K slices)

| Split | Studies | Approx. slices |
|---|---|---|
| train (70%) | ~344 | ~120,000 |
| val (15%) | ~74 | ~26,000 |
| test (15%) | ~73 | ~25,000 |

> Each study contributes all its slices to one split only. The slice counts are approximate because studies vary in thickness (20–400 slices each).

> Download the full CQ500 dataset from the [CQ500 project page](http://headctstudy.qure.ai/) to train a production-grade model.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- **Wang et al. (2021)** — *A Deep Learning Algorithm for Automatic Detection and Classification of Acute Intracranial Hemorrhages in Head CT Scans*  
  NeuroImage: Clinical. <https://doi.org/10.1016/j.nicl.2021.102785>  
  Repository: <https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection>

  Techniques from this paper implemented in this codebase:
  - **Study-based training** (`ICHStudyDataset`) — randomly samples a slice from each CT study per epoch; centre-weighted to favour informative sections.
  - **Adjacent-slice 3-channel input** — stacks slices (s-1, s, s+1) as the 3 input channels, giving the 2D CNN implicit depth context without 3D convolutions.
  - **DenseNet121/169 backbone** (`src/models/architectures/densenet.py`) — with adaptive average pooling head, matching `DenseNet121_change_avg` and `DenseNet169_change_avg`.
  - **WarmRestart scheduler** (`src/models/schedulers.py`) — SGDR (Loshchilov & Hutter, 2017) with exponentially growing cycle period; plateau on epochs 1–10 then restarts from epoch 11.
  - **BiGRU sequence model** (`src/models/sequence_model.py`) — Stage-2 model that consumes per-slice CNN logits ordered within each study and refines them using bidirectional GRU + 1D-CNN with skip connections.
  - **Numerically stable FocalLoss** (`src/models/train.py`) — log-sum-exp formulation from the reference code (available as fallback; paper's primary loss is BCE).

- **CQ500 Dataset** — Centre for Advanced Research in Imaging, Neurosciences and Genomics (CARING):  
  <http://headctstudy.qure.ai/dataset>

- **Loshchilov & Hutter (2017)** — *SGDR: Stochastic Gradient Descent with Warm Restarts*  
  <https://arxiv.org/abs/1608.03983>


```
Intra_hemorrhage_detection/
├── .github/workflows/      # CI (lint + test + Docker build) & CD (push image)
├── api/                    # FastAPI REST inference service
│   ├── main.py             # Application entrypoint + lifespan model loading
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/predict.py   # /api/v1/predict endpoint
├── configs/
│   ├── model/              # Per-architecture YAML configs
│   └── training/           # Training experiment configs (default, fast_dev)
├── data/                   # Runtime data (DVC-tracked, git-ignored)
│   ├── raw/                # Extracted DICOM files
│   ├── processed/          # train/val/test CSVs
│   ├── interim/            # Intermediate artefacts
│   └── external/           # External reference data
├── dataset/                # Source CQ500 archives (.zip)
├── docker/
│   ├── Dockerfile          # Multi-stage: trainer + api targets
│   └── docker-compose.yml  # API + MLflow tracking server
├── models/
│   ├── checkpoints/        # Training checkpoints (DVC-tracked)
│   └── production/         # Promoted production weights
├── notebooks/              # Exploratory notebooks (EDA, preprocessing, modelling)
├── paper/                  # Reference literature
├── reports/
│   ├── figures/            # Generated plots (ROC curves, GradCAM)
│   └── metrics/            # JSON test metrics (DVC metrics)
├── scripts/                # CLI entry points
│   ├── prepare_data.py     # Extract archives → build CSVs
│   ├── train.py            # Training run
│   ├── evaluate.py         # Test-set evaluation → metrics JSON
│   └── predict.py          # Single / batch inference
├── src/                    # Core library
│   ├── data/               # DICOM loading, HU windowing, Dataset, augmentation
│   ├── features/           # Metadata CSV builder & stratified splitting
│   ├── models/
│   │   ├── architectures/  # EfficientNet & ResNet backbone + head
│   │   ├── train.py        # Training loop + MLflow logging + early stopping
│   │   ├── evaluate.py     # AUC / sensitivity / specificity metrics
│   │   └── predict.py      # Inference helpers (single DICOM + batch)
│   ├── monitoring/         # PSI-based prediction drift detection
│   └── visualization/      # CT window plots, ROC curves, GradCAM
├── tests/                  # pytest unit tests
├── dvc.yaml                # DVC pipeline (prepare → train → evaluate)
├── MLproject               # MLflow Projects definition
├── Makefile                # Developer task runner
├── params.yaml             # Single source of truth for all hyperparameters
├── requirements.txt
├── requirements-dev.txt
├── setup.py / setup.cfg    # Package installation
└── pyproject.toml          # Ruff + mypy configuration
```

---

## Quickstart

### 1. Install

**Using conda (recommended):**

```bash
# Create the environment (first time only)
make conda-create

# Activate before every session
conda activate ich-detection

# After pulling changes that update environment.yml
make conda-update
```

**Using pip only (inside an already-active environment):**

```bash
make install        # runtime dependencies
make install-dev    # runtime + dev tools (lint, test, notebooks)
```

### 2. Prepare data

Place the CQ500 `.zip` archives inside `dataset/`, then:

```bash
make prepare
```

This extracts DICOMs to `data/raw/` and writes stratified `train.csv`, `val.csv`, `test.csv` to `data/processed/`.

### 3. Train

```bash
make train               # full run (params.yaml)
make train-fast          # 5-epoch smoke test
```

All hyperparameters live in `params.yaml`. Override via CLI:

```bash
python scripts/train.py --model resnet50d --epochs 30 --lr 3e-4
```

### 4. Evaluate

```bash
make evaluate
# → reports/metrics/test_metrics.json
```

### 5. Run the DVC pipeline end-to-end

```bash
make pipeline   # equivalent to: dvc repro
```

### 6. Serve the API

```bash
make serve                    # uvicorn dev server (hot-reload)
make serve-docker             # Docker Compose (API + MLflow UI)
```

API docs: http://localhost:8000/docs  
MLflow UI: http://localhost:5000

---

## Key Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Backbone | EfficientNet-B4 / ResNet-50d (timm) | Strong ImageNet priors; swappable via config |
| Input representation | 3-channel CT windows (brain, subdural, bone) | Captures complementary tissue contrasts |
| Loss | Binary Focal Loss | Handles class imbalance across subtypes |
| Experiment tracking | MLflow | Lightweight, self-hostable |
| Pipeline reproducibility | DVC | Git-native data & model versioning |
| Serving | FastAPI + uvicorn | Async, typed, OpenAPI schema auto-generated |
| Monitoring | PSI-based score drift | Stateless, no external dependencies |

---

## Running Tests

```bash
make test          # unit tests only
make test-cov      # with HTML coverage report
```

---

## Code Quality

```bash
make lint          # ruff
make format        # ruff format (auto-fix)
make type-check    # mypy
```

---

## Dataset

This project uses the **CQ500** dataset (Centre for Advanced Research in Imaging, Neurosciences & Genomics, New Delhi).

### Source files (`dataset/`)

| File | Description |
|---|---|
| `CQ500-CT-1.zip` | CT study — intraparenchymal + subarachnoid hemorrhage |
| `CQ500-CT-24.zip` | CT study — no hemorrhage (majority vote) |
| `CQ500-CT-25.zip` | CT study — no hemorrhage |
| `reads.csv` | Radiologist readings: 3 readers (R1, R2, R3) × 9 findings per study |
| `prediction_probabilities.csv` | Pre-computed AI model confidence scores (not used in training) |

### Label generation

Study-level labels are derived from `reads.csv` using **majority vote** (≥ 2 out of 3 readers must agree):

- `no_hemorrhage = 1` if fewer than 2 readers flagged ICH
- Each subtype (`epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`) `= 1` if ≥ 2 readers agree

Labels are broadcast from study level to every DICOM slice within that study.

### Current dataset statistics (3 studies, 392 slices)

| Label | Slice count |
|---|---|
| `no_hemorrhage` | 356 |
| `intraparenchymal` | 36 |
| `subarachnoid` | 36 |
| `epidural` | 0 |
| `intraventricular` | 0 |
| `subdural` | 0 |

> **Note:** Only 3 of the 491 CQ500 studies are included. Download the full dataset from the [CQ500 project page](http://headctstudy.qure.ai/) to train a production-grade model. See `paper/` for methodology references.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- **1st-place RSNA 2019 ICH Detection solution** by SeuTao et al.  
  Repository: <https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection>  
  Techniques adopted in this codebase:
  - `AdaptiveConcatPool2d` — concatenates adaptive avg-pool and max-pool before the classification head, yielding richer global features than avg-pool alone.
  - Adjacent-slice 3-channel input — stacks slices (s-1, s, s+1) as RGB channels to give the 2-D backbone implicit depth context (`adjacent_slices_to_3channel` in `src/data/preprocessing.py`).
  - Numerically stable `FocalLoss` — log-sum-exp formulation avoids sigmoid overflow on large logit magnitudes (`src/models/train.py`).

- **CQ500 Dataset** — Centre for Advanced Research in Imaging, Neurosciences and Genomics (CARING):  
  <http://headctstudy.qure.ai/dataset>
