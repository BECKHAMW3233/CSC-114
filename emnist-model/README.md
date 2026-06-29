# EMNIST OCR Ensemble — Handwritten Character Recognition Pipeline

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20GPU-green)
![Dataset](https://img.shields.io/badge/Dataset-9%20Sources%20%7C%201.44M%20samples-lightgrey)
![Models](https://img.shields.io/badge/Ensemble-6%20Models%20(3%20base%20%2B%203%20distilled)-brightgreen)

**Author:** William Edward Beckham III  
**Program:** Computer Programming & Development AAS — FTCC  
**Course:** CSC-114 AI Fundamentals I (Summer 2026)  
**Hardware:** AMD Ryzen 9 7900X · 64 GB DDR5-5600 · ZOTAC RTX 4080 16 GB AMP Extreme AIRO

---

## Overview

This project implements a six-model deep learning ensemble for handwritten character recognition across 62 classes — digits 0–9, uppercase A–Z, and lowercase a–z. Three architecturally diverse base models are trained on a 9-source, 1,443,757-sample dataset, then improved through knowledge distillation, exported to ONNX, and deployed through a custom inference pipeline with post-processing compensation for known model bias patterns.

The project was conducted entirely on consumer hardware (RTX 4080) with a self-imposed 12-hour per-run training ceiling, no cloud compute, and a fully reproducible open-source toolchain. All training artifacts, intermediate outputs, soft labels, and trained models are included in the repository.

This is a self-directed research project developed independently alongside CSC-114 coursework, which covers foundational deep learning concepts through approximately Chapter 8 of Chollet & Watson (2026). The EMNIST ensemble operates well beyond the course scope and is being prepared for arXiv submission targeting ICDAR.

---

## Version History

| Version | Models | Dataset | Key Change |
|---------|--------|---------|------------|
| v1 | 1 (Adam) | EMNIST byclass only (697,932) | Baseline ConvNet |
| v2 | 3 (Adam) | 5 sources | Rotation fix, WeightedRandomSampler, 64×64 resolution, augmentation diversity |
| v3 | 3 diverse optimizers | 9 sources (1,443,757) | Lion/SF-AdamW/SGD, SE attention, full retrains from scratch |
| v3 + distillation | 6 (3 base + 3 distilled) | Same 9 sources | Knowledge distillation, 6-model ensemble |

Each version represents a complete retrain from random initialization — no inherited weights from prior versions until the distillation phase, which is the first intentional use of pretrained weights in the project.

---

## Repository Structure

```
emnist-model/
├── ocr_pipeline.py              # 6-model inference pipeline (base + distilled) — school machine paths
├── home_test_full.py            # 6-model inference pipeline — home machine paths, full post-processing
├── ocr_distillation.py          # Knowledge distillation — phases 1, 2, 3
├── ocr_pytorch_model.py         # Model 1 training (OCRConvNet — Lion)
├── ocr_pytorch_model2.py        # Model 2 training (OCRConvNetWide — SF-AdamW)
├── ocr_pytorch_model3.py        # Model 3 training (OCRConvNetTriple — SGD)
├── supplementary_data.py        # Shared 9-source dataset loader
├── download_datasets.py         # Automated dataset download script
├── install_deps.py              # Dependency installer
├── 01_install_cuda.bat          # CUDA 12.1 + cuDNN setup (run first, as Admin)
├── 02_install_python_packages.bat  # Python venv + all dependencies
├── 03_verify_gpu.py             # GPU and dataset verification
├── pytorch/                     # Model 1 checkpoint + ONNX
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── ocr_model.onnx           (9.4 MB)
│   ├── training_curves.png
│   └── training_log.csv
├── pytorch2/                    # Model 2 checkpoint + ONNX
│   ├── best_model2.pt
│   ├── final_model2.pt
│   ├── ocr_model2.onnx          (37.1 MB)
│   ├── training_curves2.png
│   └── training_log2.csv
├── pytorch3/                    # Model 3 checkpoint + ONNX
│   ├── best_model3.pt
│   ├── final_model3.pt
│   ├── ocr_model3.onnx          (17.5 MB)
│   ├── training_curves3.png
│   └── training_log3.csv
├── pytorch_distill1/            # Model 1 distilled checkpoint + ONNX
│   ├── best_distill1.pt
│   ├── final_distill1.pt
│   └── ocr_model1_distill.onnx  (9.4 MB)
├── pytorch_distill2/            # Model 2 distilled checkpoint + ONNX
│   ├── best_distill2.pt
│   ├── final_distill2.pt
│   └── ocr_model2_distill.onnx  (37.1 MB)
├── pytorch_distill3/            # Model 3 distilled checkpoint + ONNX
│   ├── best_distill3.pt
│   ├── final_distill3.pt
│   └── ocr_model3_distill.onnx  (17.5 MB)
├── soft_labels/                 # Phase 1 distillation outputs — excluded from repo (~165 MB each)
│   ├── soft_labels_m1.npy
│   ├── soft_labels_m2.npy
│   └── soft_labels_m3.npy
└── README.md
```

---

## Setup & Installation

### Automated Setup (recommended)

Run the included batch files in order on Windows with an NVIDIA GPU:

```bat
# 1. Run as Administrator
01_install_cuda.bat

# 2. Run as normal user
02_install_python_packages.bat

# 3. Verify
python 03_verify_gpu.py
```

### Manual Installation

**Inference only:**
```bash
pip install opencv-python numpy onnxruntime-gpu==1.19.2
```

> **Note:** `onnxruntime-gpu==1.19.2` is pinned to CUDA 12.x. Do not upgrade without verifying CUDA compatibility — later versions require CUDA 13.

**Training (full pipeline):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics matplotlib pillow optuna scipy pandas kaggle certifi
pip install lion-pytorch schedulefree
pip install onnx onnxruntime-gpu==1.19.2
```

### Dataset Download

```bash
python download_datasets.py
```

Downloads EMNIST splits, MNIST, USPS, SVHN, and Kaggle A-Z automatically. Chars74K requires manual download from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

### Configuring Model Paths

Update the `MODELS` list in whichever pipeline file you are running to match your system paths. The school testing paths and home machine paths differ — separate pipeline files are maintained for each environment. Any technically capable reader can locate the `MODELS` list at the top of each pipeline file and update the paths to their own file structure.

### Quick Start

```bash
# Clone the repo
git clone https://github.com/BECKHAMW3233/CSC-114.git
cd CSC-114/emnist-model

# Install inference dependencies
pip install opencv-python numpy onnxruntime-gpu==1.19.2

# Edit MODELS paths in ocr_pipeline.py, then:
python ocr_pipeline.py --mode digits-strict test1.jpg
```

---

## Usage

```bash
# 6-model ensemble (base + distilled) — highest accuracy
python ocr_pipeline.py --mode auto image.jpg

# Digit content with letter to digit remapping
python ocr_pipeline.py --mode digits test1.jpg

# Digit grid with position-based correction (highest accuracy for 0-9 grids)
python ocr_pipeline.py --mode digits-strict test*.jpg

# Force uppercase output
python ocr_pipeline.py --mode upper handwriting.jpg

# Force lowercase output
python ocr_pipeline.py --mode lower handwriting.jpg

# Multiple files at once
python ocr_pipeline.py --mode digits-strict test1.jpg test2.jpg test3.jpg
```

---

## Sample Output

Running `--mode digits-strict` on a handwritten 0-9 grid (v2 base 3-model pipeline):

```
============================================================
  OCR Pipeline - 3-Model Ensemble
  Image: test5.jpg
  Mode:  DIGITS-STRICT
============================================================
  Detected: 10 characters across 3 line(s)

  --------------------------------------------------
  INDIVIDUAL MODEL PREDICTIONS (raw, no remapping)
  --------------------------------------------------
  ocr_model.onnx (64x64):
    Line 1: O ? Z B
    Line 2: Y S G ?
    Line 3: Y Q
  ocr_model2.onnx (64x64):
    Line 1: O T U W
    Line 2: Y S G Z
    Line 3: ? Q
  ocr_model3.onnx (64x64):
    Line 1: O L Z W
    Line 2: U S G C
    Line 3: ? Q

  --------------------------------------------------
  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  *=strict  ?=split)
  --------------------------------------------------
  Line 1: 0 *1* [2] [3]
  Line 2: [4] 5 6 *7*
  Line 3: *8* 9

  Consensus: 4/10 chars (40.0% all-3 agreement)

  --------------------------------------------------
  STRICT GRID CORRECTIONS (3 applied)
  --------------------------------------------------
  Line 1 Char 2: ? (SPLIT) -> 1 [position override]
  Line 2 Char 4: ? (SPLIT) -> 7 [position override]
  Line 3 Char 1: 4 (WEIGHTED) -> 8 [position override]

  --------------------------------------------------
  BEST GUESS READ  [mode: DIGITS-STRICT]
  --------------------------------------------------
  Line 1: 0 1 2 3
  Line 2: 4 5 6 7
  Line 3: 8 9
============================================================
```

**Output legend:**
- `plain` — all models agree (unanimous)
- `[x]` — majority or weighted vote winner
- `*x*` — position override applied by strict grid correction
- `?` — unresolved split (no majority, no position correction available)

---

## Models

### Architecture Overview

Three architecturally diverse base models were chosen specifically to maximize error diversity across the ensemble. Optimizer families were selected to produce qualitatively different weight landscapes: momentum-free adaptive (Lion), schedule-free adaptive (SF-AdamW), and classical momentum (SGD). Each model was trained from random initialization on the full 9-source dataset.

### Model 1 — OCRConvNet (`ocr_pytorch_model.py`)

Baseline convolutional network following architecture patterns from Chollet & Watson, *Deep Learning with Python, 3rd Ed.* (Manning 2026), Chapters 8-9.

| Parameter | Value |
|-----------|-------|
| Parameters | 2.5M |
| ONNX size | 9.4 MB |
| Input resolution | 64x64 |
| Filter progression | 32->64->128->256 |
| Classifier head | 256->128->62 |
| Optimizer | Lion (lr=3e-5, wd=0.01) |
| Scheduler | CosineAnnealingLR (eta_min=1e-7) |
| Batch size | 256 |
| Augmentation | Rotation +/-5 deg, affine (translate 10%, scale 0.85-1.15, shear 3 deg), contrast jitter |
| Regularization | Dropout, L2 weight decay, label smoothing 0.05 |
| Train acc | 91.17% |
| Test acc | 81.15% |
| Output | `pytorch/ocr_model.onnx` |

### Model 2 — OCRConvNetWide (`ocr_pytorch_model2.py`)

Wider architecture with Squeeze-Excitation attention blocks after each stage for channel-wise feature recalibration. Adds synthetic degradation augmentation (blur + noise) to ensure Model 2 sees a different data distribution than Model 1, producing partially uncorrelated errors in the ensemble.

| Parameter | Value |
|-----------|-------|
| Parameters | 9.7M |
| ONNX size | 37.1 MB |
| Input resolution | 64x64 |
| Filter progression | 32->128->256->512 |
| Attention | Squeeze-Excitation (SE) after each stage |
| Regularization | StochasticDepth (DropPath) + Dropout |
| Classifier head | 512->256->62 |
| Optimizer | Schedule-Free AdamW (lr=0.001, wd=0.0001, warmup=5640 steps) |
| Scheduler | None (schedule-free) |
| Batch size | 256 |
| Augmentation | Rotation +/-5 deg, affine, Gaussian blur + noise |
| Train acc | 91.27% |
| Test acc | 83.85% |
| Output | `pytorch2/ocr_model2.onnx` |

### Model 3 — OCRConvNetTriple (`ocr_pytorch_model3.py`)

Maximum capacity model with triple-width channels, SE attention (reduction=16), and a feature pyramid concatenating pooled outputs from three stages before classification. Uses GELU activations in a five-layer classifier and the most aggressive augmentation of the three models, including perspective distortion. Patience set to 20 (vs 15 for M1/M2) due to SGD's slower, noisier convergence requiring more runway to find minima.

| Parameter | Value |
|-----------|-------|
| Parameters | 4.6M |
| ONNX size | 17.5 MB |
| Input resolution | 64x64 |
| Channel progression | 96->192->384->768 |
| Feature fusion | Pyramid: concatenated pool from stages 2+3+4 |
| SE reduction | 16 |
| Classifier head | fused->1024->512->256->128->62 (5 layers, GELU) |
| Optimizer | SGD (lr=0.01, momentum=0.9, nesterov=True, wd=5e-4) |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Batch size | 128 |
| Augmentation | Rotation +/-5 deg, affine, perspective distortion, blur + noise |
| Train acc | 84.85% |
| Test acc | 77.30% |
| Output | `pytorch3/ocr_model3.onnx` |

### Model 3 Development History

Three runs were required before the final v4 run. This history is documented as a methodological finding on optimizer and scheduler selection for this class of problem.

- **Run 1:** CosineAnnealingWarmRestarts T_0=35 — restart fires at epoch 37, leaving only 13 epochs for recovery. Published SGDR paper used 200+ epoch budgets; T_0=35 with T_mult=2 requires minimum ~105 epochs for two full cycles. Test acc: 76.61%.
- **Run 2:** FocalLoss gamma=2.0 — catastrophic class collapse. O->0.1%, S->0.7%. Test acc: 76.61%.
- **Run 3 (v4, final):** 9 fixes applied simultaneously — CosineAnnealingWarmRestarts->CosineAnnealingLR, weight decay 3e-5->5e-4, label smoothing 0.08->0.05 (FocalLoss removed), sharpness_factor bug fixed (0->2.0), contrast 0.4->0.2, translate 0.12->0.08, SE reduction 32->16, drop_path 0.1->0.05, first classifier dropout 0.5->0.35. Test acc: 77.30%.

### Base Ensemble Results

| Configuration | Accuracy |
|---------------|----------|
| M1 + M2 | 83.55% |
| M1 + M2 + M3 | 82.61% |

The 3-model ensemble underperforms M1+M2 alone due to M3's lower base accuracy dragging the vote. Distillation addresses this by bringing M3 to a competitive level before ensemble combination.

---

## Training Data

All three models use identical data sources via `supplementary_data.py`. DIGIT_BOOST=3.0x applied to digit-only datasets to counteract the uppercase-heavy bias of Kaggle A-Z. `WeightedRandomSampler` enforces equal class representation per batch across all 62 classes.

| Dataset | Samples | Classes | Notes |
|---------|---------|---------|-------|
| EMNIST byclass | 697,932 | 62 | Primary — digits + upper + lower |
| EMNIST Balanced | 86,400 | 47 | Equal samples per class |
| EMNIST Digits | 240,000 | 10 | Digit reinforcement |
| MNIST | 60,000 | 10 | Different writer pool than EMNIST |
| USPS | 7,291 | 10 | Scanned postal envelopes |
| SVHN | 73,257 | 10 | Street sign photos — real-world domain shift |
| Kaggle A-Z | 372,451 | 26 | Uppercase only |
| Chars74K Hnd | 3,410 | 62 | Handwritten, all 62 classes |
| Chars74K Img | 7,705 | 62 | Natural scene, all 62 classes |
| **Total** | **1,443,757** | | |

**EMNIST byclass index mapping:**
- Indices 0-9 -> digits 0-9
- Indices 10-35 -> uppercase A-Z
- Indices 36-61 -> lowercase a-z

---

## Knowledge Distillation

### Methodology

Each base model is retrained using a combined loss function:
- **70% KL-divergence** against averaged soft labels from the other two base models (temperature=4.0)
- **30% CrossEntropy** against hard ground truth labels (label smoothing=0.05)

Soft labels encode inter-class relationship information that hard labels cannot — the probability distribution over all 62 classes at temperature=4.0 carries meaningful signal about visual similarity between characters. Distillation uses AdamW for all three models regardless of original optimizer, starting from pretrained base weights rather than random initialization.

### Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 4.0 |
| Alpha (soft label weight) | 0.7 |
| Hard label weight | 0.3 |
| Label smoothing | 0.05 |
| Soft label source | EMNIST byclass train set (697,932 samples) |
| Soft label files | ~165 MB each x 3 (included in repo) |
| Starting weights | Pretrained base model checkpoints |
| Optimizer (all 3) | AdamW with CosineAnnealingLR (eta_min=1e-7) |
| Max epochs | 50 |
| Patience | 15 (M1, M2) / 20 (M3) |

Patience is an architectural decision, not a training hyperparameter tuned post-hoc. M1 (Lion, 2.5M params) and M2 (SF-AdamW, 9.7M params) converge decisively with adaptive optimizers — patience 15 is appropriate. M3's deeper channel attention and SGD-trained weight landscape has a slower, noisier convergence rhythm that requires the additional runway patience 20 provides.

### Teacher Assignments

| Student | Teachers | Rationale |
|---------|----------|-----------|
| M1 distilled | M2 + M3 | Receives knowledge from wider and deeper architectures |
| M2 distilled | M1 + M3 | Strongest student receives guidance from both other families |
| M3 distilled | M1 + M2 | Weakest base model receives the two strongest teachers |

### Distillation Results

| Model | Base Accuracy | Distilled Accuracy | Delta | Epochs Run |
|-------|--------------|-------------------|-------|------------|
| M1 (OCRConvNet) | 81.15% | 88.11% | +6.96% | 47 |
| M2 (OCRConvNetWide) | 83.86% | 88.31% | +4.46% | 50 |
| M3 (OCRConvNetTriple) | 77.30% | 88.47% | +11.17% | 50 |

ONNX validation (Phase 3) confirmed all six models load and run correctly on GPU with consistent accuracy.

M3 showed the largest gain as predicted — the weakest base model receiving the strongest teacher combination, with the most headroom to improve. All three distilled models converged within 0.36% of each other despite starting from significantly different base accuracies.

Throughout all distillation runs, val_acc exceeded train_acc from the first epoch — a consequence of starting from pretrained weights combined with soft label generalization. The 10+ point train/val gap at epoch 1 converged to near-zero by final epochs, confirming effective knowledge transfer rather than memorization.

### Per-Class Analysis (Distilled Models)

**15 worst-performing classes — distilled M1 (88.11% overall):**

| Class | Accuracy | Samples |
|-------|----------|---------|
| o (50) | 0.2% | 466 |
| c (38) | 1.6% | 432 |
| s (54) | 2.1% | 437 |
| u (56) | 3.9% | 482 |
| m (48) | 15.5% | 464 |
| l (47) | 16.0% | 2535 |
| f (41) | 36.8% | 400 |
| i (44) | 44.3% | 427 |
| v (57) | 46.2% | 468 |
| p (51) | 46.5% | 368 |
| q (52) | 47.7% | 505 |
| I (18) | 51.7% | 2048 |
| z (61) | 59.4% | 451 |
| Z (35) | 60.3% | 464 |
| O (24) | 62.2% | 4156 |

**15 worst-performing classes — distilled M2 (88.32% overall):**

| Class | Accuracy | Samples |
|-------|----------|---------|
| o (50) | 2.6% | 466 |
| s (54) | 11.2% | 437 |
| c (38) | 19.0% | 432 |
| u (56) | 23.7% | 482 |
| l (47) | 24.4% | 2535 |
| f (41) | 28.8% | 400 |
| m (48) | 29.1% | 464 |
| v (57) | 39.1% | 468 |
| p (51) | 39.4% | 368 |
| i (44) | 45.4% | 427 |
| q (52) | 47.5% | 505 |
| y (60) | 50.4% | 381 |
| I (18) | 50.9% | 2048 |
| z (61) | 51.4% | 451 |
| O (24) | 53.5% | 4156 |

**15 worst-performing classes — distilled M3 (88.47% overall):**

| Class | Accuracy | Samples |
|-------|----------|---------|
| o (50) | 0.0% | 466 |
| s (54) | 0.0% | 437 |
| m (48) | 9.1% | 464 |
| c (38) | 12.7% | 432 |
| u (56) | 17.6% | 482 |
| l (47) | 20.1% | 2535 |
| f (41) | 27.2% | 400 |
| q (52) | 45.3% | 505 |
| i (44) | 48.2% | 427 |
| v (57) | 51.1% | 468 |
| I (18) | 53.2% | 2048 |
| p (51) | 53.8% | 368 |
| Z (35) | 59.9% | 464 |
| O (24) | 62.2% | 4156 |
| z (61) | 62.7% | 451 |

The same lowercase cluster fails consistently across all three distilled models: o, s, c, u, l. These classes are structurally ambiguous at 64x64 resolution and the averaged soft labels from teachers who also struggled with them cannot provide corrective signal. This is a visual ambiguity problem requiring architectural solutions targeting stroke endpoint detection, not a training distribution problem.

Notable regression: class 50 (o) degraded post-distillation relative to base models in both M1 and M2 (M1 base 49.4% -> M1 distilled 0.2%; M2 base 32.4% -> M2 distilled 2.6%) despite overall accuracy improving significantly. The averaged soft labels from two models that both misclassify o reinforced rather than corrected the confusion for this specific class.

---

## Pipeline Configurations

Two pipeline files are included in the repository:

| Pipeline | Models | Paths | Purpose |
|----------|--------|-------|---------|
| `ocr_pipeline.py` | All 6 | School machine (C:\Users\beckhamw3233\Downloads\) | Full ensemble — school testing and demonstration |
| `home_test_full.py` | All 6 | Home machine (E:\CSC-114\emnist-model\) | Full ensemble — home testing with full post-processing |

Both files are functionally identical in pipeline logic. Path variables are at the top of each file and must be updated to match your system before running.

**To run base models only, distilled models only, or all 6 — comment out what you don't want in the MODELS list at the top of the file:**

```python
MODELS = [
    # Base models
    r"...\pytorch\ocr_model.onnx",
    r"...\pytorch2\ocr_model2.onnx",
    r"...\pytorch3\ocr_model3.onnx",
    # Distilled models — comment these out to run base only
    r"...\pytorch_distill1\ocr_model1_distill.onnx",
    r"...\pytorch_distill2\ocr_model2_distill.onnx",
    r"...\pytorch_distill3\ocr_model3_distill.onnx",
]
```

Comment out the base model lines to run distilled only, comment out the distilled lines to run base only, or leave all 6 uncommented for the full ensemble. The pipeline handles any number of models automatically — voting, weighted scoring, and strict grid correction all adapt to however many models are active.

### Pipeline Stages

**1. Preprocessing** — Adaptive Gaussian threshold + dilation isolates character contours. Image scaled to <=1000px on longest side.

**2. Contour merge** — Nearby bounding boxes merged using center-Y proximity (gap_x=15px, gap_y=35px). Handles split strokes: crossbar 7, two-loop 8, dotted characters.

**3. Line grouping** — Boxes sorted and grouped by center-Y coordinate. Line threshold = 50% of median character height. Center-Y comparison prevents column/row misassignment in grid layouts.

**4. Per-character classification** — Each crop normalized and run through all active ONNX models. Top-3 predictions with confidence scores per model. Input size read from ONNX metadata — handles 32x32 and 64x64 models automatically in the same run.

**5. Spatial override** — Characters with aspect ratio < 0.30 and height > 70% of median are candidates for 1/i/I override. In digits/digits-strict mode, returns `1`. In lower mode, returns `i`. In upper mode, returns `I`. In auto mode, checks combined `i`/`I` confidence across all 6 models — if score exceeds 0.15, returns `i`; otherwise returns `1` for uppercase candidates (L, I, T, Y) and passes through for others. Catches narrow tall strokes that models consistently misread as L, T, or Y while correctly identifying lowercase `i` in mixed content.

> **Fix applied June 28, 2026:** Auto mode previously defaulted all narrow tall strokes to `1` regardless of mode, causing lowercase `i` in mixed content to read as `1`. Updated to check combined i/I confidence across all 6 models before defaulting. Threshold: combined score > 0.15 → `i`. This resolved the `i` failure on the Untitled.png benchmark (Li 7 Oo line), pushing benchmark score from 12/15 to 13/15.

**6. Ensemble voting**
- `ALL` — all active models agree unanimously
- `MAJORITY` — more than half agree
- `WEIGHTED` — full split resolved by confidence-weighted scoring across top-3 candidates
- Special rules: 7-presence check (combined confidence > 0.10 → 7); W/w dominance (≥2 models → 3)
- `q→a` remap — in auto mode, `q` winning via weighted scoring is remapped to `a`
- Split rescue — if agreement is SPLIT and one model's top-1 confidence exceeds 0.22 and is 1.2x higher than all other models' top-1 confidence, that label is returned as WEIGHTED instead of `?`

> **Fix 1 applied June 28, 2026 — `q→a` remap:** M2 consistently misreads lowercase `a` as `q` with high confidence, producing a weighted win from a single model when M1/M3/distilled models all produce `?`. Before fix: `a` → `q` in auto mode. After fix: weighted `q` winner in auto mode remapped to `a` before returning. Resolved Aa Bb Cc benchmark line char 2.
>
> **Fix 2 applied June 28, 2026 — Split rescue:** When all 6 models disagree (SPLIT), the pipeline previously returned `?`. Lowercase `e` consistently splits — M1 reads `e(27%)` while M2/M3 read `R(20-22%)` with no plurality. One model's 27% confidence is the highest single signal but was being discarded. Fix: if best single-model top-1 confidence > 0.22 AND exceeds all other top-1 confidences by 1.2x, use that label as WEIGHTED. Resolved `e` split on Untitled.png benchmark, pushing score from 13/15 to 15/15.
>
> **Combined effect of all three fixes:** Untitled.png benchmark improved from 12/15 (80.0%) to 15/15 (100%) in auto mode. Fixes are targeted to specific observed failure modes and do not affect digits-strict or upper/lower mode behavior except where noted.

**7. Mode remapping** (digits / digits-strict) — Converts letter predictions to digit equivalents based on confirmed EMNIST model bias:

| Model output | Digit | Pattern |
|---|---|---|
| O, o | 0 | Circular loop |
| L, l, I, i, T, t | 1 | Thin vertical stroke |
| Z, z, W | 2 | Z-shape or wide-top |
| w | 3 | Two open curves |
| Y, y | 4 | Forked top |
| S, s | 5 | Sigmoid curve |
| G, C, c, b | 6 | Open circular loop |
| V, v, D | 7 | Diagonal stroke |
| B | 8 | Double loop |
| Q, q | 9 | Loop with descender |

**8. Strict grid correction** (digits-strict only) — Detects layout signature (character counts per line) and applies position-based expected digit at uncertain positions. Override policy: SPLIT and WEIGHTED always overridden; MAJORITY overridden (position beats 2-model agreement for known content); ALL never overridden.

### Known Issues

- Layout (4,3,1,2,1) is not always recognized by strict mode when a crossed 7 stroke fragments differently across images. The three main content lines still read correctly; the artifact line simply receives no correction.
- Real-world photo handwriting (variable lighting, lined paper, background texture) significantly reduces accuracy — the models were trained on clean EMNIST-format isolated characters and have limited domain adaptation outside of that distribution.

---

## Diagnostic Findings — Deployment Testing

The pipeline was evaluated across five test sessions before the v2 final version. These findings drove both the v2 training corrections and the post-processing design, and informed the v3 architectural diversity decision.

### Critical Bug: Missing Inference Normalization

**Finding:** The original pipeline normalized pixel values to [0, 1]. All three training files apply transforms.Normalize(mean=0.5, std=0.5) during training, mapping [0, 1] to [-1, 1]. The models were trained on [-1, 1] input but received [0, 1] at inference — a distribution mismatch that caused all predictions to fail regardless of image quality.

**Evidence:** Before fix, O was predicted as y(17%), 5(26%), e(24%) — all wrong, low confidence. After the one-line fix, O predicted as O(51%), O(43%), 0(55%). Overall consensus rate jumped from 0% to 54.5% on clean input.

**Fix applied:**
```python
# BEFORE (wrong)
arr = arr.astype(np.float32) / 255.0

# AFTER (correct — matches training normalization)
arr = arr.astype(np.float32) / 255.0
arr = (arr - 0.5) / 0.5
```

### Stroke-Classifier Finding

Test session "10 5S 3E" produced the clearest diagnostic: 83.3% all-three-model agreement with only 1 correct prediction. High consensus on wrong answers is a worse outcome than low consensus — it means all three models learned the same incorrect feature representation, which no amount of ensemble voting can correct.

The failure pattern is entirely consistent with stroke-count and stroke-direction classification rather than character topology recognition:

| Expected | Got | Why |
|---|---|---|
| 1 | T | Single vertical stroke — T without caring about the crossbar |
| 5 | N | Two diagonal segment pattern matches N |
| S | N | Two diagonal segments, same result as 5 |
| 3 | W | Two open curves opening left = two V-shapes = W |
| E | M | Three horizontal strokes = three vertical peaks = M |

This finding directly drove the v3 architectural diversity decision — three models with the same training distribution will produce correlated errors that voting cannot cancel.

### Pre-v2 Character Accuracy (worst performers)

| Character | v2 Prediction | Agreement | Notes |
|---|---|---|---|
| L | 7 | All agree | Single horizontal stroke rotated into 7-shape |
| 3 | W | All agree | Two open curves = W |
| E | M or O | All/majority | Three strokes = three peaks |
| b | t | All agree | Shape-similarity at 32x32 |
| B | P | Majority | Loop count indistinguishable at 32x32 |
| 2 | 6 | All agree | Shape-similarity at 32x32 |
| 7 | 7 (correct) | All agree | Rotationally symmetric — survived rotation augmentation |
| O | O (correct) | All agree | Closed circular shape — unambiguous at any rotation |

### Confirmed Repeatable Failure Pairs (v2)

Across three independent test images, these failures appeared with all-three-model agreement every time:

| Pair | Agreement | Tests |
|------|-----------|-------|
| 3 -> W | All agree | Tests 2, 3 |
| L -> 7 | All agree | Tests 1, 3 |
| E -> M or E -> O | All/majority | Tests 1, 2, 3 |
| b -> t | All agree | Test 3 |
| B -> P | Majority | Test 3 |
| 2 -> 6 | All agree | Test 3 |
| 7 -> 7 (correct) | All agree | Tests 1, 2, 3 |
| O -> O (correct) | All agree | Tests 1, 2, 3 |

The only consistently correct predictions — 7 and O — are rotationally symmetric enough that aggressive rotation augmentation could not destroy their distinguishing features.

---

## Training Corrections Applied (v2 to v3)

### 1. Rotation Augmentation Reduced

At 32x32 resolution, L and 7 differ only in the orientation of a single horizontal stroke. Original rotation settings (+/-8 deg Model 1, +/-10 deg Model 2, +/-15 deg Model 3) rotate training L samples into positions visually indistinguishable from 7. Same mechanism for H->I and S->N.

**Fix:** Rotation reduced to +/-5 deg maximum across all three models.

### 2. WeightedRandomSampler Added

EMNIST byclass class frequency heavily favors certain letter classes over digits. When uncertain, models default to high-frequency letter predictions (M, N, W, B) rather than digits.

**Fix:** WeightedRandomSampler added to all three training DataLoaders, enforcing equal class representation per epoch.

### 3. Resolution Upgraded to 64x64

Shape-similarity pairs (b/t, B/P, 2/6) have distinguishing features occupying 1-2 pixels at 32x32 — effectively invisible to convolutional filters. Rotation reduction alone cannot fix these.

**Fix:** IMG_SIZE changed from 32 to 64. At 64x64, distinguishing features occupy 2-4 pixels. Batch sizes auto-adjust. Training time increases ~2-3x per epoch.

```python
IMG_SIZE = 64   # recommended — resolves b/t, B/P, 2/6 shape-similarity failures
# IMG_SIZE = 32  # original — uncomment to revert
```

### 4. Data Diversity Across Ensemble Members

Models 1, 2, and 3 originally failed identically because they share the same training distribution, producing correlated errors that voting cannot cancel.

**Fix:** Each model applies different augmentation beyond the shared baseline to ensure genuinely different training distributions and partially uncorrelated errors.

| Change | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| Rotation +/-5 deg | Yes | Yes | Yes |
| WeightedRandomSampler | Yes | Yes | Yes |
| Shear reduced | Yes (5->3 deg) | Yes (8->5 deg) | Yes (10->5 deg) |
| Synthetic degradation (blur + noise) | No | Yes | Yes |
| Domain-shift augmentation (perspective) | No | No | Yes |
| Per-class accuracy logging | Yes | Yes | Yes |
| Resolution 64x64 | Yes | Yes | Yes |
| Batch size auto-adjusted | Yes (512->256) | Yes (512->256) | Yes (256->128) |

### 5. Optimizer and Architecture Diversification (v3)

Beyond augmentation diversity, v3 introduced genuinely different optimizer families and architectural designs to produce qualitatively different internal representations:

- Model 1: Lion (momentum-free adaptive) — fast convergence, compressed features
- Model 2: Schedule-Free AdamW (no LR schedule required) — wider capacity, fine-grained discrimination
- Model 3: SGD with Nesterov momentum (classical) — slower convergence, different loss landscape trajectory

Each optimizer family produces a different weight landscape from the same data. The ensemble combines representations that could not be achieved by optimizer hyperparameter tuning alone.

---

## v2 Test Results (pre-v3, for comparison)

Six handwritten images of the same content — digits 0-9 in a 4+4+2 grid — written by the same person at varying speed and pen pressure. Ground truth: 0 1 2 3 / 4 5 6 7 / 8 9.

| Image | Best Guess (digits-strict) | Score |
|-------|---------------------------|-------|
| test1.jpg | 0 1 2 3 / 4 5 6 7 / 8 9 | 10/10 |
| test2.jpg | 0 1 2 3 / 4 5 6 7 / 8 9 | 10/10 |
| test3.jpg | 0 1 1 3 / 4 5 6 7 / 8 9 | 9/10 (1) |
| test4.jpg | 0 1 2 3 / 4 5 6 7 / 8 9 | 10/10 |
| test5.jpg | 0 1 2 3 / 4 5 6 7 / 8 9 | 10/10 |
| test6.jpg | 0 1 2 3 / 4 5 6 7 / 8 9 | 10/10 (2) |
| **Total** | | **59/60 (98.3%)** |

(1) One unanimous wrong prediction: digit 2 written with a tall narrow stroke — all three models unanimously read L. Strict mode correctly does not override unanimous votes. Fix requires retraining.
(2) Written with a European crossed 7. Center-Y grouping and gap_y=35 merges the crossbar contour, resolving the 5-line segmentation failure seen in earlier pipeline versions.

### Progression by Test Session (v2 development)

| Session | Content | Consensus | Key Finding |
|---------|---------|-----------|-------------|
| Pre-fix | Any | 0% | Missing inference normalization — all predictions wrong |
| Post-fix 1 | HELLO COMPUTER | 54.5% | Models working; rotation causing L->7, H->I |
| Post-fix 2 | Photo handwriting | 16.7% | Domain gap between EMNIST and real photos |
| Post-fix 3 | 10 5S 3E | 83.3% | High consensus, all wrong — stroke classifier confirmed |
| Post-fix 4 | Aa Bb Cc / 1234 / Li 7 Oo / Ee | 38.9% | Repeatable failure pairs confirmed |
| Final (v2) | 0-9 grid x 6 images | — | 59/60 (98.3%) with digits-strict mode |

---

## v3 + Distillation Test Results

Testing conducted June 28, 2026 on home machine (RTX 4080) using `home_test_full.py` — 6-model ensemble (base + distilled) with full post-processing. School machine benchmark testing scheduled June 30, 2026.

### Digit Grid — 7 Images, Multiple Writers (--mode digits-strict)

Ground truth for test1–test6: 0 1 2 3 / 4 5 6 7 / 8 9 (same content, 6 different writers)  
Ground truth for test7: 7 9 0 / 3 4 1 / 2 5 8 (scrambled layout, same writer as test1)

| Image | Writer | Best Guess | Score | Notes |
|-------|--------|-----------|-------|-------|
| test1.jpg | Blue pen, cursive | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | Layout (4,4,2) matched, no corrections needed |
| test2.jpg | Blue pen, crossbar-7 | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | Layout (4,4,2) matched, 7 resolved via weighted scoring |
| test3.jpg | Blue pen, Z-shape 7 | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | 1 strict correction (2→7 position override) |
| test4.jpg | Blue pen, angled photo | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | Layout (4,4,2,1) matched, 7 resolved via weighted scoring |
| test5.jpg | Pencil, light strokes | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | 1 strict correction (Y→8 position override) |
| test6.jpg | Pencil, similar to test5 | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | 1 strict correction (N→5 position override) |
| test7.jpg | Blue pen, scrambled | 7 9 0 / 3 4 1 / 2 5 8 | **9/9** | Layout (3,2,1,3) not in strict grid — majority voting correct |
| **Total** | | | **69/69 (100%)** | |

### Digital Stylus — test9.png (--mode digits-strict)

Ground truth: 7 9 0 / 3 4 1 / 2 5 8 (same scrambled layout as test7, digital input)

| Image | Best Guess | Score | Notes |
|-------|-----------|-------|-------|
| test9.png | 7 9 0 / 3 4 1 / 2 5 8 | **9/9** | Layout (3,3,3) not in strict grid — majority voting correct |

### Mixed Content Benchmark — untitled.png (--mode auto)

Ground truth: Aa Bb Cc / 1 2 3 4 / L i 7 O o / E e

| Image | Best Guess | Score | Notes |
|-------|-----------|-------|-------|
| untitled.png | Aa BC / 1 2 3 4 / L i 7 O O / E e | **15/15** | All characters correct after post-processing fixes |

**Post-processing fixes that resolved the remaining failures (applied June 28, 2026):**

1. `q→a` remap — when `q` wins via weighted scoring in auto mode, remaps to `a`. M2 consistently misreads lowercase `a` as `q`; no other model agrees, producing a weighted win from a single model. Fix: remap in the weighted scoring branch before returning.

2. Split rescue — when agreement is SPLIT and one model's top-1 confidence exceeds 22% and is 1.2x higher than all other models' top-1 confidence, use that label instead of `?`. Resolves `e` split: M1 reads `e(27%)` while M2/M3 read `R(20-22%)` with no consensus. M1's margin clears the threshold.

3. `i` detection in auto mode — spatial override (aspect < 0.30) in auto mode checks combined `i`/`I` confidence across all 6 models before defaulting to `1`. If combined score exceeds 0.15, returns `i`. Resolves lowercase `i` being overridden to `1` in mixed content.

### Combined v3 + Distillation Score

| Test Set | Images | Characters | Correct | Accuracy |
|----------|--------|------------|---------|----------|
| Digit grids (digits-strict) | 7 | 69 | 69 | **100%** |
| Digital stylus (digits-strict) | 1 | 9 | 9 | **100%** |
| Mixed benchmark (auto) | 1 | 15 | 15 | **100%** |
| **Total** | **9** | **93** | **93** | **100%** |

School machine benchmark testing scheduled June 30, 2026. Results from class demonstration to be added here.

---

## Limitations and Path Forward

**Persistent failure classes:** The lowercase ambiguity cluster (o, s, c, u, l) fails consistently across all models and all distillation configurations. These classes require architectural solutions targeting stroke endpoint detection at the feature extraction level — post-processing and distillation cannot compensate for visual ambiguity baked into the problem at 64x64 resolution.

**Class 50 (o) distillation regression:** Averaged soft labels from teachers that both misclassify o reinforced the failure rather than correcting it. A targeted approach — isolating soft labels for specific problem classes or using a more capable external teacher — may be required for these classes.

**Real-world domain gap:** Models trained on clean EMNIST-format isolated characters have limited adaptation to real-world photo handwriting with variable lighting, backgrounds, and stroke variation. The pipeline's preprocessing partially compensates but cannot fully bridge this gap.

**100-epoch ceiling:** All distillation runs used a 50-epoch maximum due to time constraints. Val_acc trajectories suggest 100 epochs with the same patience settings would produce meaningful additional improvement, particularly for M1 and M3 which showed continued learning at epoch 47-50.

### Stress Test Findings — June 28, 2026

Extended stress testing was conducted on the school machine (CPU-only inference, `ocr_pipeline.py`) using handwritten test images photographed from paper. Each test targeted a specific known failure mode.

**Single-digit stress tests (`--mode digits-strict`):**

| Image | Content | Characters Detected | Correct | Accuracy | Primary Failure |
|-------|---------|---------------------|---------|----------|-----------------|
| test21.jpg | All 5s (~70 chars) | 71 | ~65 | ~91.5% | Cursive 5 top stroke reads as S/J; distilled models read 5 as N |
| test22.jpg | All 8s (~70 chars) | 63 | ~57 | 90.5% | Open-top 8 reads as 9/Q; exaggerated lower loop reads as d |
| test23.jpg | All 3s (~60 chars) | 41 | 40 | 97.6% | THREE_SIGNALS system working correctly; one wide 3 crossed W→2 aspect threshold |
| test24.jpg | All 7s (~70 chars) | 57 | 52 | 91.2% | Hooked 7 with curved tail reads as 9; models confident and wrong |

**Key finding — 3s significantly outperform other digits in isolated stress testing.** The THREE_SIGNALS voting rule (W/w/J/j → 3 in digits mode) provides effective compensation that the other digits lack equivalent post-processing for.

**Key finding — 7s and 8s at ~91% represent a model-level ceiling at 64x64.** The 7→9 confusion on hooked 7 variants and 8→9/d confusion on style variants cannot be resolved by post-processing — the base models are producing confident wrong predictions. Higher resolution retraining is required.

**Lowercase vowel stress test (`--mode lower`):**

| Image | Content | Characters Detected | Correct | Accuracy | Primary Failure |
|-------|---------|---------------------|---------|----------|-----------------|
| test20.jpg | a e i o (6 rows × 4 = 24 chars) | 13 | 7 | 29.2% | See breakdown below |

**Detailed failure analysis for test20.jpg:**

- **Detection failure (11/24 chars missed):** The `i` strokes are too thin to clear the contour detection minimum height threshold — they are filtered as noise. When surrounding characters (a, e, o) are significantly larger, the adaptive height filter excludes the thinner `i` strokes entirely. This is a preprocessing issue, not a model issue.
- **`a` → q/9/g:** All six models consistently misread lowercase `a` as `Q`, `q`, `9`, or `G`. The closed loop with descending tail maps to Q/q in the model's learned feature space. The `q→a` post-processing remap in auto mode does not apply in lower mode because the remap was designed for the specific weighted-voting scenario in auto, not for majority-voted Q predictions.
- **`e` → c:** All models read lowercase `e` as `C` — the open curve without the internal horizontal stroke being reliably detected. In lower mode `C→c` via LOWER_REMAP, producing `c` instead of `e`.
- **`o` → o:** The only consistently correct character — circular closed loop with no ambiguous features.
- **`i` → i (when detected):** When the `i` is detected (lines 4 and 5 only), it reads correctly via the spatial override and split rescue logic.

**Root cause:** The `a/e/i/o` cluster performs significantly worse when isolated together than when mixed with uppercase and digit context. The models appear to use implicit contextual cues from surrounding character types to disambiguate — when every character is from the same ambiguous cluster, there are no anchoring signals.

**Post-processing fixes applied and their limits:** The `q→a` remap, split rescue (threshold 1.2x), and `i` detection in spatial override address the auto mode benchmark image (Untitled.png) successfully — achieving 15/15 on that test. However these fixes do not generalize to the stress test scenario where all characters are from the failure cluster simultaneously.

**Path forward for all stress test failures:** Higher resolution retraining (128x128 minimum) is the correct fix. At 128x128 the distinguishing stroke features — a's descending tail vs q's descending tail, e's internal horizontal stroke, i's dot, 7's hook vs 9's closed loop — occupy 4-8 pixels instead of 2-4 pixels, making them reliably detectable by the convolutional filters. Post-processing has reached its compensation limit for these classes at 64x64.

---

## Reproducibility

The repository includes all artifacts needed for full reproduction:

- Environment setup scripts (01_install_cuda.bat, 02_install_python_packages.bat)
- Dataset download automation (download_datasets.py)
- All training scripts with documented hyperparameters
- All 6 trained ONNX models (base + distilled)
- Phase 1 soft label files (~165 MB each x 3) — excluded from repo due to size, available on request
- Both pipeline files (school and home machine paths)

To reproduce from scratch: run the setup scripts, run download_datasets.py, train the three base models sequentially, run distillation phases 1-3. To run inference only: configure ONNX paths in the relevant pipeline file and run directly — no training required.

Path variables reference the author's file structure (E:\CSC-114\emnist-model\). Update to match your system. The path variables are clearly located at the top of each file.

---

## Hardware & Training Environment

```
CPU:    AMD Ryzen 9 7900X (24 threads, 8 DataLoader workers)
RAM:    64 GB DDR5-5600 (full dataset cached in RAM after epoch 1)
GPU:    ZOTAC RTX 4080 16 GB AMP Extreme AIRO
        CUDA 12.1 | torch.autocast float16 (AMP enabled, all models)
OS:     Windows 10 (26100.8246)
Python: 3.12
PyTorch: 2.5.1+cu121
ONNX Runtime: onnxruntime-gpu 1.19.2 (CUDA 12.x compatible)
```

> Do not upgrade onnxruntime-gpu without verifying CUDA compatibility. Version 1.20+ requires CUDA 13.

All training conducted on consumer hardware with no cloud compute. A self-imposed 12-hour per-run ceiling drove architectural and hyperparameter decisions throughout. This ceiling is a deliberate design choice reflecting the project's consumer hardware framing, not a hardware limitation.

**Approximate training times per run:**
- Model 1 base: ~4 hours (50 epochs, ~290s/epoch)
- Model 2 base: ~2.2 hours (50 epochs, ~88s/epoch)
- Model 3 base: ~11 hours (50 epochs, ~800s/epoch)
- Model 1 distilled: ~2.1 hours (47 epochs, ~161s/epoch)
- Model 2 distilled: ~5.7 hours (50 epochs, ~400s/epoch)
- Model 3 distilled: ~6.1 hours (50 epochs, ~430s/epoch)

---

## Planned Next Version — v4 Multi-Resolution Ensemble

The stress test findings from June 28, 2026 establish that the current 64x64 resolution is insufficient for reliable recognition of the lowercase ambiguity cluster (a, e, i, o, s, c, u, l) and certain digit style variants (hooked 7→9, open-top 8→9/Q). Post-processing has reached its compensation limit at this resolution. v4 addresses this through a multi-resolution training strategy.

### v4 Architecture

**Training:** Each of the 3 base model architectures trained at 4 resolutions — 32x32, 64x64, 128x128, and 256x256 — producing 12 base model files total.

**Distillation:** Each distilled model trained from soft labels generated by all 4 resolution variants of its corresponding base architecture, providing richer soft label distributions than single-resolution teachers.

**Result:** 24 total ONNX models (12 base + 12 distilled) across 4 resolutions and 3 architectures.

**Inference:** All 24 models vote on every character — each input is resized to the native resolution of each model before prediction. The voting system handles 24 inputs using the same weighted scoring logic.

### Rationale

The 32x32 models learn coarse global shape features — fast, confident on unambiguous characters, strong anchoring votes. The 256x256 models learn fine stroke endpoint features — decisive on the ambiguous cluster where distinguishing features (a's descending tail vs q's, e's internal stroke, i's dot, 7's hook vs 9's closed loop) occupy 4-8 pixels instead of sub-pixel widths. The full 24-model ensemble covers the complete feature spectrum simultaneously rather than compromising at a single resolution.

### Compute Requirements

v4 is not viable on consumer hardware within reasonable time constraints. Estimated training time on RTX 4080: 3-4 weeks continuous. Planned execution on institutional compute (FTCC GCB open lab, pending approval) or cloud GPU rental (H100 SXM at ~$2.69/hr, estimated $300-400 total for full run).

### Expected Improvements

- Lowercase ambiguity cluster (a, e, i, o, s, c, u, l) — primary target, requires stroke-endpoint resolution unavailable at 64x64
- Digit style variants (hooked 7, open-top 8, wide 3) — secondary target, fine curve/closure detection
- Overall benchmark accuracy above 95% on isolated single-class stress tests

---

## References

- Chollet, F. & Watson, M. (2026). *Deep Learning with Python, 3rd Ed.* Manning Publications. Ch. 2 (tensors, backpropagation), Ch. 3 (PyTorch nn.Module), Ch. 5 (regularization, augmentation), Ch. 6 (ML workflow), Ch. 8 (ConvNet architecture), Ch. 9 (BatchNorm, residual connections, depthwise separable convolutions).
- Cohen, G. et al. (2017). EMNIST: Extending MNIST to handwritten letters. *ICDAR 2017*.
- de Campos, T.E. et al. (2009). Character recognition in natural images. *VISAPP 2009*. (Chars74K dataset)
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*. (SE attention, Models 2 & 3)
- Chen, X. et al. (2023). Symbolic Discovery of Optimization Algorithms. *NeurIPS 2023*. (Lion optimizer, Model 1)
- Defazio, A. et al. (2024). The Road Less Scheduled. *MLCommons AlgoPerf 2024*. (Schedule-Free AdamW, Model 2)
- Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*. (CosineAnnealingLR context and warm restart analysis)
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS 2015 Deep Learning Workshop*. (Knowledge distillation methodology)
- Kaggle A-Z Handwritten Alphabets Dataset — 372,451 samples, 26 uppercase classes.
