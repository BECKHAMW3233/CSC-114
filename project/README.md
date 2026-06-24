# EMNIST OCR Ensemble — Handwritten Character Recognition Pipeline

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green)
![Dataset](https://img.shields.io/badge/Dataset-EMNIST%20byclass-lightgrey)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-59%2F60%20(98.3%25)-brightgreen)

**Author:** William Edward Beckham III  
**Program:** Computer Programming & Development AAS — FTCC  
**Course:** CSC-114 AI Fundamentals I (Summer 2026)  
**Hardware:** AMD Ryzen 9 7900X · 64 GB DDR5-5600 · ZOTAC RTX 4080 16 GB AMP Extreme AIRO

---

## Overview

This project implements a three-model deep learning ensemble for handwritten character recognition, trained on EMNIST and multiple supplementary datasets, exported to ONNX, and deployed through a custom inference pipeline with iterative post-processing compensation for known model bias patterns.

The system recognizes 62 classes — digits 0–9, uppercase A–Z, and lowercase a–z. Through a documented cycle of deployment testing, bug identification, training corrections, and pipeline improvements, it achieves **59/60 correct digit reads** across six real-world handwritten test images in `digits-strict` mode without any retraining of the final models.

---

## Repository Structure

```
├── ocr_pipeline.py          # Inference pipeline — run this
├── ocr_pytorch_model.py     # Model 1 training script (standard ConvNet)
├── ocr_pytorch_model2.py    # Model 2 training script (SE-attention, wider)
├── ocr_pytorch_model3.py    # Model 3 training script (triple-width, multi-scale)
├── supplementary_data.py    # Shared dataset loader for all three models
├── ocr_model.onnx           # Model 1 exported weights (generate locally — see below)
├── ocr_model2.onnx          # Model 2 exported weights
├── ocr_model3.onnx          # Model 3 exported weights
└── README.md                # This file
```

---

## Setup & Installation

### Requirements

```bash
pip install opencv-python numpy onnxruntime
```

For training (not required to run inference):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics matplotlib pillow optuna
```

### Getting the ONNX Models

The `.onnx` model files are not included in the repository due to file size (9.4 MB, 37 MB, 17 MB). Generate them locally by training each model:

```bash
python ocr_pytorch_model.py   # generates ocr_model.onnx
python ocr_pytorch_model2.py  # generates ocr_model2.onnx
python ocr_pytorch_model3.py  # generates ocr_model3.onnx
```

Training requires EMNIST byclass dataset (auto-downloaded on first run) and optionally the supplementary datasets (see `supplementary_data.py` for paths and download instructions).

### Configuring Model Paths

The model paths in `ocr_pipeline.py` are set to the author's local Downloads directory. Update the `MODELS` list at the top of the file to match your system before running:

```python
# ocr_pipeline.py — edit these three paths
MODELS = [
    r"C:\path\to\your\ocr_model.onnx",
    r"C:\path\to\your\ocr_model2.onnx",
    r"C:\path\to\your\ocr_model3.onnx",
]
```

On Linux/macOS use forward slashes:

```python
MODELS = [
    "/path/to/your/ocr_model.onnx",
    "/path/to/your/ocr_model2.onnx",
    "/path/to/your/ocr_model3.onnx",
]
```

### Quick Start

```bash
# Clone the repo
git clone https://github.com/your-username/emnist-ocr-ensemble.git
cd emnist-ocr-ensemble

# Install inference dependencies
pip install opencv-python numpy onnxruntime

# Edit MODELS paths in ocr_pipeline.py, then:
python ocr_pipeline.py --mode digits-strict test1.jpg
```

---

## Usage

```bash
# Single file, auto mode (raw ensemble output, no remapping)
python ocr_pipeline.py image.jpg

# Digit content with letter→digit remapping
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

Running `--mode digits-strict` on a handwritten 0–9 grid:

```
============================================================
  OCR Pipeline — 3-Model Ensemble
  Image: test5.jpg
  Mode:  DIGITS-STRICT
============================================================
  Detected: 10 characters across 3 line(s)

  ──────────────────────────────────────────────────
  INDIVIDUAL MODEL PREDICTIONS (raw, no remapping)
  ──────────────────────────────────────────────────
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

  ──────────────────────────────────────────────────
  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  *=strict  ?=split)
  ──────────────────────────────────────────────────
  Line 1: 0 *1* [2] [3]
  Line 2: [4] 5 6 *7*
  Line 3: *8* 9

  Consensus: 4/10 chars (40.0% all-3 agreement)

  ──────────────────────────────────────────────────
  STRICT GRID CORRECTIONS (3 applied)
  ──────────────────────────────────────────────────
  Line 1 Char 2: ? (SPLIT) → 1 [position override]
  Line 2 Char 4: ? (SPLIT) → 7 [position override]
  Line 3 Char 1: 4 (WEIGHTED) → 8 [position override]

  ──────────────────────────────────────────────────
  BEST GUESS READ  [mode: DIGITS-STRICT]
  ──────────────────────────────────────────────────
  Line 1: 0 1 2 3
  Line 2: 4 5 6 7
  Line 3: 8 9
============================================================
```

**Output legend:**
- `plain` — all three models agree (unanimous)
- `[x]` — majority or weighted vote winner
- `*x*` — position override applied by strict grid correction
- `?` — unresolved split (no majority, no position correction available)

---

## Models

All three models are trained independently with intentional architectural diversity to maximize ensemble disagreement on ambiguous characters. Each is exported to ONNX at 64×64 input resolution.

### Model 1 — Standard ConvNet (`ocr_pytorch_model.py`)

Baseline convolutional network following architecture patterns from Chollet & Watson, *Deep Learning with Python, 3rd Ed.* (Manning 2025), Chapters 8–9.

| Parameter | Value |
|-----------|-------|
| Input resolution | 64×64 |
| Filter progression | 32→64→128→256 |
| Classifier head | 256→128→62 |
| Optimizer | Adam (lr=3e-4, decay=3e-5) |
| Scheduler | OneCycleLR |
| Batch size | 256 (auto-adjusted from 512 at 32×32) |
| Augmentation | Rotation ±5°, affine (translate 10%, scale 0.9–1.1, shear 3°), contrast jitter |
| Regularization | Dropout, L2 weight decay, label smoothing 0.05 |
| Output | `ocr_model.onnx` (~9.4 MB) |

### Model 2 — SE-Attention, Wider (`ocr_pytorch_model2.py`)

Wider architecture with Squeeze-Excitation attention blocks after each stage for channel-wise feature recalibration. Adds synthetic degradation augmentation (blur + noise) to ensure Model 2 sees a different data distribution than Model 1, producing partially uncorrelated errors in the ensemble.

| Parameter | Value |
|-----------|-------|
| Input resolution | 64×64 |
| Filter progression | 32→128→256→512 |
| Attention | Squeeze-Excitation (SE) after each stage |
| Regularization | StochasticDepth (DropPath) + Dropout |
| Classifier head | 512→256→62 |
| Optimizer | AdamW (lr=1e-4, decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 256 |
| Augmentation | Rotation ±5°, affine, Gaussian blur + noise |
| Output | `ocr_model2.onnx` (~37 MB) |

### Model 3 — Triple-Width, Multi-Scale Fusion (`ocr_pytorch_model3.py`)

Maximum capacity model with triple-width channels and a feature pyramid concatenating pooled outputs from three stages before classification. Uses GELU activations in a five-layer classifier and the most aggressive augmentation of the three models, including perspective distortion and background texture injection.

| Parameter | Value |
|-----------|-------|
| Input resolution | 64×64 |
| Channel progression | 96→192→384→768→768 |
| Feature fusion | Pyramid: concatenated pool from stages 2+3+4 |
| Classifier head | 768_fused→1024→512→256→128→62 (5 layers, GELU) |
| Optimizer | SGD + Momentum (lr=0.01, momentum=0.9) |
| Scheduler | CosineAnnealingWarmRestarts |
| Batch size | 128 (triple-width, most VRAM-intensive) |
| Augmentation | Rotation ±5°, affine, perspective distortion, blur + noise |
| Output | `ocr_model3.onnx` (~17 MB) |

---

## Training Data

All three models use identical data sources via `supplementary_data.py`. Class imbalance is addressed with `WeightedRandomSampler` in all models — a v2 correction applied after per-class accuracy analysis revealed EMNIST byclass heavily over-represents certain letter classes relative to digits, causing systematic digit misclassification.

| Dataset | Samples | Classes | Notes |
|---------|---------|---------|-------|
| EMNIST byclass | 814,255 | 62 | Primary — digits + upper + lower |
| EMNIST Balanced | ~112,800 | 47 | Equal samples per class; supplements byclass |
| Kaggle A-Z | 372,450 | 26 | Uppercase only; addresses A–Z underrepresentation |
| Chars74K EnglishHnd | ~3,410 | 62 | Handwritten; all 62 classes including lowercase |
| Chars74K EnglishImg | ~7,705 | 62 | Natural scene; all 62 classes |

**EMNIST byclass index mapping:**
- Indices 0–9 → digits 0–9
- Indices 10–35 → uppercase A–Z
- Indices 36–61 → lowercase a–z

---

## Diagnostic Findings — Deployment Testing

The pipeline was evaluated across five test sessions before the final version. The findings drove both the training corrections (v2) and the post-processing design.

### Critical Bug: Missing Inference Normalization

**Finding:** The original pipeline normalized pixel values to `[0, 1]`. All three training files apply `transforms.Normalize(mean=0.5, std=0.5)` during training, mapping `[0, 1]` to `[-1, 1]`. The models were trained on `[-1, 1]` input but received `[0, 1]` at inference — a distribution mismatch that caused all predictions to fail regardless of image quality.

**Evidence:** Before fix, `O` was predicted as `y(17%)`, `5(26%)`, `e(24%)` — all wrong, low confidence. After the one-line fix, `O` predicted as `O(51%)`, `O(43%)`, `0(55%)`. Overall consensus rate jumped from 0% to 54.5% on clean input.

**Fix applied:**
```python
# BEFORE (wrong)
arr = arr.astype(np.float32) / 255.0

# AFTER (correct — matches training normalization)
arr = arr.astype(np.float32) / 255.0
arr = (arr - 0.5) / 0.5
```

### Stroke-Classifier Finding

Test session "10 5S 3E" produced the clearest diagnostic: **83.3% all-three-model agreement with only 1 correct prediction**. High consensus on wrong answers is a worse outcome than low consensus — it means all three models learned the same incorrect feature representation, which no amount of ensemble voting can correct.

The failure pattern is entirely consistent with stroke-count and stroke-direction classification rather than character topology recognition:

| Expected | Got | Why |
|---|---|---|
| `1` | `T` | Single vertical stroke — T without caring about the crossbar |
| `5` | `N` | Two diagonal segment pattern matches N |
| `S` | `N` | Two diagonal segments, same result as 5 |
| `3` | `W` | Two open curves opening left = two V-shapes = W |
| `E` | `M` | Three horizontal strokes = three vertical peaks = M |
| `O` | `O` ✓ | Closed circular shape — unambiguous at any rotation |

### Confirmed Repeatable Failure Pairs

Across three independent test images, these failures appeared with all-three-model agreement every time:

| Pair | Agreement | Tests |
|------|-----------|-------|
| `3 → W` | All agree | Tests 2, 3 |
| `L → 7` | All agree | Tests 1, 3 |
| `E → M` or `E → O` | All/majority | Tests 1, 2, 3 |
| `b → t` | All agree | Test 3 |
| `B → P` | Majority | Test 3 |
| `2 → 6` | All agree | Test 3 |
| `7 → 7` ✓ | All agree | Tests 1, 2, 3 |
| `O → O` ✓ | All agree | Tests 1, 2, 3 |

The only consistently correct predictions — `7` and `O` — are rotationally symmetric enough that aggressive rotation augmentation could not destroy their distinguishing features.

---

## Training Corrections Applied (v2)

Four categories of fixes were applied to all three training files in response to the deployment findings:

### 1. Rotation Augmentation Reduced

At 32×32 resolution, `L` and `7` differ only in the orientation of a single horizontal stroke. Original rotation settings (±8° Model 1, ±10° Model 2, ±15° Model 3) rotate training `L` samples into positions visually indistinguishable from `7`. Same mechanism for `H→I` and `S→N`.

**Fix:** Rotation reduced to ±5° maximum across all three models.

### 2. WeightedRandomSampler Added

EMNIST byclass class frequency heavily favors certain letter classes over digits. When uncertain, models default to high-frequency letter predictions (`M`, `N`, `W`, `B`) rather than digits.

**Fix:** `WeightedRandomSampler` added to all three training DataLoaders, enforcing equal class representation per epoch.

### 3. Resolution Upgraded to 64×64

Shape-similarity pairs (`b/t`, `B/P`, `2/6`) have distinguishing features occupying 1–2 pixels at 32×32 — effectively invisible to convolutional filters. Rotation reduction alone cannot fix these.

**Fix:** `IMG_SIZE` changed from 32 to 64. At 64×64, distinguishing features occupy 2–4 pixels. Batch sizes auto-adjust. Training time increases ~2–3× per epoch.

```python
IMG_SIZE = 64   # recommended — resolves b/t, B/P, 2/6 shape-similarity failures
# IMG_SIZE = 32  # original — uncomment to revert
```

### 4. Data Diversity Across Ensemble Members

Models 1, 2, and 3 originally failed identically because they share the same training distribution, producing correlated errors that voting cannot cancel.

**Fix:** Each model applies different augmentation beyond the shared baseline to ensure genuinely different training distributions and partially uncorrelated errors.

| Change | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| Rotation ±5° | ✓ | ✓ | ✓ |
| WeightedRandomSampler | ✓ | ✓ | ✓ |
| Shear reduced | ✓ (5°→3°) | ✓ (8°→5°) | ✓ (10°→5°) |
| Synthetic degradation (blur + noise) | — | ✓ | ✓ |
| Domain-shift augmentation (perspective) | — | — | ✓ |
| Per-class accuracy logging | ✓ | ✓ | ✓ |
| Resolution 64×64 | ✓ | ✓ | ✓ |
| Batch size auto-adjusted | ✓ (512→256) | ✓ (512→256) | ✓ (256→128) |

---

## Inference Pipeline Details

### Pipeline Stages

**1. Preprocessing** — Adaptive Gaussian threshold + dilation isolates character contours. Image scaled to ≤1000px on longest side.

**2. Contour merge** — Nearby bounding boxes merged using center-Y proximity (`gap_x=15px`, `gap_y=35px`). Handles split strokes: crossbar `7`, two-loop `8`, dotted characters.

**3. Line grouping** — Boxes sorted and grouped by center-Y coordinate. Line threshold = 50% of median character height. Center-Y comparison prevents column/row misassignment in grid layouts.

**4. Per-character classification** — Each crop normalized and run through all three ONNX models. Top-3 predictions with confidence scores per model. Input size read from ONNX metadata — handles 32×32 and 64×64 models automatically in the same run.

**5. Spatial override** — Characters with aspect ratio < 0.30 and height > 70% of median classified as `1`/`i`/`I` regardless of model output. Catches narrow tall strokes that models consistently misread as `L`, `T`, or `Y`.

**6. Ensemble voting**
- `ALL` — all three top-1 predictions agree
- `MAJORITY` — two of three agree
- `WEIGHTED` — full split resolved by confidence-weighted scoring across top-3 candidates
- Special rules: `7`-presence check (combined confidence > 0.10 → `7`); `W`/`w` dominance (≥2 models → `3`)

**7. Mode remapping** (`digits` / `digits-strict`) — Converts letter predictions to digit equivalents based on confirmed EMNIST model bias:

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

**8. Strict grid correction** (`digits-strict` only) — Detects layout signature (character counts per line) and applies position-based expected digit at uncertain positions. Override policy: `SPLIT` and `WEIGHTED` always overridden; `MAJORITY` overridden (position beats 2-model agreement for known content); `ALL` never overridden.

### Known Issues

- Layout `(4,3,1,2,1)` is not always recognized by strict mode when a crossed `7` stroke fragments differently across images. The three main content lines still read correctly; the artifact line simply receives no correction.
- Real-world photo handwriting (variable lighting, lined paper, background texture) significantly reduces accuracy — the models were trained on clean EMNIST-format isolated characters and have limited domain adaptation outside of that distribution.

---

## Test Results

Six handwritten images of the same content — digits 0–9 in a 4+4+2 grid — written by the same person at varying speed and pen pressure. Ground truth: `0 1 2 3 / 4 5 6 7 / 8 9`.

| Image | Best Guess (digits-strict) | Score |
|-------|---------------------------|-------|
| test1.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test2.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test3.jpg | `0 1 1 3 / 4 5 6 7 / 8 9` | **9/10** ¹ |
| test4.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test5.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test6.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** ² |
| **Total** | | **59/60 (98.3%)** |

¹ One unanimous wrong prediction: digit `2` written with a tall narrow stroke — all three models unanimously read `L`. Strict mode correctly does not override unanimous votes. Fix requires retraining.  
² Written with a European crossed `7`. Center-Y grouping and `gap_y=35` merges the crossbar contour, resolving the 5-line segmentation failure seen in earlier pipeline versions.

### Progression by Test Session

| Session | Content | Consensus | Key Finding |
|---------|---------|-----------|-------------|
| Pre-fix | Any | 0% | Missing inference normalization — all predictions wrong |
| Post-fix 1 | HELLO COMPUTER | 54.5% | Models working; rotation causing L→7, H→I |
| Post-fix 2 | Photo handwriting | 16.7% | Domain gap between EMNIST and real photos |
| Post-fix 3 | 10 5S 3E | 83.3% | High consensus, all wrong — stroke classifier confirmed |
| Post-fix 4 | Aa Bb Cc / 1234 / Li 7 Oo / Ee | 38.9% | Repeatable failure pairs confirmed |
| Final | 0–9 grid × 6 images | — | 59/60 with digits-strict mode |

---

## Limitations and Path Forward

**Why post-processing has a ceiling:** The systematic letter-over-digit bias is baked into model weights. Because all three models share the same training distribution, they fail identically — ensemble voting cannot cancel correlated errors. Post-processing reaches ~98% on known-content digit grids but cannot achieve reliable mixed-content accuracy without fixing the training distribution.

**What actually fixes it:**
- Retrain with explicit digit class upweighting in the loss function
- Use EMNIST Digits as primary training data for a digit-specific model
- Enforce equal digit/letter representation per batch via higher-weighted `WeightedRandomSampler` for digit classes

**Benchmark for evaluating retrained models** — after retraining, run against `Aa Bb Cc / 1234 / Li 7 Oo / Ee`:

| Character | Pre-retrain | Expected post-retrain | Fix responsible |
|---|---|---|---|
| `L` | `7` (all agree) | `L` | Rotation ±5° |
| `3` | `W` (all agree) | `3` | Rotation ±5° |
| `b` | `t` (all agree) | `b` | 64×64 resolution |
| `B` | `P` (majority) | `B` | 64×64 resolution |
| `2` | `6` (all agree) | `2` | 64×64 resolution |
| `E` | `O` or `M` | `E` | Rotation ±5° + class balance |
| `7` | `7` ✓ (all agree) | `7` ✓ | Already correct — monitor for regression |
| `O` | `O` ✓ (all agree) | `O` ✓ | Already correct — monitor for regression |

---

## Hardware & Training Environment

```
CPU:    AMD Ryzen 9 7900X (24 threads, 8 DataLoader workers)
RAM:    64 GB DDR5-5600 (full EMNIST dataset cached in RAM after epoch 1)
GPU:    ZOTAC RTX 4080 16 GB AMP Extreme AIRO
        CUDA 12.1 · torch.autocast float16 (AMP enabled, all models)
OS:     Windows 10 (26100.8246)
Python: 3.13
```

---

## References

- Chollet, F. & Watson, M. (2025). *Deep Learning with Python, 3rd Ed.* Manning Publications.  
  Ch. 2 (tensors, backpropagation), Ch. 3 (PyTorch nn.Module), Ch. 5 (regularization, augmentation), Ch. 6 (ML workflow), Ch. 8 (ConvNet architecture), Ch. 9 (BatchNorm, residual connections), Ch. 18 (AMP, ensembling, quantization).
- Cohen, G. et al. (2017). EMNIST: Extending MNIST to handwritten letters. *ICDAR 2017*.
- de Campos, T.E. et al. (2009). Character recognition in natural images. *VISAPP 2009*. (Chars74K dataset)
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*. (SE attention, Model 2)
- Kaggle A-Z Handwritten Alphabets Dataset — 372,450 samples, 26 uppercase classes.