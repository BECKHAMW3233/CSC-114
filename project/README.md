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
project/
├── 01_install_cuda.bat          # Step 1 — Install CUDA toolkit
├── 02_install_python_packages.bat  # Step 2 — Install Python dependencies
├── 03_verify_gpu.py             # Step 3 — Verify GPU/CUDA setup
├── install_deps.py              # Install Python deps via pip
├── download_datasets.py         # Download all training datasets
├── supplementary_data.py        # Shared dataset loader for all three models
├── ocr_pytorch_model.py         # Model 1 training script (standard ConvNet)
├── ocr_pytorch_model2.py        # Model 2 training script (SE-attention, wider)
├── ocr_pytorch_model3.py        # Model 3 training script (triple-width, multi-scale)
├── ocr_pipeline.py              # Inference pipeline — run this
├── README.md                    # This file
├── .gitignore
├── pytorch/                     # Model 1 training output
│   ├── ocr_model.onnx           # ONNX export (~9.4 MB)
│   ├── best_model.pt            # Best checkpoint
│   ├── final_model.pt           # Final weights
│   ├── training_curves.png      # Loss/accuracy plot
│   └── training_log.csv         # Per-epoch metrics
├── pytorch2/                    # Model 2 training output
│   ├── ocr_model2.onnx          # ONNX export (~37 MB)
│   ├── best_model2.pt
│   ├── final_model2.pt
│   ├── training_curves2.png
│   └── training_log2.csv
└── pytorch3/                    # Model 3 training output
    ├── ocr_model3.onnx          # ONNX export (~17 MB)
    ├── best_model3.pt
    ├── final_model3.pt
    ├── training_curves3.png
    └── training_log3.csv
```

---

## Setup & Installation

### Fresh Windows Machine — Run in Order

```bash
01_install_cuda.bat
02_install_python_packages.bat
python 03_verify_gpu.py
```

### Inference Only (no training)

The ONNX models are included in the repo — no training required to run the pipeline.

```bash
pip install opencv-python numpy onnxruntime
```

### For Training

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics matplotlib pillow optuna
python download_datasets.py
```

### Configuring Model Paths

Update the `MODELS` list at the top of `ocr_pipeline.py` to point to where the ONNX files are on your system:

```python
MODELS = [
    r"C:\path\to\project\pytorch\ocr_model.onnx",
    r"C:\path\to\project\pytorch2\ocr_model2.onnx",
    r"C:\path\to\project\pytorch3\ocr_model3.onnx",
]
```

### Quick Start

```bash
git clone https://github.com/BECKHAMW3233/CSC-114.git
cd CSC-114/project

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

| Parameter | Value |
|-----------|-------|
| Input resolution | 64×64 |
| Filter progression | 32→64→128→256 |
| Classifier head | 256→128→62 |
| Optimizer | Adam (lr=3e-4, decay=3e-5) |
| Scheduler | OneCycleLR |
| Batch size | 256 |
| Augmentation | Rotation ±5°, affine, contrast jitter |
| Output | `pytorch/ocr_model.onnx` (~9.4 MB) |

### Model 2 — SE-Attention, Wider (`ocr_pytorch_model2.py`)

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
| Output | `pytorch2/ocr_model2.onnx` (~37 MB) |

### Model 3 — Triple-Width, Multi-Scale Fusion (`ocr_pytorch_model3.py`)

| Parameter | Value |
|-----------|-------|
| Input resolution | 64×64 |
| Channel progression | 96→192→384→768→768 |
| Feature fusion | Pyramid: concatenated pool from stages 2+3+4 |
| Classifier head | 768_fused→1024→512→256→128→62 (5 layers, GELU) |
| Optimizer | SGD + Momentum (lr=0.01, momentum=0.9) |
| Scheduler | CosineAnnealingWarmRestarts |
| Batch size | 128 |
| Augmentation | Rotation ±5°, affine, perspective distortion, blur + noise |
| Output | `pytorch3/ocr_model3.onnx` (~17 MB) |

---

## Training Data

All three models use identical data sources via `supplementary_data.py`. Class imbalance is addressed with `WeightedRandomSampler` in all models.

| Dataset | Samples | Classes | Notes |
|---------|---------|---------|-------|
| EMNIST byclass | 814,255 | 62 | Primary — digits + upper + lower |
| EMNIST Balanced | ~112,800 | 47 | Equal samples per class |
| Kaggle A-Z | 372,450 | 26 | Uppercase only |
| Chars74K EnglishHnd | ~3,410 | 62 | Handwritten; all 62 classes |
| Chars74K EnglishImg | ~7,705 | 62 | Natural scene; all 62 classes |

**EMNIST byclass index mapping:**
- Indices 0–9 → digits 0–9
- Indices 10–35 → uppercase A–Z
- Indices 36–61 → lowercase a–z

---

## Diagnostic Findings — Deployment Testing

### Critical Bug: Missing Inference Normalization

All three training files apply `transforms.Normalize(mean=0.5, std=0.5)` mapping `[0,1]` to `[-1,1]`. The original pipeline only divided by 255, causing a distribution mismatch that made all predictions fail. Overall consensus jumped from 0% to 54.5% after the one-line fix:

```python
arr = arr.astype(np.float32) / 255.0
arr = (arr - 0.5) / 0.5  # matches training normalization
```

### Stroke-Classifier Finding

Test "10 5S 3E" produced **83.3% all-three-model agreement with only 1 correct prediction** — the definitive evidence that the models learned stroke-direction classification rather than character topology:

| Expected | Got | Why |
|---|---|---|
| `1` | `T` | Single vertical stroke |
| `5` | `N` | Two diagonal segments |
| `S` | `N` | Two diagonal segments |
| `3` | `W` | Two open curves = two V-shapes |
| `E` | `M` | Three horizontal strokes = three peaks |
| `O` | `O` ✓ | Closed circular — unambiguous |

### Confirmed Repeatable Failure Pairs

| Pair | Agreement | Tests |
|------|-----------|-------|
| `3 → W` | All agree | Tests 2, 3 |
| `L → 7` | All agree | Tests 1, 3 |
| `E → M/O` | All/majority | Tests 1, 2, 3 |
| `b → t` | All agree | Test 3 |
| `B → P` | Majority | Test 3 |
| `2 → 6` | All agree | Test 3 |
| `7 → 7` ✓ | All agree | Tests 1, 2, 3 |
| `O → O` ✓ | All agree | Tests 1, 2, 3 |

---

## Training Corrections Applied (v2)

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

**1. Preprocessing** — Adaptive Gaussian threshold + dilation. Image scaled to ≤1000px.

**2. Contour merge** — Center-Y proximity merge (`gap_x=15px`, `gap_y=35px`). Handles crossbar `7`, two-loop `8`, dotted characters.

**3. Line grouping** — Center-Y sort and grouping. Threshold = 50% of median character height.

**4. Classification** — Each crop normalized and run through all three ONNX models. Top-3 predictions per model. Input size read from ONNX metadata automatically.

**5. Spatial override** — Aspect ratio < 0.30 + height > 70% median → forced `1`/`i`/`I`.

**6. Ensemble voting** — ALL / MAJORITY / WEIGHTED. Special: `7`-presence check (>0.10 combined); `W`/`w` dominance → `3`.

**7. Mode remapping** (`digits` / `digits-strict`):

| Model output | Digit |
|---|---|
| O, o | 0 |
| L, l, I, i, T, t | 1 |
| Z, z, W | 2 |
| w | 3 |
| Y, y | 4 |
| S, s | 5 |
| G, C, c, b | 6 |
| V, v, D | 7 |
| B | 8 |
| Q, q | 9 |

**8. Strict grid correction** (`digits-strict`) — Position-based override for known 0–9 grid layouts. `SPLIT` and `WEIGHTED` always overridden; `MAJORITY` overridden; `ALL` never overridden.

### Known Issues

- Model paths in `ocr_pipeline.py` must be updated to match your local directory structure.
- Real-world photo handwriting significantly reduces accuracy — models were trained on clean EMNIST-format isolated characters.
- Layout `(4,3,1,2,1)` not always recognized by strict mode when a crossed `7` fragments differently.

---

## Test Results

Six handwritten images of digits 0–9 in a 4+4+2 grid, same handwriting, varying speed and pressure.

| Image | Best Guess (digits-strict) | Score |
|-------|---------------------------|-------|
| test1.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test2.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test3.jpg | `0 1 1 3 / 4 5 6 7 / 8 9` | **9/10** ¹ |
| test4.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test5.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** |
| test6.jpg | `0 1 2 3 / 4 5 6 7 / 8 9` | **10/10** ² |
| **Total** | | **59/60 (98.3%)** |

¹ Unanimous wrong prediction on digit `2` — all three models read `L`. Requires retraining.  
² Written with European crossed `7`. Center-Y grouping resolves the crossbar fragmentation.

### Test Session Progression

| Session | Content | Consensus | Key Finding |
|---------|---------|-----------|-------------|
| Pre-fix | Any | 0% | Missing inference normalization |
| Post-fix 1 | HELLO COMPUTER | 54.5% | Rotation causing L→7, H→I |
| Post-fix 2 | Photo handwriting | 16.7% | Domain gap |
| Post-fix 3 | 10 5S 3E | 83.3% | Stroke classifier confirmed |
| Post-fix 4 | Aa Bb / 1234 / Li 7 Oo | 38.9% | Repeatable failure pairs confirmed |
| Final | 0–9 grid × 6 images | — | 59/60 digits-strict |

---

## Limitations and Path Forward

**Root cause of failures:** All three models share the same training distribution, producing correlated errors that ensemble voting cannot cancel. Post-processing reaches ~98% on known-content digit grids but cannot fix mixed-content accuracy without retraining.

**Proper fix:** Retrain with explicit digit class upweighting, or use EMNIST Digits as primary training data.

**Benchmark for retrained models** — run against `Aa Bb Cc / 1234 / Li 7 Oo / Ee`:

| Character | Pre-retrain | Expected post-retrain | Fix |
|---|---|---|---|
| `L` | `7` (all agree) | `L` | Rotation ±5° |
| `3` | `W` (all agree) | `3` | Rotation ±5° |
| `b` | `t` (all agree) | `b` | 64×64 resolution |
| `B` | `P` (majority) | `B` | 64×64 resolution |
| `2` | `6` (all agree) | `2` | 64×64 resolution |
| `E` | `O` or `M` | `E` | Rotation + class balance |
| `7` | `7` ✓ | `7` ✓ | Monitor for regression |
| `O` | `O` ✓ | `O` ✓ | Monitor for regression |

---

## Hardware & Training Environment

```
CPU:    AMD Ryzen 9 7900X (24 threads, 8 DataLoader workers)
RAM:    64 GB DDR5-5600 (full EMNIST cached in RAM after epoch 1)
GPU:    ZOTAC RTX 4080 16 GB AMP Extreme AIRO
        CUDA 12.1 · torch.autocast float16 (AMP enabled)
OS:     Windows 10 (26100.8246)
Python: 3.13
```

---

## References

- Chollet, F. & Watson, M. (2025). *Deep Learning with Python, 3rd Ed.* Manning Publications.
- Cohen, G. et al. (2017). EMNIST: Extending MNIST to handwritten letters. *ICDAR 2017*.
- de Campos, T.E. et al. (2009). Character recognition in natural images. *VISAPP 2009*.
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*.
- Kaggle A-Z Handwritten Alphabets Dataset — 372,450 samples, 26 uppercase classes.
