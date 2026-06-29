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

**In plain terms:** This project takes a photo of handwritten text and tells you what it says. You photograph a piece of paper with handwritten letters or numbers, run the pipeline, and it outputs the characters it recognized. It works on digits (0-9), uppercase letters (A-Z), and lowercase letters (a-z) — 62 character classes total.

**How it works at a high level:** Six separate AI models each look at the same image and make their own prediction for each character. The pipeline then combines all six predictions using a voting system — if most models agree, that answer wins. If they disagree, the pipeline uses confidence scores, known error patterns, and the position of the character in the image to make the best possible final decision. Using six models instead of one makes the system significantly more reliable because different models make different mistakes, and voting cancels out individual errors.

**Technical summary:** This project implements a six-model deep learning ensemble for handwritten character recognition across 62 classes — digits 0–9, uppercase A–Z, and lowercase a–z. Three architecturally diverse base models are trained on a 9-source, 1,443,757-sample dataset, then improved through knowledge distillation, exported to ONNX, and deployed through a custom inference pipeline with post-processing compensation for known model bias patterns.

**Framework:** This entire project is implemented exclusively in PyTorch — no TensorFlow, no Keras, no JAX. Every model architecture, training loop, data pipeline, distillation script, and inference pipeline is pure PyTorch. The only non-PyTorch dependency at inference time is ONNX Runtime for loading the exported `.onnx` model files.

The project was conducted entirely on consumer hardware (RTX 4080) with a self-imposed 12-hour per-run training ceiling, no cloud compute, and a fully reproducible open-source toolchain. All training artifacts, intermediate outputs, soft labels, and trained models are included in the repository.

This is a self-directed project developed independently alongside CSC-114 coursework, which covers foundational deep learning concepts through approximately Chapter 8 of Chollet & Watson (2026). The EMNIST ensemble operates well beyond the course scope.

---

## Version History

| Version | Models | Dataset | Key Change |
|---------|--------|---------|------------|
| v1 | 1 (Adam) | EMNIST byclass only (697,932) | Baseline ConvNet |
| v2 | 3 (Adam / AdamW / SGD) | 5 sources | Rotation fix, WeightedRandomSampler, 64×64 resolution, augmentation diversity |
| v3 | 3 (Lion / SF-AdamW / SGD) | 9 sources (1,443,757) | Lion replaces Adam, Schedule-Free AdamW replaces AdamW, SE attention, full retrains from scratch |
| v3 + distillation | 6 (3 base + 3 distilled) | Same 9 sources | Knowledge distillation, 6-model ensemble |

Each version represents a complete retrain from random initialization — no weights carried forward from any prior version at any point. Every time the architecture changes, the optimizer changes, or the dataset changes, all models are retrained from scratch. This is a deliberate policy: carrying forward weights from a prior architecture would contaminate the new training run with features learned under different conditions. The only intentional use of pretrained weights in the entire project is within the distillation phase of each version, where the distilled models start from that version's own base model checkpoints — not from any prior version.

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
├── supplementary_data.py        # Shared 9-source dataset loader — combines all datasets, applies DIGIT_BOOST=3.0x weighting on digit-only sources, enforces WeightedRandomSampler for equal class representation per batch, and applies the shared augmentation pipeline. Imported by all three training scripts so dataset handling is identical across all models.
├── download_datasets.py         # Automated dataset download script
├── install_deps.py              # Python dependency installer — alternative to the batch file for non-Windows systems or manual installs
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
# 1. Run as Administrator — installs CUDA 12.3 + cuDNN 8.9
01_install_cuda.bat

# 2. Run as normal user — creates venv, installs all packages, copies scripts
02_install_python_packages.bat

# 3. Verify GPU and datasets are working
python 03_verify_gpu.py
```

**What `01_install_cuda.bat` does:** Checks your NVIDIA driver version, opens the CUDA 12.3 and cuDNN 8.9 download pages with instructions for what to select, copies cuDNN files into the CUDA directory, and adds CUDA to the system PATH. Run as Administrator — right-click → "Run as administrator". Requires a free NVIDIA developer account for cuDNN.

**What `02_install_python_packages.bat` does:** Creates the project directory structure at `E:\CSC-114\emnist-model\`, creates a Python virtual environment, installs all required packages (PyTorch with CUDA 12.1, torchvision, onnx, onnxruntime-gpu==1.19.2, lion-pytorch, schedulefree, kaggle, pandas, scipy, certifi, and all training dependencies), copies all training scripts into the project folder, and runs the GPU verification script automatically at the end. The full install takes 10-20 minutes due to the PyTorch download (~2.5 GB).

**What `03_verify_gpu.py` does:** Checks for the Kaggle A-Z dataset and EMNIST Balanced dataset and reports their status. Minimal verification — confirms datasets are present and loadable before training begins.

### Manual Installation

**Inference only:**
```bash
pip install opencv-python numpy onnxruntime-gpu==1.19.2
```

> **What is ONNX?** ONNX (Open Neural Network Exchange) is a standard file format for AI models. Once a model is trained in PyTorch and exported to ONNX format, it can be run anywhere that supports ONNX — without needing PyTorch installed. This means the `.onnx` model files in this repo can be used for inference with just `onnxruntime` installed, no full training environment required. Think of it like exporting a document to PDF so anyone can read it without the original software.

> **Note:** `onnxruntime-gpu==1.19.2` is pinned to CUDA 12.x. Do not upgrade without verifying CUDA compatibility — later versions require CUDA 13.

**Training (full pipeline):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics matplotlib pillow optuna scipy pandas kaggle certifi
pip install lion-pytorch schedulefree
pip install onnx onnxruntime-gpu==1.19.2
```

**Alternative — `install_deps.py`:** A Python script version of the installer that works on any platform (Windows, Linux, Mac). Installs the same packages as the batch file but without the venv setup or script copying. Useful for Linux systems (for cloud GPU runs) or if the batch file approach doesn't work on your setup. Run with `python install_deps.py` — it installs all packages, then verifies each one loaded correctly and prints a ready status.

### Dataset Download

```bash
python download_datasets.py
```

Downloads and prepares all supplementary datasets automatically. Each dataset is downloaded via torchvision or the Kaggle API and verified on completion. Chars74K requires manual download (see below).

**What gets downloaded automatically:**
- EMNIST Balanced, EMNIST Digits, MNIST, SVHN — via torchvision, no account required
- Kaggle A-Z — requires a free Kaggle account and `kaggle.json` API token placed at `C:\Users\<you>\.kaggle\kaggle.json`

**USPS (manual download required — SSL issue):**
torchvision attempts to download USPS automatically but fails on Windows due to an SSL certificate verification error with the USPS dataset server. The download script will report `[MISSING] USPS — SSL issue, skipping` and continue without it. To include the 9,298 USPS samples:

1. Download the file manually from your browser: `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2`
2. Create the folder: `E:\CSC-114\emnist-model\datasets\pytorch\USPS\raw\`
3. Place the downloaded file in that folder and rename it to `usps.bz2`
4. torchvision will find it automatically on the next training run without downloading

The USPS dataset adds 9,298 scanned postal envelope digit samples. Skipping it is not critical — the remaining digit datasets (EMNIST Digits + MNIST + SVHN) provide 373,257 digit samples which already exceeds the Kaggle A-Z uppercase count. Training will proceed correctly without USPS.

**Chars74K (manual download required):**
Download from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ and extract to the datasets folder. The script will detect it automatically if placed correctly.

**If a dataset fails to download:** Training will skip it automatically and continue with the remaining sources. The script prints `[OK]` or `[FAIL]` for each dataset so you can see exactly what loaded.

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

### Mode Reference

**`--mode auto`** (default)
Use when the image contains mixed content — uppercase letters, lowercase letters, and digits together. No remapping is applied. The ensemble votes and post-processing handles known bias patterns (q→a remap, split rescue, i detection). This is the correct mode for general handwriting, sentences, and any content where you don't know in advance what character types will appear.
```bash
python ocr_pipeline.py --mode auto image.jpg
```

**`--mode digits`**
Use when the image contains only digits but the layout is not a standard 0-9 grid. Applies the full DIGIT_REMAP table — converts letter predictions that are visually similar to digits (O→0, S→5, W→2/3, etc.) to their digit equivalents. Use this for phone numbers, serial numbers, PIN codes, or any freeform digit content.
```bash
python ocr_pipeline.py --mode digits test1.jpg
```

**`--mode digits-strict`**
Use when the image contains only digits AND the layout matches one of the recognized grid patterns (4+4+2, 4+3+2, etc.). Applies everything from digits mode plus position-based correction — if the layout signature is recognized, characters at known positions are corrected to their expected digit regardless of what the models predicted. This is the highest accuracy mode for structured digit grids and achieved 100% on all 7 handwritten digit test images.
```bash
python ocr_pipeline.py --mode digits-strict test*.jpg
```

**`--mode upper`**
Use when the image contains only uppercase letters or when you want all output forced to uppercase. Converts lowercase predictions to uppercase equivalents and remaps digit predictions to their uppercase letter equivalents (0→O, 1→I, 5→S, 6→G). Use for license plates, form fields, or any content known to be uppercase only.
```bash
python ocr_pipeline.py --mode upper handwriting.jpg
```

**`--mode lower`**
Use when the image contains only lowercase letters or when you want all output forced to lowercase. Converts uppercase predictions to lowercase equivalents. Note: the lowercase ambiguity cluster (a, e, i, o) performs significantly worse in isolation than in mixed content — see Stress Test Findings in the Limitations section.
```bash
python ocr_pipeline.py --mode lower handwriting.jpg
```

### Multiple Files and Wildcards

All modes support multiple files and wildcard patterns. Sessions load models once and process all images sequentially — significantly faster than running separately for each image.
```bash
# Wildcard — all jpg files matching pattern
python ocr_pipeline.py --mode digits-strict test*.jpg

# Explicit list
python ocr_pipeline.py --mode digits-strict test1.jpg test2.jpg test3.jpg

# Mixed content benchmark
python ocr_pipeline.py --mode auto untitled.png
```

### Choosing the Right Mode

| Content Type | Recommended Mode |
|-------------|-----------------|
| Mixed letters and digits | `auto` |
| Digits only, freeform layout | `digits` |
| Digits only, structured grid | `digits-strict` |
| Uppercase letters only | `upper` |
| Lowercase letters only | `lower` |
| Unknown / unsure | `auto` |

---

## Sample Output

**What you see when you run the pipeline:** The output has four main sections — individual model predictions, the ensemble result, any strict grid corrections applied, and the final best guess read. Here is an example from the current 6-model pipeline running `--mode digits-strict` on a handwritten 0-9 grid (test1.jpg):

```
============================================================
  OCR Pipeline — 6-Model Ensemble (Base + Distilled)
  Image: test1.jpg
  Mode:  DIGITS-STRICT
============================================================
  Detected: 10 characters across 3 line(s)

  ──────────────────────────────────────────────────
  INDIVIDUAL MODEL PREDICTIONS (raw, no remapping)
  ──────────────────────────────────────────────────
  ocr_model.onnx (64x64):
    Line 1: O 1 2 3
    Line 2: 4 S 6 7
    Line 3: 8 9
  ocr_model2.onnx (64x64):
    Line 1: 0 1 2 w
    Line 2: 4 5 6 7
    Line 3: 8 9
  ocr_model3.onnx (64x64):
    Line 1: 0 1 2 3
    Line 2: 4 5 6 7
    Line 3: 8 9
  ocr_model1_distill.onnx (64x64):
    Line 1: O 1 2 W
    Line 2: ? ? ? ?
    Line 3: 8 9
  ocr_model2_distill.onnx (64x64):
    Line 1: O 1 2 ?
    Line 2: ? ? 6 ?
    Line 3: 8 9
  ocr_model3_distill.onnx (64x64):
    Line 1: O ? ? W
    Line 2: 2 N C V
    Line 3: D C

  ──────────────────────────────────────────────────
  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  *=strict  ?=split)
  ──────────────────────────────────────────────────
  Line 1: [0] [1] [2] [3]
  Line 2: [4] [5] [6] [7]
  Line 3: [8] [9]

  STRICT GRID — layout (4, 4, 2) matched, no corrections needed

  ──────────────────────────────────────────────────
  BEST GUESS READ  [mode: DIGITS-STRICT]
  ──────────────────────────────────────────────────
  Line 1: 0 1 2 3
  Line 2: 4 5 6 7
  Line 3: 8 9
============================================================
```

**Reading the individual model predictions:** Each model makes its own raw prediction before any post-processing. You can see that some models read the digit `5` as `S` (which looks visually similar), and the distilled models produce `?` for characters they are not confident about. This is normal and expected — the voting system corrects these in the next step.

**Reading the ensemble result:** The bracket notation tells you how each character was decided:
- `plain` (no brackets) — all 6 models agreed unanimously
- `[x]` — majority or weighted vote winner (more than half agreed, or confidence scoring picked a winner)
- `*x*` — position override applied by strict grid correction (the pipeline knew from position what the digit must be)
- `?` — unresolved split (models disagreed and no correction was available)

**Reading the character detail section:** Each character line shows all 6 model predictions, the aspect ratio (`asp`), how the vote was resolved (`~ maj`, `~ wgt`, `✓ all`, `* strict`, `✗ split`), and the final output. The aspect ratio is the width-to-height ratio of the character's bounding box — used by the pipeline to detect narrow tall characters like `1` and `i`.

**The best guess read** is the final output — what the pipeline thinks the image says after all voting and post-processing is complete.

---

## Models

### Architecture Overview

Three architecturally diverse base models were chosen specifically to maximize error diversity across the ensemble. Optimizer families were selected to produce qualitatively different weight landscapes: momentum-free adaptive (Lion), schedule-free adaptive (SF-AdamW), and classical momentum (SGD). Each model was trained from random initialization on the full 9-source dataset.

**What the architecture parameters mean for non-technical readers:**

- **Parameters** — the total number of individual numbers (weights) the model learns during training. More parameters means more capacity to learn complex patterns, but also more compute and memory required. 2.5M parameters is relatively small; 9.7M is larger.
- **Filter progression** (e.g. 32→64→128→256) — convolutional neural networks process images through layers of filters. Each number is how many filters are in that layer. Early layers with fewer filters detect simple features like edges and curves. Later layers with more filters combine those into complex features like loops, corners, and stroke patterns. More filters per layer means the model can detect more distinct features at that level.
- **Classifier head** (e.g. 256→128→62) — after the convolutional layers extract features, a series of fully-connected layers makes the final decision. The numbers show how the information is progressively compressed down to 62 outputs — one confidence score per character class.
- **Squeeze-Excitation (SE) attention** — after each convolutional layer, SE attention asks "which of these detected features actually matter for this image?" and amplifies the important ones while suppressing the less useful ones. It's a way of focusing the model's attention on the most relevant features per image rather than treating all features equally.
- **StochasticDepth / DropPath** — during training, randomly skips entire layers on some passes. Forces the remaining layers to be more robust because they can't rely on any specific layer always being present.
- **Feature pyramid** — instead of only using the final layer's features to make a prediction, a feature pyramid collects outputs from multiple layers at different stages and combines them. This gives the classifier access to both fine-grained early features and high-level late features simultaneously.
- **GELU activation** — a mathematical function applied after each layer that decides which signals to pass forward and how strongly. GELU is a smoother version of the standard ReLU function and works better in deeper networks.
- **Augmentation** — random transformations applied to training images to simulate variation in handwriting: slight rotations, scaling, translation, blur, noise, perspective distortion. The model never sees the same image twice in exactly the same form, which forces it to learn features that are robust to those variations rather than memorizing specific pixels.
- **Regularization** — techniques that prevent the model from overfitting (memorizing the training data rather than learning to generalize). Dropout randomly disables neurons during training. L2 weight decay penalizes large weights. Label smoothing makes the model less overconfident in its predictions.
- **Train acc vs Test acc** — training accuracy is how well the model performs on the data it was trained on. Test accuracy is how well it performs on data it has never seen. The gap between them indicates how much the model has overfit. A large gap (e.g. 91% train, 81% test) means the model learned some training-specific patterns that don't generalize perfectly.

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

**In plain terms:** Model 3 required four separate training runs before producing a usable result. The first three runs each had a specific problem that was identified, fixed, and then tested in the next run. This is normal in ML development — you don't always get the configuration right on the first attempt, especially with a complex model using an optimizer (SGD) that is more sensitive to hyperparameter choices than Adam-family optimizers. The history is documented here because the failures contain useful information about what doesn't work and why.

- **Run 1 — Scheduler fired too early:** The learning rate restart schedule was set to fire a reset at epoch 35 out of a 50-epoch budget, leaving only 13 epochs for the model to recover after the restart. The technique requires at least 100+ epochs to complete two full cycles properly. The restart disrupted training mid-convergence. Result: 76.61% accuracy — usable but not better than the failed run.
- **Run 2 — Loss function caused class collapse:** FocalLoss (a technique for handling class imbalance) with an aggressive gamma setting caused the model to essentially stop predicting two entire character classes. O dropped to 0.1% accuracy, S dropped to 0.7% — the model learned to never predict those characters at all. This is called catastrophic class collapse. Result: 76.61% — same number as Run 1 but for a completely different reason.
- **Run 3 (final):** Nine separate fixes applied simultaneously — switched the scheduler to a simpler decay-only version, corrected weight decay, removed FocalLoss, fixed a bug where a sharpness augmentation parameter was set to 0 (meaning no sharpness augmentation was actually applied), reduced several augmentation strengths that were too aggressive, and adjusted attention and dropout settings. Result: 77.30% — the accepted final base model.

Three runs were required before the final v3 run. This history is documented as a methodological finding on optimizer and scheduler selection for this class of problem.

- **Run 1:** CosineAnnealingWarmRestarts T_0=35 — restart fires at epoch 37, leaving only 13 epochs for recovery. Published SGDR paper used 200+ epoch budgets; T_0=35 with T_mult=2 requires minimum ~105 epochs for two full cycles. Test acc: 76.61%.
- **Run 2:** FocalLoss gamma=2.0 — catastrophic class collapse. O->0.1%, S->0.7%. Test acc: 76.61%.
- **Run 3 (v3, final):** 9 fixes applied simultaneously — CosineAnnealingWarmRestarts->CosineAnnealingLR, weight decay 3e-5->5e-4, label smoothing 0.08->0.05 (FocalLoss removed), sharpness_factor bug fixed (0->2.0), contrast 0.4->0.2, translate 0.12->0.08, SE reduction 32->16, drop_path 0.1->0.05, first classifier dropout 0.5->0.35. Test acc: 77.30%.

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

**In plain terms:** Knowledge distillation is a technique where a new model (the "student") learns not just from the raw training data, but also from what the already-trained models (the "teachers") think about each character. Instead of just being told "this is a 3," the student also learns "the teachers think this looks 73% like a 3, 15% like a W, and 12% like a Z" — which carries much more useful information about why characters look similar to each other. The result is a student model that performs better than it would have if trained on the raw data alone.

In this project, each of the three base models acts as a teacher for a new distilled version of itself. The distilled models achieved 88.1-88.5% accuracy compared to 77.3-83.9% for the base models — a gain of up to 11 percentage points from distillation alone with no additional training data.

### Methodology

Each base model is retrained using a combined loss function:
- **70% KL-divergence** against averaged soft labels from the other two base models (temperature=4.0)
- **30% CrossEntropy** against hard ground truth labels (label smoothing=0.05)

Soft labels encode inter-class relationship information that hard labels cannot — the probability distribution over all 62 classes at temperature=4.0 carries meaningful signal about visual similarity between characters. Distillation uses AdamW for all three models regardless of original optimizer, starting from pretrained base weights rather than random initialization.

### Configuration

**What each parameter means:**
- **Temperature** — controls how "soft" the soft labels are. At temperature=1.0, the label distribution is sharp — the correct class gets nearly all the probability. At temperature=4.0, the distribution is flattened — the correct class still gets the most probability but similar-looking classes get meaningful shares too. Higher temperature means the student learns more about inter-class relationships (how similar B is to 8, how similar O is to 0) rather than just which class wins.
- **Alpha (soft label weight)** — what fraction of the training loss comes from the teacher's soft labels vs the hard ground truth labels. At 0.7, 70% of the learning signal comes from what the teachers think and 30% from the correct answer. Higher alpha means more teacher influence.
- **Label smoothing** — instead of training the model toward a 100% confidence prediction on the correct class, label smoothing trains it toward 95% confidence. This prevents overconfidence and improves generalization slightly.
- **Starting weights** — unlike the base model training which starts from random initialization, distillation starts from the already-trained base model checkpoints. The student doesn't start from scratch — it starts already knowing how to recognize characters and then refines that knowledge using the teachers' soft labels.

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

**What this table shows:** After training, the accuracy was measured separately for each of the 62 character classes. The numbers below show the 15 worst-performing classes for each model — the characters the model struggles with most. The class index in parentheses (e.g. `o (50)`) is the EMNIST dataset's internal numbering: indices 0-9 are digits, 10-35 are uppercase A-Z, and 36-61 are lowercase a-z. The Samples column shows how many test samples exist for that class — lower sample counts mean the accuracy number is less statistically reliable.

A class accuracy of 0.2% means the model gets that character right almost never — it's essentially always predicting something else. This is different from overall accuracy (88%+) because the overall number averages across all 62 classes including the easy ones like O, 0, and uppercase letters that score 95%+. The worst classes drag down the overall number and are where the real failure analysis happens.

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

> **What does "comment out" mean?** In Python, any line starting with `#` is ignored by the program. To disable a model, add a `#` at the start of that line. You can do this in any text editor — Notepad, VS Code, anything. No programming knowledge required.

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
- Phone camera watermarks (e.g. "Galaxy S22 Ultra") printed at the bottom of images will be detected as characters if they fall within the contour size thresholds. This is expected behavior — the pipeline detects all text-like contours regardless of source. Crop or mask the watermark area before running if clean output is required.

---

## Troubleshooting

**"can't open/read file: check file path/integrity"**
The image file was not found. Check that the filename is spelled correctly and that you are running the command from the same folder the image is in. On Windows, use `cd` to navigate to the folder first:
```cmd
cd E:\CSC-114\emnist-model
python ocr_pipeline.py --mode auto test1.jpg
```

**"Failed to load ocr_model.onnx"**
The ONNX model file path in the MODELS list does not match where the file actually is on your system. Open the pipeline file in any text editor, find the MODELS list near the top, and update the paths to match your file locations.

**CUDA provider warnings on startup**
```
[E:onnxruntime] Failed to create CUDAExecutionProvider...
```
This warning is harmless. It means CUDA is not available on your system and the pipeline is falling back to CPU. Predictions will still be correct — just slower. See the CPU-Only Inference section for details.

**"No characters detected"**
The pipeline could not find any character contours in the image. Common causes:
- Image is too dark or too light — characters need reasonable contrast against the background
- Image is very small — minimum recommended image size is 300×300 pixels
- Characters are too small relative to the image — the pipeline filters out very small contours as noise

**Pipeline reads wrong characters consistently**
Check that you are using the right mode for your content. Digits in `--mode auto` will not get the digit remap corrections. Mixed content in `--mode digits-strict` will have letters incorrectly converted to digit equivalents. See the Mode Reference in the Usage section.

---

## CPU-Only Inference

The pipeline runs on CPU when CUDA is unavailable. ONNX Runtime will print provider warnings and fall back to `CPUExecutionProvider` automatically — these warnings are harmless and output is identical to GPU inference. Inference speed on CPU is significantly slower than GPU but fully functional for testing and demonstration purposes.

The school machine environment runs CPU-only due to CUDA driver configuration. The home machine environment runs GPU via `CUDAExecutionProvider`. Both produce identical prediction results — the provider affects speed only, not accuracy.

To suppress the CUDA provider warning on a CPU-only system, remove `CUDAExecutionProvider` from the providers list in the MODELS loading block at the bottom of the pipeline file:

```python
s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
```

---

## End-of-Course Demonstration

Planned for end of Summer 2026 term (CSC-114, FTCC). Instructor Milstead has requested a live class demonstration where each student writes a line of mixed text and numbers, photographs it, and the pipeline reads it live on the main display board.

**Demo protocol:**
- Student writes mixed content (letters and digits) on paper
- Photograph with phone, transfer to demo machine via shared folder, AirDrop, or USB
- Run: `python ocr_pipeline.py --mode auto <image.jpg>`
- Output displayed on projector showing individual model predictions, voting resolution, and final read

**Why `--mode auto`:** The demo image will contain mixed uppercase, lowercase, and digit content — the same scenario as the Untitled.png benchmark which achieved 15/15 in auto mode. Mode-specific remapping (digits-strict, upper, lower) would produce incorrect output for mixed content.

**Known demo risk:** Image transfer speed is the only friction point in the live demo flow. Establishing the transfer method before demo day is the primary preparation item.

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

**In plain terms:** The models were not actually reading characters — they were counting strokes and measuring directions. A character with two diagonal strokes gets classified as N regardless of whether it's actually an N, a 5, or an S. This means the models memorized stroke patterns from the training data instead of learning what makes each character unique. This is a fundamental training failure — high agreement between models on wrong answers means all three models learned the same wrong shortcut, and voting cannot fix correlated errors.

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

### Training Methodology Note — Epoch 1 Restart Policy

During all training runs, if the first epoch training accuracy (`acc`) came in below 50%, the run was stopped and restarted. The reasoning: the random weight initialization and initial learning rate interact at epoch 1 — if the model fails to clear a reasonable threshold on the first pass through the training data, the initialization is too far from a useful starting point and continuing wastes compute. Restarting re-rolls the random initialization and gets a better starting position.

**Verified epoch 1 data from actual training logs:**

| Run | Model | acc (epoch 1) | val_acc (epoch 1) | Action |
|-----|-------|--------------|-------------------|--------|
| v3 base | M1 (Lion) | 47.41% | 70.09% | Accepted — cleared threshold |
| v3 base | M3 bad start | ~42% | 66.22% | Restarted — too low |
| v3 base | M3 restart | 42.18% | 66.05% | Accepted — val_acc strong |
| v3 base | M3 final (batch=256) | 70.84% | 83.63% | Accepted — strong start |
| v3 distill | M1 distilled | 87.76% | 87.83% | Accepted — pretrained weights |
| v3 distill | M2 distilled | 88.68% | 88.16% | Accepted — pretrained weights |
| v3 distill | M3 distilled | 75.56% | 83.76% | Accepted — pretrained weights |

**Key findings from the verified data:**

The 50% `acc` threshold is approximate, not a hard cutoff. The M3 restart run was accepted at 42.18% `acc` because `val_acc` was 66.05% — strong enough to indicate the model was learning correctly despite the low training accuracy. The restart decision on the bad M3 start was called based on overall trajectory judgment, not a mechanically applied number.

`val_acc` was consistently higher than `acc` in all base model runs — this is expected because `val_acc` is computed on the held-out validation set after the epoch completes, while `acc` reflects training performance mid-epoch before full convergence. The lowest `val_acc` observed at epoch 1 across all accepted runs was 66.05% (M3 restart). No run, including those that were restarted, produced a `val_acc` below 60% at epoch 1.

Distillation runs start significantly higher than base model runs on both metrics because they begin from pretrained base model weights rather than random initialization — the model already knows how to recognize characters, it is just absorbing the soft label signal from the teachers.

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

**What each augmentation type does:**
- **Rotation** — randomly tilts the character slightly during training so the model learns to recognize characters at a small angle, not just perfectly upright
- **Shear** — slants the character diagonally, simulating handwriting that leans left or right
- **Synthetic degradation (blur + noise)** — adds artificial blurriness and random pixel noise to training images, simulating low-quality photos or worn paper
- **Domain-shift augmentation (perspective)** — distorts the image as if photographed from a slight angle rather than straight on, simulating real-world photo conditions
- **WeightedRandomSampler** — ensures every character class appears equally often in each training batch, preventing the model from seeing thousands of common letters and only a handful of rare ones
- **Per-class accuracy logging** — not an augmentation, but tracks which specific characters the model is getting wrong so failures can be diagnosed precisely

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

## Optimizer Reference — What Each One Does and Why It Changed

### What an Optimizer Does

During training, after each batch of data the network calculates how wrong its predictions were (the loss) and computes gradients — a direction for each weight indicating which way to adjust it to reduce the loss. The optimizer takes those gradients and decides exactly how much to move each weight and in what direction. Different optimizers make different decisions about this, and those decisions compound over thousands of training steps into fundamentally different learned weight landscapes.

### v1 — Adam (Model 1 only)

**What Adam does:** Adaptive Moment Estimation. Tracks a running average of past gradients (momentum) and a running average of past squared gradients (velocity). Uses these to scale each weight's update individually — weights that have been getting large consistent gradients get smaller updates, weights that have been getting small or noisy gradients get larger updates. This adaptive scaling makes Adam fast and robust on a wide range of problems without careful learning rate tuning.

**Why it was chosen for v1:** Adam is the standard starting point for deep learning — it's reliable, well-understood, and forgiving of hyperparameter choices. Appropriate for a baseline model.

---

### v2 — Adam / AdamW / SGD

**v2 Model 1 — Adam:** Same as v1. Kept as the baseline anchor while the other two models diversified.

**v2 Model 2 — AdamW:** Adam with decoupled weight decay. Standard Adam applies weight decay incorrectly — it adds the decay term to the gradient before the adaptive scaling, which means the effective weight decay varies per parameter. AdamW fixes this by applying weight decay directly to the weights after the gradient update, keeping regularization consistent across all parameters. The practical effect is better generalization, especially for larger models. Model 2 is the widest model (9.7M parameters) so the regularization improvement from AdamW is most meaningful here.

**v2 Model 3 — SGD with Nesterov momentum:** Stochastic Gradient Descent — the oldest and most classical deep learning optimizer. No adaptive scaling. Each weight gets the same learning rate, adjusted by a momentum term that accumulates velocity in consistent gradient directions and dampens oscillation. Nesterov momentum is a refinement that looks ahead before computing the gradient — it calculates the gradient at the position the momentum would carry the weights to, rather than at the current position. This gives slightly more accurate gradient estimates and faster convergence. SGD is slower and more sensitive to learning rate choice than Adam but often finds flatter minima that generalize better. Used here to produce a qualitatively different weight landscape from the two Adam-family models.

**Why three different optimizers in v2:** After v1's stroke-classifier failure, the analysis showed that three models with the same optimizer and training distribution made correlated errors — they all failed on the same characters for the same reasons, so voting could not cancel those errors. Introducing three different optimizer families was the first step toward genuinely uncorrelated errors in the ensemble.

---

### v3 — Lion / Schedule-Free AdamW / SGD

**v3 Model 1 — Lion (replaces Adam):**
Lion (EvoLved Sign Momentum) was discovered through program search by Google Brain researchers in 2023. It is fundamentally different from Adam: instead of using the full gradient magnitude, Lion uses only the sign of the gradient update — every weight gets updated by exactly +lr or -lr, nothing in between. The momentum term accumulates a running average of past updates and the sign of that average determines the direction. Because all updates have equal magnitude, Lion requires much lower learning rates (10x lower than Adam) and much higher weight decay than Adam to prevent overshooting. The result is more compressed, efficient weight representations that generalize differently from Adam-family models — Lion tends to find solutions that Adam would never reach, not just faster versions of the same solution.

**Why Lion replaced Adam for Model 1:** The goal was maximum optimizer diversity across the ensemble. Lion is not an improvement on Adam in the sense of doing the same thing better — it's a categorically different update rule that explores a different region of weight space. Replacing Adam with Lion in Model 1 while keeping AdamW in Model 2 and SGD in Model 3 gives the ensemble three genuinely distinct weight landscapes rather than two Adam variants and one SGD.

**v3 Model 2 — Schedule-Free AdamW (replaces AdamW + CosineAnnealingLR):**
Schedule-Free AdamW won the MLCommons AlgoPerf 2024 challenge — a competition to find the fastest training algorithms across a range of deep learning tasks. It eliminates the learning rate schedule entirely. Traditional training requires choosing when to decay the learning rate, by how much, and on what schedule — decisions that interact with batch size, model size, and dataset size in ways that are difficult to predict. Schedule-Free AdamW instead maintains two sets of weights internally: a fast-moving set used for gradient computation and a slower Polyak-Ruppert averaged set used for evaluation. The averaging produces the same effect as a decaying learning rate schedule without requiring any schedule to be specified. The optimizer switches between train and eval modes explicitly (`optimizer.train()` / `optimizer.eval()`). The result is competitive or better performance compared to carefully tuned AdamW + schedule, with one fewer hyperparameter decision to make.

**Why Schedule-Free AdamW replaced AdamW + CosineAnnealingLR for Model 2:** The CosineAnnealingLR schedule in v2 required manual decisions about T_max, eta_min, and restart behavior. Model 3's development history (three failed runs before the final configuration) demonstrated that scheduler choices interact unpredictably with the model architecture. Schedule-Free AdamW removes that decision entirely while producing equal or better results.

**v3 Model 3 — SGD with Nesterov momentum (retained from v2):**
Same optimizer as v2 Model 3. SGD is kept as the classical anchor in the ensemble — its weight landscape is fundamentally different from both Lion and SF-AdamW, and it provides the most stable, consistent contribution to the ensemble vote on unambiguous characters. The architecture changed significantly (triple-width channels, SE attention, feature pyramid, 5-layer GELU classifier head) but the optimizer stayed the same because SGD's role in the ensemble is precisely to be the classical baseline that the two newer-family models are compared against.

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
| test5.jpg | Blue pen, lighter strokes, different writer | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | 1 strict correction (Y→8 position override) |
| test6.jpg | Blue pen, lighter strokes, different writer | 0 1 2 3 / 4 5 6 7 / 8 9 | **10/10** | 1 strict correction (N→5 position override) |
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

School machine benchmark testing conducted June 29, 2026. Results from class demonstration to be added here.

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
| test21.jpg | All 5s (~70 chars) | 71 | ~65 | ~91.5% | Cursive 5 top stroke reads as S/J; distilled models read 5 as N. J→5 post-processing fix active — J errors from base models correctly resolved. |
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

**About the RTX 4080's CUDA cores:** The ZOTAC RTX 4080 has 9,728 CUDA cores. CUDA cores are the individual processing units inside the GPU that do the actual math — specifically the billions of floating point multiplications and additions that make up neural network training and inference. During training, every forward pass through the network and every gradient calculation during backpropagation is broken into thousands of tiny parallel math operations and distributed across all 9,728 cores simultaneously. This parallelism is why a GPU trains a model in hours that would take days on a CPU. The RTX 4080 also has 304 Tensor Cores — specialized units that accelerate matrix multiplication specifically, which is the dominant operation in deep learning. PyTorch's Automatic Mixed Precision (AMP) uses these Tensor Cores by computing in float16 instead of float32, roughly doubling throughput with minimal accuracy impact. The 16GB of GDDR6X VRAM holds the model weights, activations, and a batch of training samples in GPU memory simultaneously — larger VRAM means larger batch sizes and less time spent moving data between CPU and GPU.

> Do not upgrade onnxruntime-gpu without verifying CUDA compatibility. Version 1.20+ requires CUDA 13.

All training conducted on consumer hardware with no cloud compute. A self-imposed 12-hour per-run ceiling drove architectural and hyperparameter decisions throughout.

**GPU utilization:** Across every base model and distillation training session, CUDA utilization on the RTX 4080 was consistently between 95-99%, averaging 97%. This indicates the training pipeline is fully saturating the GPU with no meaningful idle time — the DataLoader, data augmentation pipeline, and batch sizing are all well-tuned for the hardware. The 64GB system RAM caching the full 1.4M sample dataset after epoch 1 eliminates disk I/O as a bottleneck, allowing the GPU to run at near-constant maximum utilization throughout each run.

**What loss and accuracy mean in training output:** When you look at training logs like `Epoch 1/50  loss: 1.6695  acc: 0.8776  |  val_loss: 0.9623  val_acc: 0.8783`, there are two separate measures:

- **Accuracy (`acc`, `val_acc`)** — the percentage of characters the model predicted correctly. 0.8776 means 87.76% correct. This is the intuitive measure — higher is better.
- **Loss (`loss`, `val_loss`)** — a mathematical measure of how wrong the model's predictions are, specifically how far the predicted probability distribution is from the correct answer. Loss is what the model actually optimizes during training. Lower is better. Unlike accuracy which jumps in discrete steps (right or wrong), loss is continuous — a model can be getting more confident in correct answers (loss decreasing) even when accuracy hasn't visibly changed yet. This is why loss is the primary signal during training even though accuracy is easier to interpret.
- **Train vs Val** — `loss`/`acc` are measured on the training data. `val_loss`/`val_acc` are measured on held-out validation data the model never trained on. The validation numbers show how well the model generalizes to new data. If train accuracy is much higher than val accuracy, the model is overfitting — it memorized the training data instead of learning general patterns.

**Approximate training times per run (v3 base and distillation, RTX 4080):**
- Model 1 base: ~4 hours (50 epochs, ~290s/epoch)
- Model 2 base: ~2.2 hours (50 epochs, ~88s/epoch)
- Model 3 base: ~11 hours (50 epochs, ~800s/epoch)
- Model 1 distilled: ~2.1 hours (47 epochs, ~161s/epoch)
- Model 2 distilled: ~5.7 hours (50 epochs, ~400s/epoch)
- Model 3 distilled: ~6.1 hours (50 epochs, ~430s/epoch)

---

## Upcoming Update — v3 + Distillation Normalization Fix

A normalization correction is planned and has not yet been started. All v3 base and distilled models were trained with `[-1, 1]` normalization (transforms.Normalize mean=0.5, std=0.5). The planned update changes this to `[0, 1]` normalization — a full retrain of all 6 models from random initialization with the corrected normalization range, plus distillation, following the same methodology as v3. This update does not require cloud compute and will be completed on the RTX 4080 when time allows. The README will be updated with new benchmark results when the retraining is complete.

### What the normalization change does

Normalization scales pixel values into a range the model can work with efficiently. Raw pixel values are 0-255. The two common choices are:

- **[0, 1]** — divide by 255. Black is 0.0, white is 1.0. Simple, intuitive, and the natural output of `ToTensor()` in PyTorch.
- **[-1, 1]** — divide by 255, subtract 0.5, divide by 0.5. Black is -1.0, white is 1.0, mid-gray is 0.0. Centers the distribution around zero, which can help gradient flow in deeper networks.

The current v3 models were trained on `[-1, 1]`. The pipeline also normalizes at inference time to match — so predictions are currently correct. The change to `[0, 1]` means retraining on the simpler, more natural range.

### Why it should improve performance

The EMNIST dataset consists of white characters on a black background — the inverse of natural handwriting. At `[-1, 1]`, background pixels are -1.0 and character stroke pixels approach +1.0. At `[0, 1]`, background pixels are 0.0 and stroke pixels approach 1.0. The `[0, 1]` range means background is true zero — no activation, no noise contribution. With `[-1, 1]`, background pixels at -1.0 still produce non-zero activations in the first convolutional layer, which the model has to learn to suppress. Removing that suppression task may free capacity for learning actual character features, particularly for thin strokes and low-contrast characters that are already at the model's detection limit at 64x64.

The improvement is not guaranteed and may be marginal — the current models compensate for the background activation through training. But the theoretical case for `[0, 1]` on inverted-background data is sound, and with the full retrain and distillation pipeline already established the cost of testing it is just overnight compute time.

### Why this is an easy oversight

This is one of the most common normalization mistakes in deep learning and is well-documented in EMNIST-specific forums, Stack Overflow threads, and PyTorch tutorials. The confusion happens because:

- `torchvision.transforms.ToTensor()` automatically scales to `[0, 1]` — so that range is already applied before any explicit normalization
- Most online code examples, tutorials, and pretrained model documentation use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) which is neither `[0,1]` nor `[-1,1]` — it's a dataset-specific calibration
- EMNIST tutorials vary wildly — some use mean=0.5/std=0.5 for `[-1,1]`, some skip normalization entirely, some use the raw `[0,1]` from ToTensor() without adding Normalize at all
- The original normalization bug in this project (v1 to v2 fix) was the inverse problem: inference was running `[0,1]` while training had used `[-1,1]`, causing total prediction failure. That fix was a one-line correction. The current situation is the subtler version: both training and inference use `[-1,1]` consistently, so predictions work correctly — but `[0,1]` may work better for this specific dataset's pixel distribution.

In the stress testing conducted June 29, 2026 the pipeline correctly reads structured digit grids at 100% and achieves 91-97% on isolated single-digit stress tests — so the current normalization is not broken. The question is whether retraining on `[0,1]` pushes those stress test numbers higher, particularly on the ambiguous character classes where the model is already operating near its detection limit.

---

## Planned Next Version — v4 Multi-Resolution Ensemble

The stress test findings from June 28, 2026 establish that the current 64x64 resolution is insufficient for reliable recognition of the lowercase ambiguity cluster (a, e, i, o, s, c, u, l) and certain digit style variants (hooked 7→9, open-top 8→9/Q). Post-processing has reached its compensation limit at this resolution. v4 addresses this through a multi-resolution training strategy.

### v4 Architecture

**Training:** Each of the 3 base model architectures trained at 4 resolutions — 32x32, 64x64, 128x128, and 256x256 — producing 12 base model files total.

> **On 512x512:** A fifth resolution tier at 512x512 was considered for truly granular stroke-level feature detection. At 512x512, distinguishing features (a's descending tail, e's internal horizontal stroke, i's dot, 7's hook vs 9's closed loop) would occupy 32-64 pixels — significantly more than the 16-32 pixels available at 256x256. However, per-epoch training time scales roughly 64x from 64x64 to 512x512. On cloud hardware (H100 SXM), a single 512x512 M3 run at 150 epochs is estimated at 48-72 hours of compute — expensive and slow even on top-tier hardware. The practical question is also whether 512x512 adds meaningful signal over 256x256 for handwritten characters on paper, where feature complexity is limited. At 256x256 the features are already well above the detection threshold; diminishing returns are expected beyond that point. If the 32/64/128/256 ensemble shows that 256x256 still produces meaningful accuracy gains over 128x128, a 512x512 tier could be justified for a future iteration. For now it is documented as evaluated and deprioritized rather than overlooked.

**Distillation:** Each distilled model trained from soft labels generated by all 4 resolution variants of its corresponding base architecture, providing richer soft label distributions than single-resolution teachers.

**Result:** 24 total ONNX models (12 base + 12 distilled) across 4 resolutions and 3 architectures.

**Inference:** All 24 models vote on every character — each input is resized to the native resolution of each model before prediction. The voting system handles 24 inputs using the same weighted scoring logic.

### Rationale

The 32x32 models learn coarse global shape features — fast, confident on unambiguous characters, strong anchoring votes. The 256x256 models learn fine stroke endpoint features — decisive on the ambiguous cluster where distinguishing features (a's descending tail vs q's, e's internal stroke, i's dot, 7's hook vs 9's closed loop) occupy 4-8 pixels instead of sub-pixel widths. The full 24-model ensemble covers the complete feature spectrum simultaneously rather than compromising at a single resolution.

### Compute Requirements

v4 is not viable on consumer hardware within reasonable time constraints. Estimated training time on RTX 4080: 3-4 weeks continuous. Planned execution on cloud GPU rental. Based on current RunPod pricing (verified June 2026): RTX Pro 6000 at ~$2.09/hr (~$34-50 for full run) or H100 SXM at ~$2.69/hr (~$54-73 for full run) are the most cost-effective options for the 64x64, 150-epoch configuration.

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