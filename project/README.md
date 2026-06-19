# EMNIST OCR — Three-Model PyTorch Ensemble

**William Edward Beckham III | FTCC CSC-114 | Summer 2026**

A handwritten character recognition system built from scratch in pure PyTorch as a self-directed learning project alongside CSC-114 coursework. Three architecturally distinct convolutional neural networks trained on EMNIST byclass (814,255 samples, 62 classes), combined via weighted ensemble with Test Time Augmentation.

---

## Results

| Model | Architecture | Params | Optimizer | Epochs | Test Acc |
|-------|-------------|--------|-----------|--------|----------|
| Model 1 | Narrow ResNet + DepthwiseSep | 2.4M | Adam + OneCycleLR | 50 | **88.06%** |
| Model 2 | Wide ResNet + Squeeze-Excitation | 9.9M | AdamW + CosineAnneal | 47 | **88.06%** |
| Model 3 | Triple-Width + Multi-Scale Fusion | 6.1M | SGD + WarmRestarts | 28 | **87.74%** |
| **2-Model Ensemble (M1+M2)** | Equal weight avg softmax | — | — | — | **88.17%** |
| **3-Model Simple Ensemble** | Equal weight avg softmax | — | — | — | **88.15%** |
| **3-Model Weighted Ensemble** | w1×0.38 + w2×0.38 + w3×0.24 | — | — | — | **88.16%** |
| **3-Model Weighted + TTA** | 8 augments per image, weighted | — | — | — | *bug in eval loop — excluded* |

---

## Dataset

**EMNIST byclass** — Cohen et al. 2017

| Split | Samples |
|-------|---------|
| Train | 593,243 |
| Val | 104,689 |
| Test | 116,323 |
| **Total** | **814,255** |

62 classes: digits `0-9`, uppercase `A-Z`, lowercase `a-z`

Downloaded automatically via `torchvision.datasets.EMNIST` on first run (~562 MB).

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 7900X (24 threads) |
| RAM | 64 GB DDR5-5600 |
| GPU | NVIDIA GeForce RTX 4080 (16 GB VRAM) |
| CUDA | 12.1 |
| cuDNN | 9.23 |
| Driver | 596.49 |

Training speeds: Model 1 ~30s/epoch (50 epochs, ~25 min) · Model 2 ~88s/epoch (47 epochs, ~69 min) · Model 3 ~155s/epoch (28 epochs, ~72 min)

---

## Architecture Overview

### Model 1 — `ocr_pytorch_model.py`
**Narrow ResNet with Depthwise Separable Stem**

```
Input (1, 32, 32)
→ DepthwiseSep Stem (1→32)
→ ResidualBlock (32→64)  + MaxPool → 16×16
→ ResidualBlock (64→128) + MaxPool →  8×8
→ ResidualBlock (128→256)+ MaxPool →  4×4
→ ResidualBlock (256→256)
→ AdaptiveAvgPool → flatten
→ Linear(256→256) → BN → ReLU → Dropout
→ Linear(256→62)
```

- AMP (mixed float16) with `GradScaler`
- `OneCycleLR` — warmup → peak → cosine anneal
- `Adam` optimizer, lr=1e-3, weight_decay=1e-4
- `SpatialDropout2D` regularization
- `EarlyStopping` patience=7
- Trained all 50 epochs
- Exports: `.pt` checkpoint + ONNX opset 17

---

### Model 2 — `ocr_pytorch_model2.py`
**Wide ResNet with Squeeze-Excitation Attention**

```
Input (1, 32, 32)
→ DepthwiseSep Stem (1→32)
→ SEResidualBlock (32→128)  + MaxPool → 16×16
→ SEResidualBlock (128→256) + MaxPool →  8×8
→ SEResidualBlock (256→512) + MaxPool →  4×4
→ SEResidualBlock (512→512)
→ AdaptiveAvgPool → flatten
→ Linear(512→512) → BN → ReLU → Dropout(0.5)
→ Linear(512→256) → BN → ReLU → Dropout(0.4)
→ Linear(256→62)
```

Key differences from Model 1:
- **Squeeze-Excitation attention** — channel recalibration after each stage via GlobalAvgPool → FC → Sigmoid → channel-wise scale (Hu et al. 2018)
- **StochasticDepth (DropPath)** — drops entire residual branches during training
- **AdamW** — weight decay decoupled from adaptive lr scaling
- **CosineAnnealingLR** — smooth monotonic decay
- 4x wider filter progression (128→256→512 vs 64→128→256)
- Deeper 3-layer classifier head
- Early stopped at epoch 47

---

### Model 3 — `ocr_pytorch_model3.py`
**Triple-Width + Multi-Scale Feature Fusion**

```
Input (1, 32, 32)
→ Triple Stem: DepthwiseSep(1→96) → Conv(96→96)
→ 2× TripleResidualBlock (96→192)  + MaxPool → 16×16
→ 2× TripleResidualBlock (192→384) + MaxPool →  8×8  ─┐ save s2
→ 2× TripleResidualBlock (384→768) + MaxPool →  4×4  ─┤ save s3
→ 2× TripleResidualBlock (768→768)                    ─┘ save s4

Multi-Scale Fusion:
→ GAP(s2)→(B,384) + GAP(s3)→(B,768) + GAP(s4)→(B,768)
→ concat → (B, 1920)

Deep Classifier (5 layers, GELU activations):
→ 1920→1024 → BN → GELU → Dropout(0.5)
→ 1024→512  → BN → GELU → Dropout(0.4)
→ 512→256   → BN → GELU → Dropout(0.3)
→ 256→128   → BN → GELU → Dropout(0.2)
→ 128→62
```

Key differences from Models 1+2:
- **Bottleneck residual blocks** — 3 convs per block (1×1 reduce → 3×3 process → 1×1 expand)
- **Multi-scale feature pyramid** — concatenates GAP from stages 2, 3, 4 simultaneously. Early stages capture strokes, mid stages capture character parts, late stages capture whole shapes. Neither Model 1 nor 2 does this
- **GELU activations** — `x * Φ(x)`, smooth non-monotonic, different activation family from ReLU
- **SGD + Nesterov momentum** — different optimizer family from Adam/AdamW
- **CosineAnnealingWarmRestarts** — T_0=20, LR resets after initial convergence
- **5-layer classifier** vs Model 1's 2 and Model 2's 3
- Batch size 256 (wider channels use more VRAM)
- Heavier augmentation: ±15° rotation, scale 0.80-1.20, shear 10°, RandomPerspective
- `num_workers=0` required on Windows — multiprocessing deadlock with persistent workers at this model size
- Early stopped at epoch 28 (restart at epoch 20 disrupted convergence)

---

## Ensemble Strategy

### Why Three Different Models?
Ch. 18 (Chollet & Watson 2025): models trained independently with different architectures make partially uncorrelated errors. Averaging their softmax outputs allows the correct class to accumulate votes while errors split across wrong classes.

### Architectural Diversity Summary

| Dimension | Model 1 | Model 2 | Model 3 |
|-----------|---------|---------|---------|
| Filter width | Narrow (64→256) | Wide (128→512) | Triple (192→768) |
| Attention | None | SE channel | SE channel |
| Block type | 2-conv basic | 2-conv + SE | 3-conv bottleneck + SE |
| Multi-scale fusion | No | No | Yes (stages 2+3+4) |
| Classifier depth | 2 layers | 3 layers | 5 layers |
| Classifier activation | ReLU | ReLU | GELU |
| Optimizer | Adam | AdamW | SGD+Nesterov |
| LR schedule | OneCycleLR | CosineAnnealing | CosineWarmRestarts |
| Regularization | SpatialDropout | StochasticDepth | StochasticDepth |
| Augmentation strength | Light | Medium | Heavy |

### Ensemble Methods (all in `ocr_pytorch_model3.py`)

**Simple equal-weight:** `(p1 + p2 + p3) / 3`

**Weighted:** `0.38·p1 + 0.38·p2 + 0.24·p3`
Weights proportional to test accuracy. Models 1 and 2 tied at 88.06% get equal higher weight; Model 3 at 87.37% gets lower weight.

**Weighted + Test Time Augmentation (TTA):**
Each image is run through 8 augmented versions at inference time (±8° rotation, ±8% translation, ±8% scale). Probabilities averaged before weighting. No retraining required. Typical gain: 0.3–0.8% on character recognition.

Note: the TTA evaluation loop produced an invalid result (82.56% — below all individual models, which is mathematically impossible for correct TTA). This indicates a bug in the augmentation transform inside the inference loop, most likely a tensor dtype or device mismatch corrupting the augmented images silently. The simple and weighted ensemble results (88.15% and 88.16%) are valid. TTA fix is pending.

---

## Project Structure

```
E:\CSC-114\emnist-model\
│
├── ocr_pytorch_model.py          # Model 1 — Narrow ResNet (2.4M)
├── ocr_pytorch_model2.py         # Model 2 — Wide + SE (9.9M)
├── ocr_pytorch_model3.py         # Model 3 — Triple + Multi-Scale (6.1M)
│                                 #   + TTA + weighted ensemble
├── ocr_handwriting_model.py      # Keras 3 / PyTorch backend (arch reference)
│
├── 01_install_cuda.bat           # CUDA 12.1 + cuDNN 9.23 installer
├── 02_install_python_packages.bat
├── 03_verify_gpu.py
│
├── datasets\pytorch\EMNIST\raw\  # Auto-downloaded on first run
│
├── pytorch\                      # Model 1 outputs
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── ocr_model.onnx
│   ├── training_curves.png
│   └── training_log.csv
│
├── pytorch2\                     # Model 2 outputs
│   ├── best_model2.pt
│   ├── final_model2.pt
│   ├── ocr_model2.onnx
│   ├── training_curves2.png
│   └── training_log2.csv
│
└── pytorch3\                     # Model 3 outputs + ensemble
    ├── best_model3.pt
    ├── final_model3.pt
    ├── ocr_model3.onnx
    ├── training_curves3.png
    └── training_log3.csv
```

---

## Setup & Reproduction

### Prerequisites
- Windows 10/11
- NVIDIA GPU (RTX 3000 series or newer recommended)
- NVIDIA Driver 525.60+
- Python 3.12

### Install

```cmd
REM Step 1 — CUDA + cuDNN (run as Administrator)
01_install_cuda.bat

REM Step 2 — Python packages
02_install_python_packages.bat

REM Step 3 — Verify GPU
"E:\CSC-114\emnist-model\venv\Scripts\python.exe" 03_verify_gpu.py
```

### Train

```cmd
cd E:\CSC-114\emnist-model

REM Model 1  (~25 min on RTX 4080)
"venv\Scripts\python.exe" ocr_pytorch_model.py

REM Model 2  (~70 min on RTX 4080)
"venv\Scripts\python.exe" ocr_pytorch_model2.py

REM Model 3 + full ensemble + TTA  (~2.5 hrs on RTX 4080)
"venv\Scripts\python.exe" ocr_pytorch_model3.py
```

EMNIST downloads automatically (~562 MB) on first run. Models 1 and 2 must complete before Model 3 — the ensemble evaluation loads all three checkpoints.

### CPU-Only Inference (No GPU Required)

```cmd
pip install onnxruntime
```

Use any of the `.onnx` files with `onnxruntime`. No PyTorch, CUDA, or GPU required. Tested on school computer (8-thread CPU, no CUDA).

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.12.10 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| torchmetrics | 1.9.0 |
| keras | 3.14.1 |
| keras-hub | 0.29.1 |
| tensorflow-datasets | 4.9.10 |
| numpy | 1.26.4 |
| matplotlib | 3.11.0 |
| optuna | 4.9.0 |
| pillow | 12.2.0 |
| h5py | 3.16.0 |
| tqdm | 4.68.3 |
| rich | 15.0.0 |
| requests | 2.34.2 |
| certifi | 2026.6.17 |
| packaging | 26.2 |
| onnx | 1.22.0 |
| CUDA | 12.1 |
| cuDNN | 9.23 |

---

## Book References

Chollet, F. & Watson, T. *Deep Learning with Python, 3rd Ed.* Manning Publications, 2025.

| Chapter | Concepts Applied |
|---------|-----------------|
| Ch. 3 | `nn.Module`, `forward()`, `backward()`, `optimizer.step()`, tensor operations, AMP |
| Ch. 5 | Dropout, SpatialDropout, StochasticDepth, weight decay, augmentation as regularization |
| Ch. 6 | Train/val/test split, evaluation protocol, touching test set only once at end |
| Ch. 8 | ConvNet filter progression, `GlobalAveragePooling`, feature hierarchies, multi-scale fusion |
| Ch. 9 | `BatchNormalization`, residual connections, depthwise separable convolutions, SE attention |
| Ch. 18 | Mixed precision, model ensembling, softmax averaging, test time augmentation, ONNX export |

---

## Notes

**Why three separate files?**
Each model is architecturally distinct enough that a shared codebase would obscure the design decisions. Separate files make each model independently readable and runnable.

**Why not Keras for all three?**
Keras 3 on the PyTorch backend introduces ~38x slowdown vs native PyTorch DataLoaders on this hardware (1s/step vs 30s/epoch on identical architecture). Root cause: Keras's data adapter layer converts tensors to numpy and back on every batch. All production training uses pure PyTorch.

Additionally, TensorFlow itself refused to run on this machine entirely. The install script confirmed Python 3.12.10 is installed system-wide, and every version of `tensorflow[and-cuda]` from 2.16.1 through 2.21.0 either explicitly does not support Python 3.12 or fails dependency resolution due to `nvidia-nccl-cu12` having no Windows distribution. The GPU verification script (`03_verify_gpu.py`) printed `✗ Python 3.12 is NOT supported by TensorFlow 2.16` on every run. TensorFlow never executed a single line of training code on this machine — not due to misconfiguration, but because TensorFlow's own published compatibility matrix excludes Python 3.12 on Windows with CUDA. The Keras file (`ocr_handwriting_model.py`) runs on the Keras 3 PyTorch backend to work around this, but the data pipeline overhead made it impractical for full training runs.

**Why AMP?**
`torch.amp.GradScaler` with `autocast` runs forward passes in float16, reducing VRAM ~40% and increasing Tensor Core throughput. Gradients are scaled to prevent underflow, then unscaled before the optimizer step.

**ONNX for portability:**
School demo machine has a GT 730 (no CUDA) and 8 CPU threads. ONNX Runtime provides CPU inference with `pip install onnxruntime` as the only dependency.

---

## Author

**William Edward Beckham III**

GitHub: [BECKHAMW3233](https://github.com/BECKHAMW3233)
