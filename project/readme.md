# EMNIST OCR — Handwritten & Printed Character Recognition
**Author:** William Beckham | **Course:** CSC-114 | **FTCC Spring/Summer 2026**

Custom OCR model trained to recognize handwritten and printed characters from scratch using the EMNIST dataset. Covers the full pipeline from environment setup through GPU-accelerated training to inference — no pretrained weights, no shortcuts.

**Recognizes:** digits 0–9, letters A–Z and a–z (62 classes total)  
**Dataset:** EMNIST byclass — 814,255 samples (Cohen et al. 2017)  
**Framework:** PyTorch 2.5+ with CUDA 12.1 (pure PyTorch) and Keras 3 on PyTorch backend  

---

## Book Reference
All architecture and training decisions are grounded in:

> Chollet & Watson, *Deep Learning with Python, 3rd Edition* (Manning, 2025)

| Chapter | Applied concepts |
|---|---|
| Ch. 2 | Tensor math, normalization, gradient descent, backpropagation |
| Ch. 3 | PyTorch tensors, `nn.Module`, `backward()`, `optimizer.step()`, `zero_grad()` |
| Ch. 5 | Overfitting, Dropout, weight decay, data augmentation as regularization |
| Ch. 6 | Universal ML workflow: define → measure → prepare → model → tune → evaluate |
| Ch. 7 | Functional API, compile/fit/evaluate, EarlyStopping, ModelCheckpoint, CSVLogger |
| Ch. 8 | ConvNet architecture, MaxPooling, GlobalAveragePooling, filter progression, augmentation |
| Ch. 9 | BatchNormalization, residual connections, depthwise separable convolutions |
| Ch. 18 | Mixed-precision training, LossScaleOptimizer, KerasTuner/Optuna, model ensembling, int8 quantization |

---

## Hardware Used
| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 7900X @ 5.3 GHz (24 threads) |
| RAM | 64 GB DDR5-5600 |
| GPU | NVIDIA RTX 4080 16 GB |
| OS | Windows 11 |

Minimum to reproduce: any NVIDIA GPU with 8+ GB VRAM and CUDA 12.x support. CPU-only training works but will be significantly slower.

---

## Repository Contents

```
├── 01_install_cuda.bat           # Step 1 — CUDA 12.3 + cuDNN install guide (run as Admin)
├── 02_install_python_packages.bat # Step 2 — Python venv + all pip packages
├── 03_verify_gpu.py              # Step 3 — confirms GPU is visible before training
├── ocr_pytorch_model.py          # Pure PyTorch training pipeline (recommended)
├── ocr_handwriting_model.py      # Keras 3 on PyTorch backend training pipeline
├── .gitignore                    # excludes venv, datasets, and compiled files
└── README.md                     # this file
```

---

## Setup — Run in This Order

### Prerequisites
- Windows 10/11 64-bit
- Python 3.10 or 3.11 (3.12 not supported by TensorFlow; PyTorch supports 3.12)
- NVIDIA GPU with CUDA 12.x support
- ~15 GB free disk space for CUDA, packages, and dataset

### Step 1 — Install CUDA and cuDNN
Run as Administrator:
```
01_install_cuda.bat
```
This opens the CUDA 12.3 and cuDNN download pages with instructions on what to select and where to copy files. You will need a free NVIDIA developer account for cuDNN.

### Step 2 — Install Python packages
```
02_install_python_packages.bat
```
Creates a virtual environment at `E:\CSC-114\emnist-model\venv\` and installs:
- PyTorch 2.5+ with CUDA 12.1
- TensorFlow 2.16+ (for Keras file)
- Keras 3, KerasHub, torchmetrics, Optuna, matplotlib, pillow

> **Note:** Change the `BASE_DIR` path inside both `.py` files if you want output somewhere other than `E:\CSC-114\emnist-model\`

### Step 3 — Verify GPU
```
"E:\CSC-114\emnist-model\venv\Scripts\python.exe" 03_verify_gpu.py
```
Must show your GPU name and `CUDA available: True` before proceeding.

Quick PyTorch GPU check:
```python
import torch
print(torch.cuda.is_available())        # must be True
print(torch.cuda.get_device_name(0))    # must show your GPU
```

### Step 4 — Train
**PyTorch version (recommended):**
```
"E:\CSC-114\emnist-model\venv\Scripts\python.exe" ocr_pytorch_model.py
```

**Keras version:**
```
"E:\CSC-114\emnist-model\venv\Scripts\python.exe" ocr_handwriting_model.py
```

First run downloads the EMNIST dataset (~540 MB) automatically. Subsequent runs use the cached copy.

---

## What Happens During Training

```
Epoch   1/50  loss: 1.8432  acc: 0.4821  |  val_loss: 1.4201  val_acc: 0.5934
Epoch   2/50  loss: 1.3104  acc: 0.6147  |  val_loss: 1.1823  val_acc: 0.6501
...
[Checkpoint] val_loss improved to 0.4821 — saved to best_model.pt
...
[EarlyStopping] Stopping training.

Test accuracy : 0.8743  (87.43%)
```

EarlyStopping halts training automatically when validation loss stops improving. Best weights are saved at every improvement. On an RTX 4080 expect roughly 15–25 minutes total.

---

## Output Files

All output lands in `E:\CSC-114\emnist-model\pytorch\` (PyTorch) or `E:\CSC-114\emnist-model\` (Keras):

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint saved during training |
| `final_model.pt` | Model state at end of training |
| `ocr_model.onnx` | ONNX export — runs anywhere without PyTorch |
| `ocr_model_quantized.pt` | int8 quantized model for faster CPU inference |
| `training_curves.png` | Accuracy and loss plot |
| `training_log.csv` | Per-epoch metrics |

---

## Model Architecture

```
Input (1, 32, 32) — grayscale character image

Stem:    DepthwiseSeparableConv(1 → 32)
Stage 1: ResidualBlock(32 → 64)  + MaxPool2d + Dropout2d
Stage 2: ResidualBlock(64 → 128) + MaxPool2d + Dropout2d  
Stage 3: ResidualBlock(128 → 256)+ MaxPool2d
Stage 4: ResidualBlock(256 → 256)
Pool:    AdaptiveAvgPool2d(1)  [GlobalAveragePooling equivalent]
Head:    Linear(256) → BatchNorm → ReLU → Dropout → Linear(62)

Output: 62 class logits
```

**Parameters:** 2,469,927 (~9.4 MB)

Each ResidualBlock (Ch. 9): Conv → BN → ReLU → Conv → BN → add skip → ReLU  
Depthwise separable convolution (Ch. 9): ~8x fewer parameters than standard Conv2D

---

## Running Inference After Training

Drop a character image into the output folder named `sample_char.png` and rerun — the script predicts automatically at the end.

To run inference only without retraining:

```python
from ocr_pytorch_model import load_saved_model, predict_image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_saved_model(r"E:\CSC-114\emnist-model\pytorch\best_model.pt", device)

results = predict_image(model, "your_character.png", device, top_k=5)
for char, confidence in results:
    print(f"{char}  {confidence:.4f}")
```

---

## Running on a Machine Without a GPU

Use the ONNX export — no PyTorch or CUDA needed:

```
pip install onnxruntime pillow numpy
```

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

session   = ort.InferenceSession("ocr_model.onnx")
img       = Image.open("your_character.png").convert("L").resize((32, 32))
arr       = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
arr       = arr[np.newaxis, np.newaxis, :, :]   # (1, 1, 32, 32)
logits    = session.run(None, {"image": arr})[0]
predicted = logits.argmax()

LABEL_MAP = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz")
print(f"Predicted: {LABEL_MAP[predicted]}")
```

---

## Reproducing From Scratch on Another Machine

```
git clone https://github.com/BECKHAMW3233/CSC-114
cd CSC-114/project
```

Then follow Steps 1–4 above. The venv is not included in the repo — recreate it with `02_install_python_packages.bat`. The EMNIST dataset downloads automatically on first training run.

---

## License
MIT
