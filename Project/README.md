# MNIST OCR — Multi-Model Training Suite

Handwritten digit recognition (0–9) using an ensemble of PyTorch CNN architectures trained
across multiple resolutions with diverse optimizers. Designed as a research-grade training
pipeline with full hardware monitoring, automatic batch size detection, ONNX export, and
per-class accuracy analysis.

**Author:** William Edward Beckham III
**Program:** Computer Programming & Development AAS — FTCC
**Course:** CSC-114 AI Fundamentals I (Summer 2026)
**Hardware:** AMD Ryzen 9 7900X · ASUS ROG Crosshair X670E Hero · 64 GB DDR5-5600 · ZOTAC RTX 4080 16 GB AMP Extreme AIRO
**Training started:** 2026-07-06

---

## Project Context — Pivot from EMNIST v4

This project is a deliberate pivot from the EMNIST v4 62-class ensemble (digits +
uppercase + lowercase), which completed training, distillation, ONNX validation, and
per-class accuracy analysis across all 12 models in July 2026. The pivot is driven by
four specific empirical findings from that project, all confirmed with measured per-class
data from `ocr_class_accuracy.py` run against all 12 v4 ONNX models on 2026-07-06.

**Finding 1 — The lowercase ambiguity cluster is an architecture-level failure.**
Across all six v4 distilled models at 64×64, per-class accuracy on o, s, c, u, l, f
ranged from 0.0% to 28.3%. Distillation made class o *worse* for M1 (base 49.4% →
distilled 0.2% at 32×32). Post-processing compensation has reached its ceiling. This
is a resolution and stroke-endpoint detection problem, not a training problem.

**Finding 2 — SGD produces a non-monotonic resolution response.**
M3 base is the only v4 model where 64×64 overall accuracy (76.93%) is lower than
32×32 (78.62%). At 64×64, M3 routes S to incorrect classes at 1.9% (down from 7.8%
at 32×32) and O at 2.1% (down from 8.3%). M1 and M2 both improve with resolution;
M3 regresses on its hardest classes. The MNIST project directly tests whether this
inversion is a 62-class artifact or an SGD fundamental behavior.

**Finding 3 — Distilled models generalize worse to real-world photos than base models.**
Distillation trained exclusively on clean EMNIST byclass (697,932 samples) produces
models that overfit to that distribution. Base models trained on the full 11-source
dataset including SVHN, Chars74K, and USPS generalize better to real handwriting.
No distillation phase in this project until dataset selection is resolved.

**Finding 4 — Digit accuracy within the 62-class ensemble is constrained by letter
recognition demands.**
v4 stress tests confirmed 7→9 confusion (~91%) and 8→9 confusion (~90.5%) at 64×64.
A digits-only pipeline removes the 62-class competing objective entirely.

Per-class baseline established 2026-07-06: `ocr_class_accuracy.py` run against all
12 v4 ONNX models on the school machine (i7-10700, CPU-only). This baseline is the
reference for all MNIST ensemble comparative analysis.

---

## Hardware Target

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 7900X — 12 cores / 24 threads, Zen 4, base 4.7GHz (confirmed 4701MHz), boost up to 5.6GHz, 64MB L3 cache, 170W TDP |
| Motherboard | ASUS ROG Crosshair X670E Hero (Rev 1.xx) — AMD X670E chipset, Socket AM5, 18+2 phase power delivery, PCIe 5.0, BIOS 3003 (5/5/2025, UEFI, American Megatrends Inc.) |
| GPU | ZOTAC GeForce RTX 4080 AMP Extreme AIRO 16GB (ZT-D40810B-10P) — 9,728 CUDA cores, 304 Tensor cores (4th gen), 76 RT cores (3rd gen), boost clock 2565MHz (factory OC), 256-bit bus, 22.4 Gbps GDDR6X. Driver 596.49, VBIOS 95.03.1e.00.cf, WDDM mode, CUDA runtime (driver-reported) 13.2 — note PyTorch build below targets CUDA 12.1 toolkit, which is backward compatible with driver-reported 13.2 |
| RAM | G.SKILL Trident Z5 RGB (F5-5600J3636D32GA2-TZ5RK) — 64GB (2×32GB) DDR5-5600, CL36-36-36-89, 1.25V, Intel XMP 3.0. Confirmed installed: 64.0GB, 63.2GB total physical, 67.2GB total virtual (4.00GB page file) |
| CPU Cooling | Custom single loop, mixed-generation EKWB parts — EKWB EK-Quantum Velocity² D-RGB AM5 CPU water block (nickel/plexi, current-gen, required for AM5 socket support) → original 2017 EKWB EK-CoolStream XE 360 radiator kit (triple 120mm, copper fins/brass chambers, 60mm thick) → same-era EKWB DDC pump and 3× EK-Vardar 120mm PWM fans, all carried forward from the original XE 360 kit purchase rather than bought new for this build. Exact pump model/spec and fan variant/RPM not confirmed — this is 2017-era hardware and current EKWB spec sheets (DDC 4.2 PWM, EK-Vardar EVO series) don't necessarily reflect what shipped in that kit |
| GPU Cooling | Air (stock triple-fan AMP Extreme AIRO cooler) — not part of the watercooling loop. EKWB does make a full-cover block for this exact card (EK-Quantum Vector² AMP/Trinity RTX 4080, D-RGB, Nickel + Plexi — announced Nov 2022, shipping from late Dec 2022, MSRP ~€265/$301 USD). It was sold out when checked in late 2023/early 2024; combined with the price, the GPU stayed air-cooled. Loop is CPU-only, EKWB products only |
| Thermal Paste (CPU + GPU) | Thermal Grizzly Kryonaut — 12.5 W/mK thermal conductivity, non-electrically-conductive, -250°C to +350°C operating range |
| Case | Thermaltake Core P8 Tempered Glass E-ATX Full-Tower (CA-1Q2-00M1WN-00) |
| PSU | 850W, 80 PLUS Gold — manufacturer/model not yet confirmed. Sufficient headroom for the 320W GPU cap + 170W CPU TDP combined load (490W) with margin for the rest of the system |
| Storage (C: — OS boot, "BOOT") | SanDisk SDSSDA240G, 238GB usable, SATA SSD — Windows install, page file (C:\pagefile.sys) |
| Storage (D: — bulk/archival, "GAME") | SAMSUNG HD103SJ, 1TB, SATA HDD |
| Storage (E: — training/dataset, "New Volume") | Samsung SSD 980 PRO 500GB, NVMe, PCIe interface |
| Storage (F: — portable/thumb drive) | USB DISK 3.0, ~25.6GB, USB 3.0 — used for dataset mirroring on school machine |
| Storage (G: — MB support media) | ~5.3GB, "MB Support CD" — motherboard driver disc, mounted virtual/optical media |
| OS | Windows 11 Pro, version 10.0.26200 (Build 26200), kernel 10.0.26200.8737 |
| Python | 3.12.10 |
| CUDA | 12.1 (PyTorch build) / 13.2 (driver-reported runtime) |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |

---

## Setup

**1. Install CUDA (Windows):**
```bash
01_install_cuda.bat
```

**2. Install Python packages:**
```bash
02_install_python_packages.bat
```
This creates a dedicated virtual environment at `E:\CSC-114\project\venv`,
activates it, installs every package listed in Dependencies below, copies
the training scripts into `E:\CSC-114\project`, and runs GPU verification
automatically as its final step (calls `03_verify_gpu.py` if present).
Takes 5–15 minutes; the PyTorch/CUDA download alone is ~2.5GB.

**3. Verify GPU is visible to PyTorch (manual re-check, optional):**
```bash
python 03_verify_gpu.py
```
Already run automatically at the end of step 2 — rerun manually only if you
need to re-confirm after changing drivers or the venv.

**4. Install into an already-active environment (alternative to step 2):**
```bash
python install_deps.py
```
Use this instead of step 2 only if the venv already exists and is currently
activated — this installs into whatever Python environment is active, it
does not create or activate the venv itself.

MNIST, EMNIST Digits, USPS, and SVHN download automatically via torchvision
on first run of any training script. ARDIS IV must be sourced and placed
manually per `supplementary_data.py`'s expected path — there is no separate
dataset-download script in this repo.

---

## Running

Create `E:\CSC-114\project\` before first run. All subfolders are created automatically.

Run any script independently in any order:
```bash
# Lion
python mnist_lion_64.py
python mnist_lion_128.py

# AdamW
python mnist_adamw_64.py
python mnist_adamw_128.py

# SGD
python mnist_sgd_64.py
python mnist_sgd_128.py

# SOAP
python ocr_soap_64.py
python ocr_soap_128.py
```

To override batch size on any base script:
```bash
python mnist_lion_128.py --batch-size 512
```

To resume after a crash or force stop — just rerun the same command. The script
detects the resume state and checkpoint automatically and continues from the last
completed epoch.

---

## Dependencies

Training runs inside a dedicated virtual environment at
`E:\CSC-114\project\venv`, created and populated by
`02_install_python_packages.bat` (Windows batch, full setup including CUDA
package install) or `install_deps.py` (Python-only installer, assumes venv
already active). This is a separate, project-scoped environment — not the
same as the global Python 3.12 install used for other tools on this machine
(see note below).

| Package | Purpose |
|---|---|
| `torch`, `torchvision`, `torchaudio` | Core training (`torch==2.5.1+cu121`) |
| `torchmetrics` | Training metrics |
| `onnx` | Model export |
| `onnxruntime-gpu==1.19.2` | Pipeline inference — version pinned to match CUDA 12.1, do not upgrade without checking compatibility |
| `lion-pytorch` | Lion optimizer |
| `schedulefree` | Schedule-Free AdamW |
| `pytorch_optimizer` | SOAP and 100+ other optimizers |
| `numpy`, `matplotlib` | Numerics and plotting |
| `pandas` | Kaggle CSV preprocessing |
| `scipy` | SVHN `.mat` file loading |
| `certifi` | SSL certificate fix for USPS download |
| `kaggle` | Supplementary dataset download API — requires `kaggle.json` at `C:\Users\Will\.kaggle\kaggle.json` |
| `psutil` | CPU and RAM monitoring |
| `optuna` | Hyperparameter search (installed, not currently used per charter scope guard) |
| `keras>=3.0`, `keras-hub`, `tensorflow-datasets` | Course assignment requirements, unrelated to this project's training pipeline |
| `nvidia-smi` | Real VRAM usage reporting (included with NVIDIA driver, not pip-installed) |

Install via `02_install_python_packages.bat` (creates the venv, installs
everything, copies scripts, runs GPU verification) or `python install_deps.py`
(installs into whatever environment is currently active — run this only after
manually activating `E:\CSC-114\project\venv`).

**Note on environment separation:** the `pip freeze` capture in the appendix
below was taken from this machine's *global* Python 3.12 environment
(`C:\Users\Will\AppData\Local\Programs\Python\Python312\...`), not from the
project venv. That's why it shows unrelated tooling (ComfyUI, transformers,
Flask) and is missing `onnx`, `onnxruntime-gpu`, `lion-pytorch`,
`schedulefree`, and `pytorch_optimizer` — those are correctly installed
inside `E:\CSC-114\project\venv`, confirmed by the SOAP training log
referencing `E:\CSC-114\project\venv\Lib\site-packages\torch\...` directly.
The appendix is kept for reference on what's globally available on this
machine, but the venv (not the global environment) is the actual training
environment and matches the table above.

<details>
<summary>Full <code>pip freeze</code> output — global environment, captured 2026-07-08</summary>

```
absl-py==2.4.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.2
aiosignal==1.4.0
alembic==1.17.2
annotated-types==0.7.0
attrs==25.4.0
av==16.0.1
blinker==1.9.0
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.2.1
colorama==0.4.6
comfyui-embedded-docs==0.3.1
comfyui-workflow-templates-core==0.3.27
comfyui-workflow-templates-media-api==0.3.20
comfyui-workflow-templates-media-image==0.3.27
comfyui-workflow-templates-media-other==0.3.40
comfyui-workflow-templates-media-video==0.3.15
comfyui_frontend_package==1.34.8
comfyui_workflow_templates==0.7.54
duckduckgo_search==8.0.2
einops==0.8.1
filelock==3.19.1
Flask==3.1.1
frozenlist==1.8.0
fsspec==2025.9.0
greenlet==3.3.0
h5py==3.16.0
huggingface-hub==0.36.0
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
keras==3.14.1
kornia==0.8.2
kornia_rs==0.1.10
lxml==5.4.0
Mako==1.3.10
markdown-it-py==4.2.0
MarkupSafe==3.0.2
mdurl==0.1.2
ml_dtypes==0.5.4
mpmath==1.3.0
multidict==6.7.0
namex==0.1.0
networkx==3.5
numpy==2.3.1
optree==0.19.1
packaging==25.0
pandas==2.3.1
pillow==11.3.0
primp==0.15.0
propcache==0.4.1
psutil==7.1.3
pydantic==2.12.5
pydantic-settings==2.12.0
pydantic_core==2.41.5
Pygments==2.20.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
pytz==2025.2
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.3
rich==15.0.0
safetensors==0.7.0
scipy==1.16.3
sentencepiece==0.2.1
setuptools==70.2.0
six==1.17.0
spandrel==0.4.1
SQLAlchemy==2.0.45
sympy==1.13.1
tokenizers==0.22.1
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchsde==0.2.6
torchvision==0.20.1+cu121
tqdm==4.67.1
trampoline==0.1.2
transformers==4.57.3
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.4.0
Werkzeug==3.1.3
yarl==1.22.0
```

</details>

## Project Structure

Current actual contents of `BECKHAMW3233/CSC-114/Project/` on GitHub, as of
commit `c260fae`:

```
CSC-114/Project/
├── adamw_64/                          # AdamW 64×64 — COMPLETE (99.46%)
│   ├── v1_adamw_64_64.onnx
│   ├── v1_adamw_64_best_64.pt
│   ├── v1_adamw_64_cli_20260707_052554.txt
│   ├── v1_adamw_64_curves_64.png
│   ├── v1_adamw_64_final_64.pt
│   └── v1_adamw_64_log_64.csv
├── adamw_128/                         # AdamW 128×128 — COMPLETE (99.42%)
│   ├── v1_adamw_128_128.onnx
│   ├── v1_adamw_128_best_128.pt
│   ├── v1_adamw_128_cli_20260709_234437.txt
│   ├── v1_adamw_128_curves_128.png
│   ├── v1_adamw_128_final_128.pt
│   └── v1_adamw_128_log_128.csv
├── lion_64/                           # Lion 64×64 — COMPLETE (99.49%)
│   ├── v1_lion_64_64.onnx
│   ├── v1_lion_64_best_64.pt
│   ├── v1_lion_64_cli_20260706_234832.txt
│   ├── v1_lion_64_curves_64.png
│   ├── v1_lion_64_final_64.pt
│   ├── v1_lion_64_log_64.csv
│   └── v1_lion_64_quantized_64.pt
├── lion_128/                          # Lion 128×128 — COMPLETE (99.45%)
│   ├── v1_lion_128_128.onnx
│   ├── v1_lion_128_best_128.pt
│   ├── v1_lion_128_cli_20260709_113115.txt
│   ├── v1_lion_128_curves_128.png
│   ├── v1_lion_128_final_128.pt
│   ├── v1_lion_128_log_128.csv
│   └── v1_lion_128_quantized_128.pt
├── sgd_64/                            # SGD 64×64 — COMPLETE (99.49%)
│   ├── v1_sgd_64_64.onnx
│   ├── v1_sgd_64_best_64.pt
│   ├── v1_sgd_64_cli_20260707_134326.txt
│   ├── v1_sgd_64_curves_64.png
│   ├── v1_sgd_64_final_64.pt
│   └── v1_sgd_64_log_64.csv
├── sgd_128/                           # SGD 128×128 — COMPLETE (98.86%)
│   ├── v1_sgd_128_128.onnx
│   ├── v1_sgd_128_best_128.pt
│   ├── v1_sgd_128_cli_20260710_163528.txt
│   ├── v1_sgd_128_curves_128.png
│   ├── v1_sgd_128_final_128.pt
│   └── v1_sgd_128_log_128.csv
├── pytorch_soap_64/                   # SOAP 64×64 — COMPLETE (99.65%)
│   ├── soap_64.onnx
│   ├── soap_64_20260709_003312.log
│   ├── soap_64_best.pt
│   ├── soap_64_curves.png
│   ├── soap_64_final.pt
│   └── soap_64_training_log.csv
├── pytorch_soap_128/                  # SOAP 128×128 — COMPLETE (99.66%)
│   ├── soap_128.onnx
│   ├── soap_128_20260711_113053.log
│   ├── soap_128_best.pt
│   ├── soap_128_curves.png
│   ├── soap_128_final.pt
│   └── soap_128_training_log.csv
├── datasets/
│   └── kaggle/
├── 01_install_cuda.bat
├── 02_install_python_packages.bat
├── 03_verify_gpu.py
├── README.md
├── install_deps.py
├── mnist_adamw_64.py
├── mnist_adamw_128.py
├── mnist_lion_64.py
├── mnist_lion_128.py
├── mnist_sgd_64.py
├── mnist_sgd_128.py
├── ocr_soap_64.py
├── ocr_soap_128.py
├── ocr_pipeline_mnist.py
└── supplementary_data.py
```

**Model output folders present:** `adamw_64`, `adamw_128`, `lion_64`,
`lion_128`, `sgd_64`, `sgd_128`, `pytorch_soap_64`, `pytorch_soap_128`
— each containing the full set of training artifacts (ONNX export,
best/final checkpoints, per-epoch CSV log, training curves PNG, and CLI
transcript).

**All 8 planned models now complete.**

**Note on file naming:** each script name identifies its optimizer family and
resolution directly (`mnist_lion_64.py`, `ocr_soap_128.py`, etc.) without
needing to open the file. Each script is fully independent — one optimizer,
one resolution, one output folder.

---

## Datasets

### Primary

**MNIST** — LeCun et al., 1998
60,000 training / 10,000 test samples. 10 classes: digits 0–9. 28×28 grayscale.
Downloaded automatically by torchvision on first run.

### Supplementary Digit Sources

All five sources load automatically via `supplementary_data.py`. Each is gracefully
skipped if not present. Letter-only datasets (Kaggle A-Z, Chars74K, PG-HWLD, EMNIST
Balanced) are explicitly excluded — this is a digits-only pipeline.

| Dataset | Samples | Description |
|---|---|---|
| EMNIST Digits | 240,000 | NIST digits split, same lineage as MNIST |
| MNIST (supplementary) | 60,000 | Loaded via wrapper for transform consistency |
| USPS | 7,291 | Scanned US Postal Service envelopes |
| SVHN | 73,257 | Street View House Numbers, real-world photographs |
| ARDIS IV | 7,600 | Swedish historical church records, 19th–20th century writers |

**Confirmed combined training set: 439,148 samples** — verified on first run
2026-07-06. Class weight range: `0.000019 — 0.000024` (extremely tight — confirms
well-balanced digit distribution across all five sources).

Dataset location:
```
E:\CSC-114\emnist-model\datasets\pytorch\
```

Portable USB drive (F:) mirrors the dataset folder for use on school machines.
Drive letter changes per machine — only one path reference needs updating per session.

### Class Weighting

`supplementary_data.py` uses pure inverse-frequency weighting (`DIGIT_BOOST = 1.0`,
`NUM_CLASSES = 10`). No boost multiplier needed — with letter datasets excluded there
is nothing to counterbalance. The original EMNIST v4 pipeline used `DIGIT_BOOST = 3.0x`
to compensate for 372,450 Kaggle A-Z letter samples; that is not needed here.

---

## Models and Scripts

### Architecture Overview

Three base architectures across four optimizer families:

**OCRConvNet** — narrow depthwise-separable ConvNet.
Channel progression: 1→32→64→128→256. ~2.5M parameters. Depthwise-separable
convolutions split spatial and cross-channel learning into two cheaper operations,
giving ~8-9x fewer parameters for the same receptive field. Lowest memory footprint
of the three architectures.

**OCRConvNetWide** — wider with Squeeze-Excitation channel attention.
Channel progression: 1→32→128→256→512. ~9.7M parameters. SE blocks after each
stage learn per-channel feature recalibration — the network amplifies useful feature
detectors and suppresses less useful ones per input. This is why M2 consistently
handled structurally ambiguous letter classes better than M1 in v4 at the same
resolution. Highest parameter count of the three architectures.

**OCRConvNetTriple** — maximum capacity with multi-scale feature pyramid fusion.
Channel progression: 1→96→192→384→768. ~4.6M parameters. Concatenates pooled
outputs from stages 2, 3, and 4 (fused dim = 1920) before the classifier, giving
the model simultaneous access to low-level stroke geometry, mid-level part
relationships, and high-level whole-character identity. Used by SGD and SOAP
scripts. The pyramid fusion is an FPN pattern from object detection applied
to character recognition.

---

### Lion — OCRConvNet (`mnist_lion_64.py`, `mnist_lion_128.py`)

| Property | Value |
|---|---|
| Architecture | OCRConvNet (depthwise-separable, residual) |
| Optimizer | Lion (Evolved Sign Momentum — Chen et al., 2023) |
| LR | 3e-5 (Lion requires ~10x lower LR than Adam) |
| Weight decay | 1e-2 (higher WD compensates for sign-based updates) |
| Betas | (0.9, 0.99) |
| Scheduler | CosineAnnealingLR (T_max=10000, eta_min=1e-7) |
| Resolutions | 64×64, 128×128 |
| Batch sizes | 1024 (64×64), 128 (128×128) |
| Output | `E:\CSC-114\project\lion_64\`, `lion_128\` |

Lion uses the sign of a gradient interpolation rather than adaptive per-parameter
learning rates. Converges to smoother loss minima than Adam, which tends to produce
better real-world generalization. Memory efficient — stores one momentum buffer vs
Adam's two.

**Confirmed hardware data:**
- 64×64, batch 1024 (auto-detected): epoch 1 105s, steady ~94s/epoch, 14.0/14.4GB VRAM, 60–72°C
- 64×64 reached 99.0%+ val_acc by epoch 4; 99.4%+ by epoch 10 — exceptionally fast convergence
- 64×64 production run: **99.49% test accuracy**, 143 epochs, patience fired epoch 143
- 128×128, batch 128: 7.8/7.9GB VRAM, 63–67°C, **99.45% test accuracy**, 104 epochs (10.07h wall clock)

---

### AdamW — OCRConvNetWide (`mnist_adamw_64.py`, `mnist_adamw_128.py`)

| Property | Value |
|---|---|
| Architecture | OCRConvNetWide (SE attention, StochasticDepth) |
| Optimizer | Schedule-Free AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | None (Schedule-Free handles LR internally) |
| Resolutions | 64×64, 128×128 |
| Batch sizes | 512 (64×64, override), 128 (128×128) |
| Output | `E:\CSC-114\project\adamw_64\`, `adamw_128\` |

Schedule-Free AdamW eliminates LR scheduler tuning entirely. Requires
`optimizer.train()` before training and `optimizer.eval()` before validation —
handled automatically in the training loop.

> **BatchNorm warm-up pass:** OCRConvNetWide uses BatchNorm throughout. Before
> each val eval and the final test eval, a 50-batch warm-up pass (`model.train()`
> + `optimizer.eval()` + forward-only) updates BatchNorm running stats at the
> averaged parameter point, as required by the Schedule-Free docs for any model
> using BatchNorm.

---

### SGD — OCRConvNetTriple (`mnist_sgd_64.py`, `mnist_sgd_128.py`)

| Property | Value |
|---|---|
| Architecture | OCRConvNetTriple (triple-width, feature pyramid, GELU classifier) |
| Optimizer | SGD + Nesterov momentum |
| LR | 0.01 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Scheduler | CosineAnnealingLR (T_max=10000, eta_min=1e-6) |
| Resolutions | 64×64, 128×128 |
| Batch sizes | 512 (64×64, override), 128 (128×128, override) |
| Output | `E:\CSC-114\project\sgd_64\`, `sgd_128\` |

> **Watch point:** EMNIST v4 showed SGD produces non-monotonic resolution behavior —
> M3 base 64×64 accuracy (76.93%) was *lower* than 32×32 (78.62%), the only model
> in v4 where this occurred. The MNIST SGD scripts directly test whether this
> inversion is a 62-class artifact or an SGD fundamental behavior. Per-class accuracy
> at each resolution will be compared against the v4 baseline.

---

### SOAP — OCRConvNetTriple (`ocr_soap_64.py`, `ocr_soap_128.py`)

Kronecker-factored second-order optimizer (Shampoo + Adam hybrid).

| Property | Value |
|---|---|
| Architecture | OCRConvNetTriple variant |
| Optimizer | SOAP |
| LR | 1e-3 |
| Betas | (0.95, 0.95) |
| Weight decay | 5e-4 |
| Precondition frequency | every 100 steps |
| Warmup | 500 steps linear → CosineAnnealingLR (50-epoch horizon) |
| Resolutions | 64×64, 128×128 |
| Batch sizes | 512 (64×64, auto — hardcoded ceiling), 128 (128×128, override) |
| Output | `E:\CSC-114\project\pytorch_soap_64\`, `pytorch_soap_128\` |

SOAP approximates the full curvature matrix as a Kronecker product of two smaller
factor matrices per weight tensor, capturing interactions within each layer's row and
column spaces. Precondition frequency is set to every 100 steps, keeping
Kronecker-factoring overhead low while still updating curvature estimates
roughly 1,400+ times per epoch.

---

## Resolution Coverage — Hardware Confirmed

| Script | Architecture | Optimizer | 64×64 | 128×128 |
|---|---|---|---|---|
| mnist_lion_64/128 | OCRConvNet | Lion | ✓ | ✓ |
| mnist_adamw_64/128 | OCRConvNetWide | SF-AdamW | ✓ | ✓ |
| mnist_sgd_64/128 | OCRConvNetTriple | SGD | ✓ | ✓ |
| ocr_soap_64/128 | OCRConvNetTriple | SOAP | ✓ | ✓ |

**Target model count: 8** (4 optimizer families × 2 resolutions) — all complete.

---

## Training Configuration

### Script Architecture Change

The original three multi-resolution scripts (`ocr_pytorch_model.py`, `ocr_pytorch_model2.py`,
`ocr_pytorch_model3.py`) were split into individual per-resolution, per-optimizer scripts.

**Reason:** The original scripts ran all resolutions sequentially in one process with
no way to run a specific resolution independently. If 64×64 was already complete and
you needed to run 128×128 only, you had to restart the entire script or modify the
`RESOLUTIONS` list manually. The split scripts solve this — each is fully independent,
named by optimizer and resolution, and can be started, stopped, and resumed without
affecting any other script.

### Exit Conditions (all scripts)

Training stops on whichever fires first:

1. **Early stopping** — val loss/acc fails to improve for PATIENCE epochs. Best
   checkpoint saved automatically. Primary exit at 64×64 and 128×128 — MNIST at 10
   classes converges fast. Confirmed: Lion 64×64 was still finding marginal improvements
   at epoch 100+ before patience fired at epoch 143.
   - Lion, AdamW scripts: PATIENCE = 15
   - SGD, SOAP scripts: PATIENCE = 20
2. **10-hour wall clock** — elapsed time since run start exceeds 10 hours, checked at
   end of current epoch. Never cuts mid-epoch.
3. **OOM / cuDNN engine failure** — caught by exception handler and steps down to
   next batch size candidate. If all candidates fail, script exits cleanly.

No fixed epoch cap. The wall clock and patience are the only governors.

### Batch Size Auto-Detection and Override

All scripts probe the largest safe batch size via a forward+backward pass at each
candidate. After each probe, nvidia-smi is queried for actual dedicated VRAM usage —
this catches Windows' silent shared memory spillover which PyTorch's own memory
reporting cannot detect.

All base split scripts accept a `--batch-size` argument to skip probing entirely:

```bash
python mnist_lion_128.py --batch-size 512
```

**Confirmed batch sizes:**

| Script | Resolution | Batch | VRAM | Notes |
|---|---|---|---|---|
| mnist_lion_64 | 64×64 | 1024 (auto) | 14.0/14.4GB peak | 94s steady, 143 epochs, 99.49% |
| mnist_lion_128 | 128×128 | 128 | 7.8/7.9GB peak | ~346s avg (derived), 104 epochs, 99.45% |
| mnist_adamw_64 | 64×64 | 512 (override) | 13.6/14.4GB peak | 244–255s steady, 119 epochs, 99.46% |
| mnist_adamw_128 | 128×128 | 128 | 11.8/13.2GB peak | ~864s steady, 42 epochs, 99.42% |
| mnist_sgd_64 | 64×64 | 512 (override) | 11.9/15.1GB peak | 256–383s, 104 epochs, 99.49% |
| ocr_soap_64 | 64×64 | 512 (auto — hardcoded ceiling) | 9.5/12.9GB peak | ~172s steady, 107 epochs, 99.65% |
| mnist_sgd_128 | 128×128 | 128 (override) | 11.7/14.4GB peak | ~961–1257s, 33 epochs (10.17h wall clock), 98.86% |
| ocr_soap_128 | 128×128 | 128 (override — auto-detect skipped) | 9.5/12.9GB peak | ~622–707s, 49 epochs (patience exit), 99.66% |

### Resume Capability

All scripts save a resume state after every epoch as a `.pt` file:
```
v1_lion_64_resume_64.pt
v1_lion_128_resume_128.pt
...
```

On restart, if both the resume `.pt` and the best checkpoint `.pt` file exist, training
continues from the next epoch with full state restored: model weights, optimizer state,
scheduler state, scaler state, early stop counter, best val loss, and full history.
Resume file is deleted on clean completion so a fresh restart doesn't accidentally
resume a finished run. Crash or force-stop loses at most one epoch of progress.

### VRAM Monitoring Fix

Original scripts used `torch.cuda.memory_allocated()` which captures the idle snapshot
between epochs after tensors are freed — reporting near-zero values while Task Manager
showed 9-11GB actually in use. Fixed to `torch.cuda.max_memory_allocated()` with
`torch.cuda.reset_peak_memory_stats()` called at epoch start. Now reports actual peak
VRAM during the training forward+backward pass, consistent with Task Manager readings.

### Windows DataLoader Fix

`NUM_WORKERS=4` during training, `num_workers_override=0` for val and test DataLoaders:

```python
train_loader = make_dataloader(train_ds, batch_size, use_weighted_sampler=True)  # NUM_WORKERS=4
val_loader   = make_dataloader(val_ds,  batch_size, num_workers_override=0)
test_loader  = make_dataloader(test_ds, batch_size, num_workers_override=0)
```

At `NUM_WORKERS=8` (original value), Windows DataLoader spawned 8 worker processes
during the post-training eval pass. This caused a deadlock that froze VS Code, the
terminal, and snipping tool — requiring a force kill. Root cause: Windows multiprocessing
for DataLoader is unreliable outside the main training loop. Val and test loaders use
0 workers (main thread). Training loaders use 4 workers for throughput.

---

## Normalization

All models use **[0, 1] normalization** — `ToTensor()` alone, no mean/std shift:
```python
arr = arr / 255.0
```
Inference pipelines must normalize identically before passing to any model or ONNX session.

---

## Completed Training Results

### Lion 64×64 — COMPLETE (2026-07-06 23:48:32 → 2026-07-07)

| Metric | Value |
|---|---|
| Test accuracy | **99.49%** |
| Test loss | 0.2973 |
| Best checkpoint | Epoch 128 (val_loss 0.2829, val_acc 1.0000) |
| Patience fired | Epoch 143 |
| Total epochs | 143 |
| Epoch 1 time | 105s (auto-detect overhead) |
| Steady epoch time | ~94s |
| VRAM peak | 14.0 / 14.4 GB dedicated |
| GPU temp range | 60–72°C |
| Batch size | 1024 (auto-detected) |
| Parameters | 2,456,563 |
| Model size | 9.4 MB (float32) |

**Per-class accuracy on test set (10,000 samples):**

| Digit | Accuracy | Samples |
|---|---|---|
| 0 | 99.8% | 980 |
| 1 | 99.8% | 1,135 |
| 2 | 99.5% | 1,032 |
| 3 | **100.0%** | 1,010 |
| 4 | 99.1% | 982 |
| 5 | 99.2% | 892 |
| 6 | 99.3% | 958 |
| 7 | 99.4% | 1,028 |
| 8 | 99.8% | 974 |
| 9 | 98.9% ← worst | 1,009 |

All 10 classes ≥ 98.0% — charter threshold met. ✓

**Convergence highlights from training log:**
- Epoch 1: val_acc 94.41%, val_loss 0.4980
- Epoch 4: val_acc 99.10% — crossed 99% on epoch 4
- Epoch 10: val_acc 99.51%
- Epoch 50: val_acc 99.91%
- Epoch 81: val_acc 99.98% (first near-perfect val)
- Epoch 95: val_acc 100.00% (first perfect val)
- Epoch 128: val_loss 0.2829 — best checkpoint saved

**Output files (all confirmed clean):**

| File | Size |
|---|---|
| `v1_lion_64_best_64.pt` | — |
| `v1_lion_64_final_64.pt` | 9.4 MB |
| `v1_lion_64_64.onnx` | 9.4 MB |
| `v1_lion_64_quantized_64.pt` | 9.2 MB |
| `v1_lion_64_log_64.csv` | 143 rows |
| `v1_lion_64_curves_64.png` | — |
| `v1_lion_64_cli_20260706_234832.txt` | — |

Resume state cleared on completion.

### Lion 128×128 — COMPLETE (2026-07-09)

| Metric | Value |
|---|---|
| Test accuracy | **99.45%** |
| Test loss | 0.2986 |
| Best checkpoint | Epoch 104 (10h wall clock) |
| Total epochs | 104 |
| Steady epoch time | ~346s (derived: 10.07h ÷ 104 epochs — not directly logged per-epoch) |
| VRAM peak | 7.8 / 7.9 GB dedicated |
| GPU temp range | 63–67°C |
| Batch size | 128 |
| Model size | 9.4 MB (float32) |

**Per-class accuracy on test set:**

| Digit | Accuracy | Samples |
|---|---|---|
| 0 | 99.7% | 980 |
| 1 | 99.7% | 1,135 |
| 2 | 99.5% | 1,032 |
| 3 | **99.8%** | 1,010 |
| 4 | 99.8% | 982 |
| 5 | 98.9% | 892 |
| 6 | 99.2% | 958 |
| 7 | 99.4% | 1,028 |
| 8 | 99.7% | 974 |
| 9 | 98.7% ← worst | 1,009 |

All 10 classes ≥ 98.0% — charter threshold met. ✓

**Output files:**

| File | Size |
|---|---|
| `v1_lion_128_final_128.pt` | 9.4 MB |
| `v1_lion_128_128.onnx` | 9.4 MB |
| `v1_lion_128_quantized_128.pt` | 9.2 MB |
| `v1_lion_128_log_128.csv` | — |
| `v1_lion_128_curves_128.png` | — |
| `v1_lion_128_cli_20260709_113115.txt` | — |

### AdamW 64×64 — COMPLETE (2026-07-07 05:25:54 → 2026-07-07)

| Metric | Value |
|---|---|
| Test accuracy | **99.46%** |
| Test loss | 0.2992 |
| Best checkpoint | Epoch 104 (val_loss 0.2827, val_acc 1.0000) |
| Patience fired | Epoch 119 |
| Total epochs | 119 |
| Epoch 1 time | 264s |
| Steady epoch time | ~244–255s |
| VRAM peak | 13.6 / 14.4 GB dedicated |
| GPU temp range | 62–69°C |
| Batch size | 512 (override) |
| Parameters | 9,712,490 |
| Model size | 37.1 MB (float32) |

**Per-class accuracy on test set (10,000 samples):**

| Digit | Accuracy | Samples |
|---|---|---|
| 3 | 99.9% | 1,010 |
| 0 | 99.8% | 980 |
| 8 | 99.8% | 974 |
| 1 | 99.6% | 1,135 |
| 5 | 99.6% | 892 |
| 9 | 99.5% | 1,009 |
| 2 | 99.3% | 1,032 |
| 6 | 99.1% | 958 |
| 7 | 99.0% | 1,028 |
| 4 | 99.0% ← worst | 982 |

All 10 classes ≥ 99.0% — charter threshold met. ✓

**Convergence highlights:**
- Epoch 1: val_acc 98.71%, val_loss 0.3361 — started higher than Lion (94.41%) due to larger model
- Epoch 14: val_loss 0.2893 — first patience counter fired epoch 15
- Epoch 21: val_acc 99.88%, val_loss 0.2878
- Epoch 53: val_acc 99.99% (first near-perfect val)
- Epoch 65: val_acc 100.00% (first perfect val)
- Epoch 104: val_loss 0.2827 — best checkpoint saved
- Epoch 119: patience fired 15/15 — halted

**Output files (all confirmed clean):**

| File | Size |
|---|---|
| `v1_adamw_64_best_64.pt` | — |
| `v1_adamw_64_final_64.pt` | 37.1 MB |
| `v1_adamw_64_64.onnx` | 37.1 MB |
| `v1_adamw_64_log_64.csv` | 119 rows |
| `v1_adamw_64_curves_64.png` | — |
| `v1_adamw_64_cli_20260707_052554.txt` | — |

Resume state cleared on completion.

### AdamW 128×128 — COMPLETE (2026-07-09)

| Metric | Value |
|---|---|
| Test accuracy | **99.42%** |
| Test loss | 0.2999 |
| Best checkpoint | Epoch 42 (val_loss 0.2845, val_acc 0.9993) |
| Total epochs | 42 (10.20h wall clock) |
| Steady epoch time | ~864–865s |
| VRAM peak | 11.8 / 13.2 GB dedicated |
| GPU temp range | 63–71°C |
| Batch size | 128 |
| Parameters | 9,712,490 |
| Model size | 37.1 MB (float32) |

**Per-class accuracy on test set:**

| Digit | Accuracy | Samples |
|---|---|---|
| 0 | 99.9% | 980 |
| 3 | 99.8% | 1,010 |
| 8 | 99.8% | 974 |
| 1 | 99.7% | 1,135 |
| 5 | 99.6% | 892 |
| 9 | 99.4% | 1,009 |
| 7 | 99.2% | 1,028 |
| 4 | 99.0% | 982 |
| 6 | 99.0% | 958 |
| 2 | 98.8% ← worst | 1,032 |

**Output files:**

| File | Size |
|---|---|
| `v1_adamw_128_final_128.pt` | 37.1 MB |
| `v1_adamw_128_128.onnx` | 37.1 MB |
| `v1_adamw_128_log_128.csv` | — |
| `v1_adamw_128_curves_128.png` | — |
| `v1_adamw_128_cli_20260709_234437.txt` | — |

### SGD 64×64 — COMPLETE (2026-07-07 13:43:26 → 2026-07-07)

| Metric | Value |
|---|---|
| Test accuracy | **99.49%** |
| Test loss | 0.3009 |
| Best checkpoint | Epoch 84 (val_loss 0.2953, val_acc 0.9964) |
| Patience fired | Epoch 104 (20/20) |
| Total epochs | 104 |
| Epoch 1 time | 649s (DataLoader cold-start) |
| Steady epoch time | ~256–383s (rising through run as RAM climbed) |
| VRAM peak | 11.9 / 15.1 GB dedicated |
| GPU temp range | 57–66°C |
| Batch size | 512 (override) |
| Parameters | 4,581,354 |
| Model size | 17.5 MB (float32) |

**Per-class accuracy on test set (10,000 samples):**

| Digit | Accuracy | Samples |
|---|---|---|
| 0 | 99.9% | 980 |
| 3 | 99.9% | 1,010 |
| 1 | 99.8% | 1,135 |
| 8 | 99.8% | 974 |
| 9 | 99.5% | 1,009 |
| 2 | 99.4% | 1,032 |
| 7 | 99.3% | 1,028 |
| 6 | 99.2% | 958 |
| 4 | 99.1% | 982 |
| 5 | 98.9% ← worst | 892 |

All 10 classes ≥ 98.0% — charter threshold met. ✓

**Non-monotonic resolution finding — initial data point:**
SGD 64×64 achieved 99.49%, matching Lion 64×64 exactly. This is the baseline for
the non-monotonic resolution test. SGD 128×128 result will determine whether the
v4 inversion (64×64 < 32×32) was a 62-class artifact or an SGD fundamental behavior.

**Convergence highlights:**
- Epoch 1: val_acc 84.97% — slow SGD start vs Lion (94.41%) and AdamW (98.71%)
- Epoch 5: val_acc 96.31% — first 5 epochs gained 11+ points
- Epoch 10: val_acc 98.33%
- Epoch 32: val_acc 99.52% — first time above 99.5%
- Epoch 84: val_loss 0.2953 — best checkpoint saved
- Epoch 104: patience 20/20 fired — halted
- Note: epoch times rose from ~256s to ~383s in late epochs due to RAM growth
  (14.4GB → 16.9GB system RAM across the run). No VRAM impact.

**Output files (all confirmed clean):**

| File | Size |
|---|---|
| `v1_sgd_64_best_64.pt` | — |
| `v1_sgd_64_final_64.pt` | 17.6 MB |
| `v1_sgd_64_64.onnx` | 17.5 MB |
| `v1_sgd_64_log_64.csv` | 104 rows |
| `v1_sgd_64_curves_64.png` | — |
| `v1_sgd_64_cli_20260707_134326.txt` | — |

Resume state cleared on completion.

### SGD 128×128 — COMPLETE (2026-07-10)

| Metric | Value |
|---|---|
| Test accuracy | **98.86%** |
| Test loss | 0.3290 |
| Best checkpoint | Epoch 27 (val_loss 0.3275, val_acc 0.9898) |
| Total epochs | 33 (10.17h wall clock) |
| Steady epoch time | ~961–1257s |
| VRAM peak | 11.7 / 14.4 GB dedicated |
| GPU temp range | 59–65°C |
| Batch size | 128 |
| Parameters | 4,581,354 |
| Model size | 17.5 MB (float32) |

**Per-class accuracy on test set:**

| Digit | Accuracy | Samples |
|---|---|---|
| 0 | 99.6% | 980 |
| 3 | 99.6% | 1,010 |
| 1 | 99.6% | 1,135 |
| 8 | 99.5% | 974 |
| 5 | 99.3% | 892 |
| 7 | 99.2% | 1,028 |
| 9 | 98.8% | 1,009 |
| 6 | 97.9% | 958 |
| 4 | 97.7% | 982 |
| 2 | 97.4% ← worst | 1,032 |

**Output files:**

| File | Size |
|---|---|
| `v1_sgd_128_final_128.pt` | 17.6 MB |
| `v1_sgd_128_128.onnx` | 17.5 MB |
| `v1_sgd_128_log_128.csv` | — |
| `v1_sgd_128_curves_128.png` | — |
| `v1_sgd_128_cli_20260710_163528.txt` | — |

### SOAP 64×64 — COMPLETE (2026-07-09)

| Metric | Value |
|---|---|
| Test accuracy | **99.65%** |
| Best val accuracy | 100.00% (epoch 87) |
| Total epochs | 107 (patience exit) |
| Steady epoch time | ~166–178s |
| Parameters | 7,573,482 |
| Batch size | 512 |
| VRAM peak | 9.5 / 12.9 GB dedicated |
| GPU temp range | 61–67°C |

**Convergence highlights:**

| Epoch | Val Acc |
|---|---|
| 1 | 98.62% |
| 9 | 99.80% |
| 87 | **100.00%** — best checkpoint |
| 107 | 100.00% — patience exhausted |

Outputs in `E:\CSC-114\project\pytorch_soap_64\` — ONNX exported and
validated (`soap_64.onnx`).

**Comparison across all completed 64×64 models:** SOAP (99.65%) is the
strongest result so far, ahead of Lion (99.49%), SGD (99.49%), and AdamW
(99.46%).

---

### SOAP 128×128 — COMPLETE (2026-07-11)

| Metric | Value |
|---|---|
| Test accuracy | **99.66%** |
| Best val accuracy | 99.88% (epoch 29) |
| Total epochs | 49 (patience 20 exhausted) |
| Steady epoch time | ~622–707s |
| Parameters | 7,573,482 |
| Batch size | 128 (override) |
| VRAM peak | 9.5 / 12.9 GB dedicated |
| GPU temp range | 62–67°C |

**Convergence highlights:**

| Epoch | Val Acc |
|---|---|
| 1 | 99.20% |
| 13 | 99.71% |
| 19 | 99.78% |
| 29 | **99.88%** — best checkpoint |
| 49 | 99.82% — patience exhausted |

Outputs in `E:\CSC-114\project\pytorch_soap_128\` — ONNX exported and
validated (`soap_128.onnx`).

**Comparison across all completed 128×128 models:** SOAP (99.66%) is the
strongest result, ahead of AdamW (99.42%) and SGD (98.86%).

---

## Estimated Training Times

*Updated with measured data as runs complete.*

| Script | Resolution | Batch | Est. epoch time | Est. total (patience) | Est. total (10h cap) |
|---|---|---|---|---|---|
| mnist_lion_64 | 64×64 | 1024 | **94s (measured)** | **143 epochs (~3.7h)** | N/A |
| mnist_lion_128 | 128×128 | 128 | **~346s (derived avg)** | **104 epochs — wall clock (10.07h)** | N/A — complete |
| mnist_adamw_64 | 64×64 | 512 | **244–255s (measured)** | **119 epochs (~8.3h)** | N/A |
| mnist_adamw_128 | 128×128 | 128 | **~864s (measured)** | **42 epochs — wall clock (10.20h)** | N/A — complete |
| mnist_sgd_64 | 64×64 | 512 | **256–383s (measured)** | **104 epochs (~8.5h)** | N/A |
| mnist_sgd_128 | 128×128 | 128 | **~961–1257s (measured)** | **33 epochs — wall clock (10.17h)** | N/A — complete |
| ocr_soap_64 | 64×64 | 512 | **~172s (measured)** | **107 epochs — patience exit (5.1h)** | N/A — complete |
| ocr_soap_128 | 128×128 | 128 | **~622–707s (measured)** | **49 epochs — patience exit (~8.6h)** | N/A — complete |

**Total estimated training week:** All 8 models across the full week starting
2026-07-07. Scripts run sequentially on one machine (RTX 4080). Overnight runs
handle the longer 128×128 sessions. Resume capability means interrupted
runs continue without loss.

*This table will be updated with actual measured epoch times and final epoch counts
once training runs complete.*

---

## Real-World Testing — 8-Model Ensemble (2026-07-12)

**Pipeline fix applied before this run:** `--model-dir` originally scanned
recursively with no exclusions, which swept up the entire `venv/` tree —
including onnx's own unit-test fixtures (hundreds of files literally named
`model.onnx`) and unrelated model-zoo files — resulting in a 1925-model
"ensemble" that was mostly garbage. Fixed by pruning `venv`, `.venv`, `env`,
`site-packages`, `__pycache__`, `.git`, and `node_modules` from the walk
before it descends into them. Confirmed working: this run correctly found
and loaded exactly the intended 8 trained models.

First full real-world run of the completed 8-model ensemble (all four
optimizer families, both resolutions) through `ocr_pipeline_mnist.py`
against 14 handwritten digit-sheet photos taken on a Galaxy S22 Ultra, scanned
via `--model-dir`. No hand-scored ground truth was recorded for this run —
the numbers below are the pipeline's own detection/consensus output, not
accuracy against a known-correct transcription.

**Note on scope:** a 15th image (`test9.png`) was not included in this run —
the shell glob used (`test*.jpg`) only matched `.jpg` files, so the one
`.png` in the set was silently excluded before the script ever saw it. Not
a pipeline bug; `resolve_image_paths()` does support `.png`. Re-run with
`test*.jpg test*.png` (or `test*.*`) to include it next time.

**Per-image results:**

| Image | Chars detected | Full-agreement consensus |
|---|---|---|
| test1.jpg | 10 | 80.0% |
| test2.jpg | 10 | 80.0% |
| test3.jpg | 11 | 72.7% |
| test4.jpg | 11 | 72.7% |
| test5.jpg | 10 | 70.0% |
| test6.jpg | 11 | 54.5% |
| test7.jpg | 9 | 66.7% |
| test21.jpg | 71 | 32.4% |
| test22.jpg | 63 | 63.5% |
| test23.jpg | 41 | 95.1% |
| test24.jpg | 58 | 32.8% |
| test30.jpg | 15 | 60.0% |
| test31.jpg | 15 | 53.3% |
| test32.jpg | 17 | 64.7% |

**Per-model accuracy against known ground truth (test1–test7):**

test1–test6 are photos of a handwritten 0–9 reference grid (ground truth
`0123456789`); test7 is a handwritten shuffled grid (ground truth
`790341258`). Scored each model's individual read against these known-correct
sequences, character by character (69 total ground-truth digits across the
7 images):

| Model | Correct | Total | Accuracy |
|---|---|---|---|
| soap_128 | 67 | 69 | **97.1%** |
| soap_64 | 67 | 69 | **97.1%** |
| adamw_64 | 66 | 69 | 95.7% |
| sgd_128 | 66 | 69 | 95.7% |
| sgd_64 | 66 | 69 | 95.7% |
| adamw_128 | 64 | 69 | 92.8% |
| lion_64 | 63 | 69 | 91.3% |
| lion_128 | 51 | 69 | **73.9%** |

SOAP is the strongest real-world generalizer at both resolutions, matching
its lead on the held-out test split. `lion_128` is the clear outlier —
roughly 20 points behind every other model, missing 18 of 69 characters
where every other model in the ensemble got it right. Per-image breakdown:

| Image | adamw_128 | adamw_64 | lion_128 | lion_64 | soap_128 | soap_64 | sgd_128 | sgd_64 |
|---|---|---|---|---|---|---|---|---|
| test1.jpg | 10/10 | 10/10 | 9/10 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| test2.jpg | 10/10 | 10/10 | 8/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| test3.jpg | 9/10 | 9/10 | 8/10 | 8/10 | 9/10 | 9/10 | 10/10 | 9/10 |
| test4.jpg | 9/10 | 10/10 | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 |
| test5.jpg | 7/10 | 8/10 | 6/10 | 8/10 | 9/10 | 9/10 | 8/10 | 8/10 |
| test6.jpg | 10/10 | 10/10 | 5/10 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| test7.jpg | 9/9 | 9/9 | 6/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |

`lion_128` never once posts the best score of the 8 models on any of the 7
scored images, and drops as low as 50% on test6.jpg (a pencil-written, lower
contrast sheet) — the largest single-image gap of any model. test5.jpg was
the hardest sheet across the board (every model dropped below 90%), which
lines up with it being visibly lower contrast than the others.

**Ensemble (team vote) accuracy — the number that actually matters:**

The 8 models don't just report individually — `ocr_pipeline_mnist.py` combines
all 8 predictions per character into one majority/weighted vote. Scoring that
combined ensemble output (not any single model) against the same 69
ground-truth digits:

| Metric | Accuracy |
|---|---|
| **Ensemble vote (final combined answer)** | **97.1%** (67/69) |
| Plain average of the 8 individual models | 92.4% |
| Best single model (soap_128 / soap_64) | 97.1% |
| Worst single model (lion_128) | 73.9% |

The ensemble ties the strongest individual model and completely absorbs
`lion_128`'s 73.9% drag — when the other 7 models outvote it, its wrong
answers get overridden rather than dragging the team average down. This is
the intended behavior of an ensemble: robust to one weak member, not
just an average of its parts.

**Per-image ensemble result vs. ground truth:**

| Image | Ensemble read | Ground truth | Correct |
|---|---|---|---|
| test1.jpg | 0123456789 | 0123456789 | 10/10 |
| test2.jpg | 0123456789 | 0123456789 | 10/10 |
| test3.jpg | 01234562891 | 0123456789 | 9/10 |
| test4.jpg | 01234567891 | 0123456789 | 10/10 |
| test5.jpg | 0123456989 | 0123456789 | 9/10 |
| test6.jpg | 01234567891 | 0123456789 | 10/10 |
| test7.jpg | 790341258 | 790341258 | 9/9 |

**Against the charter's ≥99.2% ensemble accuracy threshold:** 97.1% falls
short. Two things worth being upfront about here: (1) this is only 69
scored characters, well under the charter's own ≥200-character minimum for
the real-world benchmark, so 97.1% is a preliminary read, not a final
verdict against that threshold; (2) both misses (test3.jpg, test5.jpg)
involve real distribution shift — test5.jpg in particular is visibly
lower-contrast than the other reference sheets, and that's reflected in
every one of the 8 individual models also dropping below 90% on that same
image, not just the ensemble.

### Finding 1 — lion_128 is the consistent outlier, now confirmed against ground truth

Across the character-level detail in the log, `lion_128` is disproportionately
the model that splits off from an otherwise-unanimous group, and when it does,
its own top-choice confidence is also visibly lower than the rest of the
ensemble on that same character (e.g. everyone-says-2-at-95%+ vs. lion_128
saying 1 at 42.9%; everyone-says-7-at-95%+ vs. lion_128 saying 1 at 33.1%).
Scoring against the known-correct reference grids above confirms this isn't
just a confidence artifact: `lion_128` scores 73.9% real-world accuracy
against 91-97% for every other model — roughly a 20-point gap. This is a
reproducible pattern across multiple images, not an isolated mistake — worth
flagging as a genuine per-optimizer generalization gap between 64×64 and
128×128 for Lion specifically, distinct from the other three optimizer
families.


### Finding 2 — confidence calibration varies by optimizer, not just by vote

`soap_64`, `soap_128`, and `adamw_64` consistently land 94-97% confidence on
their top prediction even under real-world (non-test-set) conditions.
`lion_128` and `sgd_128` are the two models that regularly dip into the
30-80% confidence range on harder/lower-contrast characters. This is a
useful axis for the optimizer comparison beyond raw test-set accuracy —
confidence calibration under distribution shift, not just correctness.

### Finding 3 — consensus rate tracks character density/layout, not just legibility

Clean, single-block reference sheets (test1, test2) landed at 80% full
agreement; a densely-packed 14-line stress sheet (test21) dropped to 32.4%.
This looks like a segmentation/line-grouping strain issue at high character
density rather than a per-character model-accuracy problem — the `get_boxes()`
line-grouping logic may need retuning for dense multi-line layouts. Untested
so far: whether the low-consensus characters on dense sheets are actually
being misread, or whether consensus is dropping because of box/line-grouping
noise while the underlying reads are still correct. That distinction isn't
answered by this run and needs a follow-up pass with hand-scored ground truth.

**Action items:**
1. ~~Score a representative subset of these images against hand-transcribed
   ground truth~~ — done above for test1-test7 (69 known digits); still
   outstanding for the remaining 7 images (test21-test24, test30-test32),
   which don't have an obvious known-correct sequence to score against.
2. Investigate the `lion_128` accuracy gap specifically (73.9% vs. 91-97%
   for every other model) — is it a training issue (undertrained, needs a
   longer patience run) or a resolution mismatch in how Lion's optimizer
   settings were tuned for 128×128?
3. Re-run including `test9.png` (see scope note above).
4. Look at whether `get_boxes()`'s line-grouping thresholds need adjustment
   for the dense sheets (test21, test24, test22) specifically.

---

## Output Structure

**Local training output** writes to `E:\CSC-114\project\` with each script in
its own subfolder. Lion, AdamW, and SGD scripts use short names matching
their GitHub folder names directly (`lion_64`, `adamw_64`, `sgd_64`, etc.).
SOAP scripts write to `pytorch_soap_*` locally, which also matches its GitHub
folder name — see Project Structure above for the confirmed repo layout.
Subfolders created automatically on first run.

```
E:\CSC-114\project\
├── lion_64\                    # Lion 64×64 outputs
├── lion_128\                   # Lion 128×128 outputs
├── adamw_64\                   # AdamW 64×64 outputs
├── adamw_128\                  # AdamW 128×128 outputs
├── sgd_64\                     # SGD 64×64 outputs
├── sgd_128\                    # SGD 128×128 outputs
├── pytorch_soap_64\            # SOAP 64×64 outputs
└── pytorch_soap_128\           # SOAP 128×128 outputs
```

Each script produces per run:

```
v1_lion_64_best_64.pt           # Best checkpoint (val loss) — saved every improvement
v1_lion_64_final_64.pt          # Final epoch weights + metadata
v1_lion_64_64.onnx              # ONNX export (opset 17, dynamic batch axis)
v1_lion_64_quantized_64.pt      # int8 quantized (Lion scripts only)
v1_lion_64_log_64.csv           # Per-epoch metrics including peak VRAM
v1_lion_64_curves_64.png        # Accuracy + loss training curves
v1_lion_64_resume_64.pt         # Resume state (deleted on clean completion)
v1_lion_64_cli_YYYYMMDD_HHMMSS.txt  # Timestamped full CLI transcript
```

Datasets remain in their original location:
```
E:\CSC-114\emnist-model\datasets\pytorch\
```

---

## Total Training Time

Summed directly from the per-epoch wall-clock seconds recorded in each
model's training log/CLI transcript — every epoch's `[...s]` value added up
per model, not estimated.

| Model | Epochs | Total time |
|---|---|---|
| Lion 64×64 | 143 | 13,522s (3.76h) |
| Lion 128×128 | 104 | 36,251s (10.07h) |
| AdamW 64×64 | 119 | 29,650s (8.24h) |
| AdamW 128×128 | 42 | 36,698s (10.19h) |
| SGD 64×64 | 104 | 30,060s (8.35h) |
| SGD 128×128 | 33 | 36,603s (10.17h) |
| SOAP 64×64 | 107 | 17,869s (4.96h) |
| SOAP 128×128 | 49 | 31,245s (8.68h) |

**Total GPU training time across all 8 models: 231,898 seconds — 64.42 hours
(~2.68 days) of continuous RTX 4080 compute.**

This is training time only — dataset loading, ONNX export/validation, and
per-class accuracy analysis at the end of each run are not included in these
per-epoch sums.

---
