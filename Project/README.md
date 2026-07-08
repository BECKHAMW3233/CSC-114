# MNIST OCR — Multi-Model Training Suite

Handwritten digit recognition (0–9) using an ensemble of PyTorch CNN architectures trained
across multiple resolutions with diverse optimizers. Designed as a research-grade training
pipeline with full hardware monitoring, automatic batch size detection, ONNX export, and
per-class accuracy analysis.

**Author:** William Edward Beckham III
**Program:** Computer Programming & Development AAS — FTCC
**Course:** CSC-114 AI Fundamentals I (Summer 2026)
**Hardware:** AMD Ryzen 9 7900X · 64 GB DDR5-5600 · ZOTAC RTX 4080 16 GB AMP Extreme AIRO
**Training started:** 2026-07-06

---

## Project Structure

```
├── mnist_lion_64.py           # Lion optimizer — 64×64
├── mnist_lion_128.py          # Lion optimizer — 128×128
├── mnist_lion_256.py          # Lion optimizer — 256×256
├── mnist_adamw_64.py          # Schedule-Free AdamW — 64×64
├── mnist_adamw_128.py         # Schedule-Free AdamW — 128×128
├── mnist_adamw_256.py         # Schedule-Free AdamW — 256×256 (VRAM infeasible — documented)
├── mnist_sgd_64.py            # SGD + Nesterov — 64×64
├── mnist_sgd_128.py           # SGD + Nesterov — 128×128
├── ocr_adahessian_64.py       # AdaHessian (2nd-order) — 64×64
├── ocr_adahessian_128.py      # AdaHessian (2nd-order) — 128×128
├── ocr_soap_64.py             # SOAP (Kronecker-factored) — 64×64
├── ocr_soap_128.py            # SOAP (Kronecker-factored) — 128×128
├── ocr_soap_256.py            # SOAP (Kronecker-factored) — 256×256
├── supplementary_data.py      # Digit supplementary dataset loader
├── download_datasets.py       # Dataset download utility
├── install_deps.py            # Dependency installer
└── README.md
```

**Note on file naming:** Scripts were renamed from the original `ocr_pytorch_model*.py`
convention to optimizer-based names (`mnist_lion_64.py` etc.) so the script name alone
identifies the optimizer family and resolution without opening the file. Each script is
fully independent — one optimizer, one resolution, one output folder.

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

Three base architectures across five optimizer families:

**OCRConvNet** — narrow depthwise-separable ConvNet.
Channel progression: 1→32→64→128→256. ~2.5M parameters. Depthwise-separable
convolutions split spatial and cross-channel learning into two cheaper operations,
giving ~8-9x fewer parameters for the same receptive field. Lowest memory footprint
of the three architectures — the only one that can train at 256×256 on 16GB VRAM.

**OCRConvNetWide** — wider with Squeeze-Excitation channel attention.
Channel progression: 1→32→128→256→512. ~9.7M parameters. SE blocks after each
stage learn per-channel feature recalibration — the network amplifies useful feature
detectors and suppresses less useful ones per input. This is why M2 consistently
handled structurally ambiguous letter classes better than M1 in v4 at the same
resolution. Highest parameter count; infeasible at 256×256 on 16GB VRAM (confirmed
2026-07-06: 5.5GB shared memory spillover at batch 128).

**OCRConvNetTriple** — maximum capacity with multi-scale feature pyramid fusion.
Channel progression: 1→96→192→384→768. ~4.6M parameters. Concatenates pooled
outputs from stages 2, 3, and 4 (fused dim = 1920) before the classifier, giving
the model simultaneous access to low-level stroke geometry, mid-level part
relationships, and high-level whole-character identity. Used by SGD, AdaHessian,
and SOAP scripts. The pyramid fusion is an FPN pattern from object detection applied
to character recognition.

---

### Lion — OCRConvNet (`mnist_lion_64.py`, `mnist_lion_128.py`, `mnist_lion_256.py`)

| Property | Value |
|---|---|
| Architecture | OCRConvNet (depthwise-separable, residual) |
| Optimizer | Lion (Evolved Sign Momentum — Chen et al., 2023) |
| LR | 3e-5 (Lion requires ~10x lower LR than Adam) |
| Weight decay | 1e-2 (higher WD compensates for sign-based updates) |
| Betas | (0.9, 0.99) |
| Scheduler | CosineAnnealingLR (T_max=10000, eta_min=1e-7) |
| Resolutions | 64×64, 128×128, 256×256 |
| Batch sizes | 1024 (64×64), TBD (128×128), 128 (256×256 confirmed) |
| Output | `E:\CSC-114\project\lion_64\`, `lion_128\`, `lion_256\` |

Lion uses the sign of a gradient interpolation rather than adaptive per-parameter
learning rates. Converges to smoother loss minima than Adam, which tends to produce
better real-world generalization. Memory efficient — stores one momentum buffer vs
Adam's two.

**Confirmed hardware data:**
- 64×64, batch 1024 (auto-detected): epoch 1 105s, steady ~94s/epoch, 14.0/14.4GB VRAM, 60–72°C
- 64×64 reached 99.0%+ val_acc by epoch 4; 99.4%+ by epoch 10 — exceptionally fast convergence
- 64×64 production run: **99.49% test accuracy**, 143 epochs, patience fired epoch 143
- 256×256, batch 128: 15.6/16.0GB dedicated, 1.0GB shared spillover — stable and viable
- 256×256 estimated epoch time: ~15-20 minutes → ~30-40 epochs in 10-hour budget

---

### AdamW — OCRConvNetWide (`mnist_adamw_64.py`, `mnist_adamw_128.py`, `mnist_adamw_256.py`)

| Property | Value |
|---|---|
| Architecture | OCRConvNetWide (SE attention, StochasticDepth) |
| Optimizer | Schedule-Free AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | None (Schedule-Free handles LR internally) |
| Resolutions | 64×64, 128×128 (256×256 hardware infeasible — see note) |
| Output | `E:\CSC-114\project\adamw_64\`, `adamw_128\` |

Schedule-Free AdamW eliminates LR scheduler tuning entirely. Requires
`optimizer.train()` before training and `optimizer.eval()` before validation —
handled automatically in the training loop.

> **Fixes applied (2026-07-07):**
> 1. **BatchNorm warm-up pass** — OCRConvNetWide uses BatchNorm throughout. Before each
>    val eval and the final test eval, a 50-batch warm-up pass (`model.train()` +
>    `optimizer.eval()` + forward-only) updates BatchNorm running stats at the averaged
>    parameter point `x` rather than the gradient evaluation point `y`. Without this,
>    val metrics are computed at the wrong parameter point. Required by the Schedule-Free
>    docs for any model using BatchNorm.
> 2. **Resume scheduler guard** — `scheduler` is `None` when Schedule-Free is active.
>    The resume save and load now guard `scheduler.state_dict()` and
>    `scheduler.load_state_dict()` with `if scheduler is not None` checks to prevent
>    the `NoneType has no attribute state_dict` error logged on epoch 1 of the first run.

> **256×256 hardware finding (2026-07-06):** OCRConvNetWide at 256×256, batch 128
> produced 5.5GB shared GPU memory spillover and 21.0GB total GPU memory usage —
> well beyond the 16GB dedicated ceiling. The wider channel progression (512 channels
> at stage 4) makes 256×256 infeasible on 16GB VRAM at any practical batch size.
> `mnist_adamw_256.py` exists in the repository as a documented attempt; it should
> not be run without hardware with more than 16GB VRAM.

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
| Output | `E:\CSC-114\project\sgd_64\`, `sgd_128\` |

> **Watch point:** EMNIST v4 showed SGD produces non-monotonic resolution behavior —
> M3 base 64×64 accuracy (76.93%) was *lower* than 32×32 (78.62%), the only model
> in v4 where this occurred. The MNIST SGD scripts directly test whether this
> inversion is a 62-class artifact or an SGD fundamental behavior. Per-class accuracy
> at each resolution will be compared against the v4 baseline.

---

### AdaHessian — OCRConvNetTriple (`ocr_adahessian_64.py`, `ocr_adahessian_128.py`)

Second-order optimizer using diagonal Hessian approximation via Hutchinson's method.

| Property | Value |
|---|---|
| Architecture | OCRConvNetTriple variant |
| Optimizer | AdaHessian |
| LR | 0.15 |
| Betas | (0.9, 0.999) |
| Hessian power | 1.0 |
| Warmup | 500 steps linear → CosineAnnealingLR (50-epoch horizon) |
| Resolutions | 64×64, 128×128 |
| Batch size | 256 (auto — 512 OOM on 16GB with create_graph=True) |
| Output | `E:\CSC-114\project\pytorch_adahessian_64\`, `pytorch_adahessian_128\` |

AdaHessian uses the diagonal of the Hessian matrix (second derivative of loss
with respect to each weight) to scale updates. High curvature dimensions get smaller
steps; flat dimensions get larger steps — calibrated to the actual geometry of the
loss surface rather than accumulated gradient history. Should in theory find deeper,
flatter minima that generalize better to real-world input.

> **Critical:** Requires `loss.backward(create_graph=True)`. All `.grad` fields are
> explicitly nulled after `optimizer.step()` to break the reference cycle and prevent
> cumulative VRAM leaks (bug found and fixed during EMNIST v4 development). AMP
> disabled — full precision required for Hessian computation. 256×256 is not supported:
> `create_graph=True` roughly doubles VRAM footprint of the backward pass — infeasible
> at 256×256 even at minimum batch size on 16GB VRAM.

> **Performance characteristics:** Batch 256 is the ceiling (512 OOM). ~1125s/epoch
> at 64×64 — 2x the step count of SGD (1,716 vs 858 steps) × 2x per-step cost from
> `create_graph=True` = ~4x wall time vs SGD at equivalent batch size. nvidia-smi
> reports 44–60% CUDA utilization due to polling landing in CPU/GPU alternation gaps
> during Hessian sampling — actual utilization is 100% (confirmed via Task Manager).
> Wall clock is the expected exit condition (~32 epochs in 10h at 64×64).

> **Fixes applied 2026-07-08:**
> 1. **`num_classes=62` → `num_classes=10`** — silent functional bug in both files.
>    `OCRConvNetTriple` default was `num_classes=62` (EMNIST v4 leftover). Call sites
>    pass only `drop_path` and `dropout`, relying on the default, so both files were
>    building 62-class models with 52 dead output neurons receiving zero gradient signal.
>    Parameter count confirmed fixed: 7,580,190 (62-class) → 7,573,482 (10-class),
>    a difference of 6,708 = 128 × 52 exactly. Outputs from any run using the unfixed
>    file produce incompatible ONNX exports and must be discarded.
> 2. **`WarmupCosineScheduler.state_dict()` / `load_state_dict()`** — `WarmupCosineScheduler`
>    is a plain Python class with no serialization methods. The resume save block called
>    `scheduler.state_dict()` each epoch and caught the `AttributeError` silently,
>    meaning no resume state was ever saved. Added explicit methods serializing
>    `step_count`, `base_lrs`, `warmup_steps`, `total_steps`, and `eta_min`.
> 3. **VRAM reporting** — `get_gpu_stats()` used `torch.cuda.memory_allocated()`
>    (idle snapshot after epoch, near-zero) instead of `torch.cuda.max_memory_allocated()`
>    (actual peak during training pass). Fixed to `max_memory_allocated` /
>    `max_memory_reserved` with `torch.cuda.reset_peak_memory_stats()` at epoch start.
>    Batch probe `memory_allocated` calls correctly left as-is.
> 4. **`import json` removed** — unused dead import in both files.

---

### SOAP — OCRConvNetTriple (`ocr_soap_64.py`, `ocr_soap_128.py`, `ocr_soap_256.py`)

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
| Resolutions | 64×64, 128×128, 256×256 |
| Output | `E:\CSC-114\project\pytorch_soap_64\`, `pytorch_soap_128\`, `pytorch_soap_256\` |

SOAP approximates the full curvature matrix as a Kronecker product of two smaller
factor matrices per weight tensor, capturing interactions within each layer's row and
column spaces. Better curvature approximation than AdaHessian's diagonal, at higher
computational cost. Uses standard first-order gradients so 256×256 is feasible
unlike AdaHessian.

> **Fix applied during EMNIST v4 development:** `precondition_frequency` raised from
> 10 to 100. At frequency 10, CPU-side eigendecomposition for Kronecker factoring
> dominated runtime (~50% CPU / ~50% CUDA, near-zero epoch throughput). At 100,
> overhead drops 10× while curvature updates still occur ~1,400+ times per epoch.
>
> **Fix applied 2026-07-07:** `ocr_soap_128.py` had `HAS_SUPPLEMENTARY = False`
> hardcoded, preventing supplementary data from loading regardless of whether
> `supplementary_data.py` was present. Fixed to use the same try/except import block
> as `ocr_soap_64.py` and `ocr_soap_256.py`. Also corrected `ocr_soap_256.py`
> docstring header which incorrectly identified the file as `ocr_soap_128.py`.

> **Fixes applied 2026-07-08:**
> 1. **`num_classes=62` → `num_classes=10`** — silent functional bug in all three files,
>    identical to AdaHessian fix above. 52 dead output neurons, incompatible ONNX exports.
> 2. **`WarmupCosineScheduler.state_dict()` / `load_state_dict()`** — same resume fix
>    as AdaHessian. All three files were silently failing resume saves every epoch.
> 3. **VRAM reporting** — same `max_memory_allocated` / `reset_peak_memory_stats()` fix
>    as AdaHessian. Batch probe `memory_allocated` calls correctly left as-is in all three.
> 4. **`import json` removed** — unused dead import in all three files.
> 5. **`ocr_soap_128.py` supplementary data block missing** — `get_loaders()` skipped
>    the entire `HAS_SUPPLEMENTARY` conditional block and trained on base MNIST only
>    (~51k samples instead of 439k). The 2026-07-07 fix corrected the `HAS_SUPPLEMENTARY`
>    flag but left the data loading logic incomplete. Fixed 2026-07-08 to match the full
>    conditional block in `ocr_soap_64.py` and `ocr_soap_256.py`.

---

## Resolution Coverage — Hardware Confirmed

| Script | Architecture | Optimizer | 64×64 | 128×128 | 256×256 |
|---|---|---|---|---|---|
| mnist_lion_64/128/256 | OCRConvNet | Lion | ✓ | ✓ | ✓ batch 128 |
| mnist_adamw_64/128 | OCRConvNetWide | SF-AdamW | ✓ | ✓ | ✗ 5.5GB spillover |
| mnist_sgd_64/128 | OCRConvNetTriple | SGD | ✓ | ✓ | — |
| ocr_adahessian_64/128 | OCRConvNetTriple | AdaHessian | ✓ | ✓ | — |
| ocr_soap_64/128/256 | OCRConvNetTriple | SOAP | ✓ | ✓ | TBD |

**Hardware-confirmed total: 11 models minimum** (AdamW 256×256 confirmed infeasible,
SOAP 256×256 not yet tested). Target model count is 11–12 depending on SOAP 256×256
viability.

256×256 findings:
- **Lion 256×256** — viable at batch 128. 15.6/16.0GB dedicated, 1.0GB shared
  spillover, 94% CUDA, 60°C. Not yet started as of 2026-07-07.
- **AdamW 256×256** — infeasible. 21.0GB total GPU memory, 5.5GB shared spillover
  at batch 128. OCRConvNetWide's 512-channel stage 4 exceeds 16GB at this resolution.
- **SGD 256×256** — not attempted. OCRConvNetTriple pyramid fusion at 256×256
  would require similar or greater VRAM than AdamW given the multi-scale concatenation.
- **AdaHessian 256×256** — not attempted. `create_graph=True` doubles VRAM of
  backward pass — infeasible even before activation maps are considered.
- **SOAP 256×256** — not yet tested. Standard first-order backward gives it the
  best chance after Lion.

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
   - SGD, AdaHessian, SOAP scripts: PATIENCE = 20
2. **10-hour wall clock** — elapsed time since run start exceeds 10 hours, checked at
   end of current epoch. Never cuts mid-epoch. Primary governor at 256×256.
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
python mnist_lion_256.py --batch-size 128
python mnist_lion_128.py --batch-size 512
```

**Confirmed batch sizes:**

| Script | Resolution | Batch | VRAM | Notes |
|---|---|---|---|---|
| mnist_lion_64 | 64×64 | 1024 (auto) | 14.0/14.4GB peak | 94s steady, 143 epochs, 99.49% |
| mnist_lion_256 | 256×256 | 128 (override) | 15.6GB dedicated, 1.0GB shared | ~15-20min/epoch est. |
| mnist_adamw_64 | 64×64 | 512 (override) | 13.6/14.4GB peak | 244–255s steady, 119 epochs, 99.46% |
| mnist_sgd_64 | 64×64 | 512 (override) | 11.9/15.1GB peak | 256–383s, 104 epochs, 99.49% |
| ocr_adahessian_64 | 64×64 | 256 (auto — 512 OOM) | 14.6/16.0GB peak | ~1125s/epoch, wall clock exit |
| mnist_adamw_256 | 256×256 | 128 (attempted) | 21.0GB total — infeasible | do not run |

128×128 batch sizes TBD — first runs pending.

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

**Note — ensemble check skipped:** The script attempted to load Model 1 (OCRConvNet)
at 64×64 from the EMNIST v4 path. Load failed due to classifier shape mismatch
(v4 was 62-class; this project is 10-class). Expected — no action needed.

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

**Ensemble check error (expected):**
Script attempted to load OCRConvNet from v4 path — failed on 62-class vs 10-class
classifier shape mismatch. Expected. This will not occur in the corrected scripts.

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

### AdaHessian 64×64 — Baseline Run (2026-07-07 22:54 → 2026-07-08 08:54) — DISCARDED

This run produced convergence data but unusable outputs. Run was started before the
`num_classes=62` bug was identified and fixed. Outputs discarded; folder deleted.

| Metric | Value |
|---|---|
| Epochs completed | 27 (10h wall clock) |
| Epoch time | ~1125s steady |
| Batch size | 256 (512 OOM — create_graph=True doubles VRAM) |
| VRAM | ~13.5GB reserved (allocated reporting broken — pre-fix file) |
| GPU temp | 60–63°C |
| Parameters | 7,580,190 (62-class — incorrect) |
| Resume saves | Silently failing every epoch (pre-fix file) |

**Convergence trajectory (valid signal despite discarded outputs):**

| Epoch | Val Acc | Notes |
|---|---|---|
| 1 | 97.38% | 500-step warmup baked into epoch 1 |
| 9 | 99.43% | New best |
| 20 | 99.52% | New best |
| 26 | 99.67% | Best epoch reached before wall clock |

AdaHessian is converging competitively. The trajectory confirms the architecture and
optimizer are working correctly — the 62-class output layer added dead weight but did
not corrupt the feature extraction. Corrected run started 2026-07-08 07:44:51.

### AdaHessian 64×64 — Corrected Run (2026-07-08 07:44:51 → IN PROGRESS)

| Metric | Value |
|---|---|
| Parameters | 7,573,482 (10-class — correct) |
| Batch size | 256 (512 OOM confirmed) |
| VRAM | 14.6/16.0 GB dedicated, 0.2GB shared (noise) |
| GPU temp | 61°C |
| CUDA utilization | 100% (Task Manager confirmed) |
| Resume | Functional — state_dict fix applied |
| Wall clock fires | ~2026-07-08 17:44 |

---

## Estimated Training Times

*Updated with measured data as runs complete.*

| Script | Resolution | Batch | Est. epoch time | Est. total (patience) | Est. total (10h cap) |
|---|---|---|---|---|---|
| mnist_lion_64 | 64×64 | 1024 | **94s (measured)** | **143 epochs (~3.7h)** | N/A |
| mnist_lion_128 | 128×128 | TBD | ~250-400s | ~30-40 epochs (~3-4h) | N/A |
| mnist_lion_256 | 256×256 | 128 | ~15-20min | unlikely before 10h | ~30-40 epochs |
| mnist_adamw_64 | 64×64 | 512 | **244–255s (measured)** | **119 epochs (~8.3h)** | N/A |
| mnist_adamw_128 | 128×128 | TBD | ~350-500s | ~30-40 epochs (~4-5h) | N/A |
| mnist_sgd_64 | 64×64 | 512 | **256–383s (measured)** | **104 epochs (~8.5h)** | N/A |
| mnist_sgd_128 | 128×128 | TBD | ~400-600s | ~30-40 epochs (~5-6h) | N/A |
| ocr_adahessian_64 | 64×64 | 256 | **~1125s (measured)** | unlikely before 10h | **~32 epochs** |
| ocr_adahessian_128 | 128×128 | TBD | ~2000-3000s est. | unlikely before 10h | ~12-18 epochs |
| ocr_soap_64 | 64×64 | TBD | ~180-250s | ~30-40 epochs (~2.5h) | N/A |
| ocr_soap_128 | 128×128 | TBD | ~500-700s | ~25-35 epochs (~5-7h) | possible |
| ocr_soap_256 | 256×256 | TBD | ~20-30min | unlikely before 10h | ~20-30 epochs |

**Total estimated training week:** All 11 models across the full week starting
2026-07-07. Scripts run sequentially on one machine (RTX 4080). Overnight runs
handle the longer 128×128 and 256×256 sessions. Resume capability means interrupted
runs continue without loss.

*This table will be updated with actual measured epoch times and final epoch counts
once training runs complete.*

---

## Real-World Testing — 64×64 Ensemble (2026-07-08)

First real-world benchmark of the completed 64×64 ensemble (`adamw_64`, `lion_64`,
`sgd_64`) run through `ocr_pipeline_mnist.py` against handwritten digit sheets
photographed on a Galaxy S22 Ultra. All three ONNX models loaded and voted per
character; ensemble result compared against ground truth read by hand.

**Test images:**

| Image | Ground truth (by line) | Detected chars | Ensemble result | Errors | Extra detections |
|---|---|---|---|---|---|
| test30.jpg | 467 / 582 / 319 / 048 / 297 | 15 | 467 / 582 / 319 / 048 / 29**9** | 1 | 0 |
| test31.jpg | 954 / 792 / 483 / 271 | 15 | **011** / 954 / 792 / 483 / 271 | 0 | 3 (phantom line 1) |
| test32.jpg | 678 / 321 / 012 / 456 / 999 | 17 | **01** / 678 / 321 / 012 / 456 / 999 | 0 | 2 (phantom line 1) |

**Overall accuracy on true ground-truth digits: 41/42 (97.6%)**

### Error 1 — test30.jpg, last character: ground truth 7, ensemble read 9

All three models agreed on the wrong answer (adamw_64 96.3%/9, lion_64 94.8%/9,
sgd_64 97.3%/9), so this was not a close vote — all three models were confidently
wrong. Consistent with the EMNIST v4 Finding 4 (7→9 confusion ~91% at 64×64):
digit pairs with overlapping stroke geometry at low resolution remain a real
weak point even after the pivot to a digits-only pipeline.

### Phantom detections — test31.jpg and test32.jpg

Both images produced an extra "Line 1" of 2–3 characters (`0 1 1` and `0 1`
respectively) that does not correspond to any digit in the ground truth. Both
occurred at the top of the page in the same position. Likely a `get_boxes()`
contour detection calibration issue — something in the image (edge artifact,
staple mark, page fold, or partial character from cropping) is being picked up
as valid character regions. This is a known calibration watch point from
handoff02 (`NON_DIGIT_CONF_FLOOR = 0.40`, contour thresholds tunable per image).
Not a model accuracy problem — the models are voting correctly on whatever
crop they're given; the crop itself is the issue.

**Action item:** inspect test31.jpg and test32.jpg directly to identify what
physical feature at the top of the page triggers the false-positive box
detection, and consider whether `NON_DIGIT_CONF_FLOOR` or contour area/aspect
thresholds need adjustment.

---

## Output Structure

All outputs write to `E:\CSC-114\project\` with each script in its own subfolder.
Subfolders created automatically on first run.

```
E:\CSC-114\project\
├── lion_64\                    # Lion 64×64 outputs
├── lion_128\                   # Lion 128×128 outputs
├── lion_256\                   # Lion 256×256 outputs
├── adamw_64\                   # AdamW 64×64 outputs
├── adamw_128\                  # AdamW 128×128 outputs
├── sgd_64\                     # SGD 64×64 outputs
├── sgd_128\                    # SGD 128×128 outputs
├── pytorch_adahessian_64\      # AdaHessian 64×64 outputs
├── pytorch_adahessian_128\     # AdaHessian 128×128 outputs
├── pytorch_soap_64\            # SOAP 64×64 outputs
├── pytorch_soap_128\           # SOAP 128×128 outputs
└── pytorch_soap_256\           # SOAP 256×256 outputs
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

## Hardware Target

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 7900X (24 threads) |
| RAM | 64 GB DDR5-5600 |
| GPU | NVIDIA GeForce RTX 4080 16 GB dedicated |
| Storage (primary) | NVMe SSD (E:) |
| Storage (portable) | USB drive (F:) — datasets mirrored for school machine use |
| OS | Windows 11 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |

---

## Setup

**Install dependencies:**
```bash
python install_deps.py
```

**Download datasets:**
```bash
python download_datasets.py
```

MNIST, EMNIST Digits, USPS, and SVHN download automatically via torchvision on
first run. ARDIS IV requires `download_datasets.py`.

---

## Running

Create `E:\CSC-114\project\` before first run. All subfolders are created automatically.

Run any script independently in any order:
```bash
# Lion
python mnist_lion_64.py
python mnist_lion_128.py
python mnist_lion_256.py --batch-size 128   # 256x256 requires override

# AdamW
python mnist_adamw_64.py
python mnist_adamw_128.py
# mnist_adamw_256.py — not recommended, hardware infeasible on 16GB VRAM

# SGD
python mnist_sgd_64.py
python mnist_sgd_128.py

# AdaHessian
python ocr_adahessian_64.py
python ocr_adahessian_128.py

# SOAP
python ocr_soap_64.py
python ocr_soap_128.py
python ocr_soap_256.py
```

To override batch size on any base script:
```bash
python mnist_lion_128.py --batch-size 512
```

To resume after a crash or force stop — just rerun the same command. The script
detects the resume state and checkpoint automatically and continues from the last
completed epoch.

---

## Normalization

All models use **[0, 1] normalization** — `ToTensor()` alone, no mean/std shift:
```python
arr = arr / 255.0
```
Inference pipelines must normalize identically before passing to any model or ONNX session.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Core training |
| `onnx`, `onnxruntime` | ONNX export and validation |
| `lion-pytorch` | Lion optimizer |
| `schedulefree` | Schedule-Free AdamW |
| `pytorch_optimizer` | AdaHessian and SOAP |
| `numpy`, `matplotlib` | Numerics and plotting |
| `psutil` | CPU and RAM monitoring |
| `nvidia-smi` | Real VRAM usage reporting (included with NVIDIA driver) |

Install Python packages via `python install_deps.py`.
