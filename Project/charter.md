# Project Charter: MNIST OCR — Multi-Model Digit Recognition Ensemble

## What we're building (one sentence)

A multi-architecture PyTorch ensemble that recognizes handwritten digits 0–9,
trained across multiple resolutions and five optimizer families, exported to
ONNX for inference, and evaluated against a real-world benchmark image set
with documented per-class accuracy analysis.

## Cohort

Image

## The data or tools we'll use

**Primary dataset**
- MNIST — 60,000 training / 10,000 test samples (torchvision)

**Supplementary digit datasets** (loaded via `supplementary_data.py`, graceful
skip if not present)

| Dataset | Samples | Description |
|---|---|---|
| EMNIST Digits | 280,000 | NIST digit split, same lineage as MNIST |
| USPS | 9,298 | Scanned US Postal Service envelopes |
| SVHN | 73,257 | Street View House Numbers — real-world photos |
| ARDIS IV | 7,600 | Swedish historical church records, non-NIST writers |

**Combined training set: ~430,000 samples across 10 digit classes**

**Models and training configuration**

Thirteen models trained in total across three architectures, five optimizer
families, and three resolution tiers. Each model is trained independently from
random initialization — no weights carried between resolutions or optimizers.

| Script | Architecture | Optimizer | Resolutions | Output models |
|---|---|---|---|---|
| `ocr_pytorch_model.py` | OCRConvNet (narrow, depthwise-separable, residual) | Lion | 64×64, 128×128, 256×256 | 3 |
| `ocr_pytorch_model2.py` | OCRConvNetWide (SE attention, StochasticDepth) | Schedule-Free AdamW | 64×64, 128×128, 256×256 | 3 |
| `ocr_pytorch_model3.py` | OCRConvNetTriple (triple-width, feature pyramid, GELU) | SGD + Nesterov | 64×64, 128×128 | 2 |
| `ocr_adahessian_64.py` | OCRConvNetTriple variant | AdaHessian (2nd-order) | 64×64 | 1 |
| `ocr_adahessian_128.py` | OCRConvNetTriple variant | AdaHessian (2nd-order) | 128×128 | 1 |
| `ocr_soap_64.py` | OCRConvNetTriple variant | SOAP (Kronecker-factored) | 64×64 | 1 |
| `ocr_soap_128.py` | OCRConvNetTriple variant | SOAP (Kronecker-factored) | 128×128 | 1 |
| `ocr_soap_256.py` | OCRConvNetTriple variant | SOAP (Kronecker-factored) | 256×256 | 1 |
| **Total** | | | | **13 ONNX models** |

**Architecture summary:**

- **OCRConvNet** — narrow depthwise-separable ConvNet. Channel progression
  1→32→64→128→256. Fast convergence via Lion optimizer. Lowest parameter count
  of the three base architectures (~2.5M).

- **OCRConvNetWide** — wider filter progression (1→32→128→256→512) with
  Squeeze-Excitation attention after each stage for channel-wise feature
  recalibration, and StochasticDepth (DropPath) regularization. ~9.7M
  parameters. Schedule-Free AdamW eliminates LR scheduler tuning entirely.

- **OCRConvNetTriple** — maximum capacity architecture. Triple-width channel
  progression (1→96→192→384→768), bottleneck residual blocks, multi-scale
  feature pyramid concatenating pooled outputs from stages 2, 3, and 4
  (fused dim = 1920), and a 5-layer GELU classifier head
  (1920→1024→512→256→128→10). Used by all four experimental optimizer scripts
  as well as Model 3. ~4.6M parameters.

**Resolution rationale:** 32×32 is excluded — EMNIST v4 showed that
distinguishing stroke features for digit confusion pairs (7→9, 8→9) occupy
fewer pixels than convolutional filters can reliably detect at that scale.
256×256 is excluded for OCRConvNetTriple with AdaHessian — the `create_graph=True`
requirement for Hessian computation makes it infeasible at that resolution even
at minimum batch size on 16GB VRAM.

**Training status and 256×256 caveat:** No models have been trained yet at
the time this charter is written. All scripts are built, verified, and ready
to run. The actual training runs will determine whether the 256×256 resolution
is achievable on the RTX 4080 (16GB VRAM) for each architecture. All scripts
include automatic batch size detection that steps down from 1024→512→256 on
OOM — but if OOM occurs at the minimum candidate batch size, that resolution
cannot be trained on this hardware. 256×256 is the highest risk tier:
OCRConvNetWide at 256×256 (Model 2) may OOM due to its wide filter progression
(512 channels at stage 4); SOAP at 256×256 is the most likely to succeed since
it uses standard first-order gradients. The final model count of 13 is the
target — the actual count may be lower if one or more 256×256 runs prove
infeasible. Any OOM result at 256×256 will be documented as a finding rather
than treated as a project failure.

**Tooling**
- PyTorch + torchvision (training, ONNX export)
- pytorch_optimizer (AdaHessian, SOAP second-order optimizers)
- lion-pytorch (Lion optimizer)
- schedulefree (Schedule-Free AdamW)
- ONNX Runtime (inference validation)
- psutil + nvidia-smi (per-epoch hardware monitoring)
- GitHub (Sacred Flow — Issues, branches, PRs, self-reviews)

## Project context — pivot from EMNIST v4

This project is a deliberate pivot from the EMNIST v4 62-class ensemble
(digits + uppercase + lowercase), which completed training and distillation
in July 2026. The pivot is motivated by three findings from that project:

1. **The lowercase ambiguity cluster (o, s, c, u, l, f) fails systematically
   across all architectures and optimizer families at 64×64 resolution.** Per-class
   accuracy on these classes reached 0.0–28.3% across all six distilled models.
   Post-processing compensation has reached its ceiling. This is a resolution
   problem, not a training problem.

2. **Digit recognition within the 62-class ensemble is constrained by the
   competing demands of letter recognition.** Weighting, sampling, and
   architecture choices made to handle 62 classes produce suboptimal digit
   accuracy. A digits-only pipeline removes those constraints entirely.

3. **The experimental second-order optimizer scripts (AdaHessian, SOAP) were
   created to test whether OCRConvNetTriple's capacity was being fully extracted
   by SGD.** These scripts target 64×64 and 128×128 — resolutions the EMNIST
   ensemble never fully explored. The MNIST project provides the clean test bed
   for those experiments without the 62-class complexity.

The EMNIST v4 project is not abandoned — it is complete as a 62-class system
and documented as such. This project addresses what comes next: a purpose-built,
higher-resolution digit recognizer informed by everything v4 taught about what
fails and why.

## Definition of "good enough"

Before we build, we agree this project is good enough when:

**1. MNIST test set accuracy ≥ 99.2%** across the full 13-model ensemble
(ensemble average, not individual model). Baseline for comparison: the EMNIST
v4 6-model ensemble achieved 100% on structured digit grids in real-world
benchmark testing. With 13 models, five optimizer families, and three resolution
tiers, clearing 99.2% on the clean MNIST test set is the minimum credible
result.

**2. Real-world benchmark accuracy ≥ 93.7%** on a hand-photographed image set
covering all 10 digits across multiple writers, instruments, and lighting
conditions. The 93.7% figure is the EMNIST v4 baseline (95 characters, initial
benchmark 2026-07-05) — the MNIST ensemble must match or exceed it on a larger
benchmark (minimum 200 characters).

**3. Per-class accuracy ≥ 98.0% on every digit class** in the MNIST test set.
The EMNIST v4 stress tests showed 7→9 confusion on hooked 7 variants (~91%
accuracy) and 8→9/Q confusion on open-top 8 variants (~90.5%) at 64×64.
Higher resolution training is expected to close these gaps. No single digit
class may fall below 98.0% in the final ensemble evaluation.

**4. All 13 ONNX models export cleanly** (opset 17, dynamic batch axis,
validated by `onnx.checker`) and produce correct top-1 predictions on a
smoke-test set covering all 10 digits.

*The metric is MNIST test accuracy and real-world benchmark accuracy.
The definition of good is the thresholds above — specific, measurable, and
grounded in the predecessor project's actual benchmark data.*

## What we are NOT doing (scope guard)

- **No letter or character recognition.** This is digits 0–9 only. The EMNIST
  62-class pipeline is a separate, complete body of work. Reintroducing letter
  classes is out of scope for this project regardless of how naturally it might
  follow.

- **No knowledge distillation in this phase.** EMNIST v4 showed that distilled
  models trained exclusively on clean EMNIST data generalize worse to real-world
  photos than base models trained on the full 11-source dataset. Distillation
  dataset selection requires analysis before another distillation phase is run.
  Base models only for this iteration.

- **No inference pipeline or post-processing overhaul.** The EMNIST v4
  pipeline (`ocr_pipeline.py`) already handles digit recognition modes
  (digits, digits-strict). Extending it for MNIST outputs is a separate
  project. ONNX export is the delivery artifact for this phase.

- **No automated hyperparameter search.** Optimizer hyperparameters are set
  from literature values established during EMNIST v4 development and held
  fixed. Optuna or Ray Tune are not part of this iteration.

- **No deployment.** Serving, containerization, and API endpoints are out
  of scope. The ONNX files are the deliverable.

## Team & roles

**Solo — William Beckham**

Self-review: each PR receives a written self-review documenting what changed,
what was tested, what the output confirmed, and what (if anything) was
different from expected before merge.

AI partnership (Claude): used for code generation, debugging, document
drafting, and architecture analysis. All AI output is reviewed, tested, and
explicitly accepted or rejected before commit. See `agent-guardrails.md`.
