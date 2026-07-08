# Project Charter: MNIST OCR — Multi-Model Digit Recognition Ensemble

## What we're building (one sentence)

An ensemble of PyTorch models that recognizes handwritten digits 0–9.

*(Expanded: three CNN architectures, five optimizer families, and multiple
resolutions are each trained independently and exported to ONNX, then
evaluated together against a real-world handwritten benchmark.)*

## Cohort

Image

## The data or tools we'll use

**Primary dataset**
- MNIST — 60,000 training / 10,000 test samples (torchvision)

**Supplementary digit datasets** (loaded via `supplementary_data.py`, graceful
skip if not present)

| Dataset | Samples | Description |
|---|---|---|
| EMNIST Digits | 240,000 | NIST digit split, same lineage as MNIST |
| MNIST (supplementary) | 60,000 | Loaded via wrapper for transform consistency |
| USPS | 7,291 | Scanned US Postal Service envelopes |
| SVHN | 73,257 | Street View House Numbers — real-world photos |
| ARDIS IV | 7,600 | Swedish historical church records, non-NIST writers |

**Confirmed combined training set: 439,148 samples** across 10 digit classes.

**Models and training configuration**

Thirteen models trained in total across three architectures, five optimizer
families, and up to three resolution tiers. Each model is trained
independently from random initialization — no weights carried between
resolutions or optimizers.

| Script family | Architecture | Optimizer | Resolutions | Output models |
|---|---|---|---|---|
| `mnist_lion_*.py` | OCRConvNet (narrow, depthwise-separable) | Lion | 64×64, 128×128, 256×256 | 3 |
| `mnist_adamw_*.py` | OCRConvNetWide (SE attention, StochasticDepth) | Schedule-Free AdamW | 64×64, 128×128 (256×256 infeasible) | 2 |
| `mnist_sgd_*.py` | OCRConvNetTriple (triple-width, feature pyramid) | SGD + Nesterov | 64×64, 128×128 | 2 |
| `ocr_adahessian_*.py` | OCRConvNetTriple variant | AdaHessian (2nd-order) | 64×64, 128×128 | 2 |
| `ocr_soap_*.py` | OCRConvNetTriple variant | SOAP (Kronecker-factored) | 64×64, 128×128, 256×256 | 3 |
| **Total** | | | | **12–13 ONNX models** |

**Architecture summary:**

- **OCRConvNet** — narrow depthwise-separable ConvNet, ~2.5M parameters.
  Lowest memory footprint of the three; the only one confirmed viable at
  256×256 on 16GB VRAM.
- **OCRConvNetWide** — wider filter progression with Squeeze-Excitation
  attention, ~9.7M parameters. Confirmed infeasible at 256×256 (5.5GB shared
  memory spillover, 21GB total).
- **OCRConvNetTriple** — maximum-capacity architecture with multi-scale
  feature pyramid fusion, ~4.6M parameters. Used by SGD, AdaHessian, and SOAP.

**Tooling**
- PyTorch + torchvision (training, ONNX export)
- pytorch_optimizer (AdaHessian, SOAP)
- lion-pytorch (Lion optimizer)
- schedulefree (Schedule-Free AdamW)
- ONNX Runtime (inference validation)
- psutil + nvidia-smi (per-epoch hardware monitoring)
- GitHub (Sacred Flow — Issues, branches, PRs, self-reviews)

## Project context — pivot from EMNIST v4

This project is a deliberate pivot from the EMNIST v4 62-class ensemble
(digits + uppercase + lowercase), which completed training, distillation, and
ONNX validation across all 12 models in July 2026. The pivot is motivated by
four measured findings from that project:

1. **The lowercase ambiguity cluster is an architecture-level failure.**
   Per-class accuracy on o, s, c, u, l, f ranged from 0.0–28.3% across all six
   distilled models at 64×64. This is a resolution and stroke-endpoint
   detection problem, not a training problem.
2. **SGD produces a non-monotonic resolution response.** The v4 SGD model was
   the only one where 64×64 accuracy was lower than 32×32. This project
   directly tests whether that was a 62-class artifact.
3. **Distilled models generalize worse to real-world photos than base
   models.** No distillation phase in this project until dataset selection
   is resolved.
4. **Digit accuracy within the 62-class ensemble was constrained by letter
   recognition demands** (7→9 confusion ~91%, 8→9 confusion ~90.5% at 64×64).
   A digits-only pipeline removes that competing objective entirely.

The EMNIST v4 project is not abandoned — it is complete and documented as a
62-class system. This project addresses what comes next.

## Definition of "good enough"

Before we build, we agree this project is good enough when:

- **MNIST test set accuracy ≥ 99.2%**, averaged across the full ensemble.
- **Real-world benchmark accuracy ≥ 93.7%** on a hand-photographed image set
  of at least 200 characters (the EMNIST v4 baseline, which this project must
  match or exceed on a larger sample).
- **Per-class accuracy ≥ 98.0% on every digit class** in the MNIST test set.
- **Every trained model exports cleanly to ONNX** (opset 17, dynamic batch
  axis, validated by `onnx.checker`) and produces correct top-1 predictions
  on a smoke-test set covering all 10 digits.

*The metric is MNIST test accuracy and real-world benchmark accuracy. The
definition of good is the thresholds above — specific, measurable, and
grounded in the predecessor project's actual benchmark data.*

## What we are NOT doing (scope guard)

- **No letter or character recognition.** Digits 0–9 only.
- **No knowledge distillation in this phase.** Distillation dataset selection
  needs its own analysis before another distillation phase is run.
- **No inference pipeline overhaul.** `ocr_pipeline_mnist.py` handles
  ensemble voting and real-world testing; extending it further is a separate
  project.
- **No automated hyperparameter search.** Optimizer hyperparameters are set
  from literature values and held fixed.
- **No deployment.** ONNX files are the deliverable.

## Team & roles

**Solo — William Beckham**

Self-review: each PR receives a written self-review documenting what changed,
what was tested, what the output confirmed, and what (if anything) was
different from expected before merge.

AI partnership (Claude): used for code generation, debugging, document
drafting, and benchmark analysis. All AI output is reviewed, tested, and
explicitly accepted or rejected before commit. See `agent-guardrails.md`.
