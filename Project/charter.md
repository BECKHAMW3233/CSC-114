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
  1→32→64→128→256. Fast convergence via Lion optimizer. ~2.5M parameters.

- **OCRConvNetWide** — wider filter progression (1→32→128→256→512) with
  Squeeze-Excitation attention after each stage and StochasticDepth
  regularization. ~9.7M parameters. Schedule-Free AdamW eliminates LR
  scheduler tuning entirely.

- **OCRConvNetTriple** — maximum capacity architecture. Triple-width channel
  progression (1→96→192→384→768), bottleneck residual blocks, multi-scale
  feature pyramid concatenating pooled outputs from stages 2, 3, and 4
  (fused dim = 1920), and a 5-layer GELU classifier head
  (1920→1024→512→256→128→10). Used by Model 3 and all four experimental
  optimizer scripts. ~4.6M parameters.

**Resolution rationale:** 32×32 is excluded — EMNIST v4 confirmed that
distinguishing stroke features for digit confusion pairs (7→9, 8→9) are
unreliable at that scale, and the ambiguous lowercase cluster (o, s, c)
collapses to 0.0% across all three distilled models at 32×32. 256×256 is
excluded for OCRConvNetTriple with AdaHessian — the `create_graph=True`
requirement for Hessian computation makes it infeasible at that resolution
even at minimum batch size on 16GB VRAM.

**Training status and 256×256 caveat:** No models have been trained yet at
the time this charter is written. All scripts are built, verified, and ready
to run. The 256×256 resolution tier has not been validated on this hardware
for these architectures — the automatic batch size detection steps down from
1024→512→256 on OOM, but if OOM occurs at the minimum candidate batch size
that resolution cannot be trained. 256×256 is the highest risk tier:
OCRConvNetWide at 256×256 (Model 2) is the highest OOM risk due to its
512-channel stage 4; SOAP at 256×256 is most likely to succeed since it uses
standard first-order gradients. The target is 13 models — the actual count
may be lower if one or more 256×256 runs prove infeasible. Any OOM result
at 256×256 is a documented finding, not a project failure.

**Tooling**
- PyTorch + torchvision (training, ONNX export)
- pytorch_optimizer (AdaHessian, SOAP second-order optimizers)
- lion-pytorch (Lion optimizer)
- schedulefree (Schedule-Free AdamW)
- ONNX Runtime (inference validation)
- psutil + nvidia-smi (per-epoch hardware monitoring)
- GitHub (Sacred Flow — Issues, branches, PRs, self-reviews)

---

## Project context — pivot from EMNIST v4

This project is a deliberate pivot from the EMNIST v4 62-class ensemble
(digits + uppercase + lowercase), which completed training, distillation,
ONNX validation, and initial real-world benchmark testing in July 2026. The
pivot is motivated by four specific findings from that project, all now
confirmed with per-class accuracy data from `ocr_class_accuracy.py` run
against all 12 v4 ONNX models on 2026-07-06.

**Finding 1 — The lowercase ambiguity cluster fails at the architecture
level, not the training level.**
Across all six distilled models at 64×64, per-class accuracy on o, s, c, u,
l, and f ranged from 0.0% to 28.3%. M1 distilled 64×64: o=0.9%, s=1.4%,
c=1.2%. M3 distilled 64×64: o=0.0%, s=0.0%, c=10.9%. M2 distilled 64×64
— the widest architecture — showed the best results but still only reached
o=4.9%, s=11.7%, c=23.4%. Post-processing has reached its ceiling on these
classes. This is a resolution and stroke-endpoint detection problem requiring
a different approach, not a training distribution problem.

**Finding 2 — SGD produces a fundamentally different error geometry at
64×64, and not a better one.**
M3 base is the only model where 64×64 overall accuracy (76.93%) is *lower*
than 32×32 (78.62%). At 64×64, M3 base routes S to other classes at 1.9%
accuracy (down from 7.8% at 32×32) and O at 2.1% (down from 8.3%). M1 and
M2 improve with resolution; M3 gets worse. The M3 distilled models partially
compensate via soft label transfer from M1+M2 teachers, but o and s remain
at 0.0% in M3 distilled at 64×64 — SGD's learned weight geometry for those
classes cannot be fully corrected by distillation after the fact. In the
MNIST digits-only project, Model 3 (SGD) is running at 64×64 and 128×128 —
this finding flags SGD as the highest-risk optimizer for resolution scaling
and the one most likely to show non-monotonic accuracy behavior across
resolution tiers.

**Finding 3 — Distilled models trained on clean EMNIST data generalize worse
to real-world photos than base models.**
On all real-world benchmark images, the distilled models produced
significantly more `?` outputs and off-label predictions than the base models
at the same resolution. The base models — particularly M2 and M3 at 64×64 —
were the cleanest raw readers on real handwriting. Distillation was trained
exclusively on EMNIST byclass (697,932 samples of clean, controlled data);
the base models trained on the full 11-source dataset including noisier
real-world sources (SVHN, Chars74K, USPS). This is a documented finding for
the paper and directly informs the MNIST project: no distillation phase until
the distillation dataset selection problem is solved.

**Finding 4 — Digit recognition within the 62-class ensemble is constrained
by letter recognition demands.**
The DIGIT_BOOST=3.0x weighting, the 11-source dataset composition, and the
architecture choices were all calibrated to balance 62 classes. The v4 stress
tests confirmed 7→9 confusion on hooked 7 variants (~91% accuracy) and 8→9
confusion on open-top 8 variants (~90.5%) at 64×64. A digits-only pipeline
running at 64/128×128 removes the 62-class competing constraints entirely.

The EMNIST v4 project is not abandoned — it is complete and documented. This
project addresses what comes next.

---

## Definition of "good enough"

Before we build, we agree this project is good enough when:

**1. MNIST test set accuracy ≥ 99.2%** across the full ensemble (ensemble
average). Baseline: the EMNIST v4 12-model ensemble achieved 100% on
structured digit grids in real-world benchmark testing (80/80 characters
across 8 handwriting samples, 2026-07-05). With 13 models, five optimizer
families, three resolution tiers, and a digits-only training distribution,
clearing 99.2% on the clean MNIST test set is the minimum credible result.

**2. Real-world benchmark accuracy ≥ 93.7%** on a hand-photographed image
set covering all 10 digits across multiple writers, instruments, and lighting
conditions. The 93.7% figure is the EMNIST v4 baseline (95 characters, initial
benchmark 2026-07-05) — the MNIST ensemble must match or exceed it on a larger
benchmark (minimum 200 characters).

**3. Per-class accuracy ≥ 98.0% on every digit class** in the MNIST test
set. The v4 stress tests showed 7→9 confusion (~91%) and 8→9 confusion
(~90.5%) at 64×64. Higher resolution training across 64/128×128 is the
specific intervention expected to close these gaps. No digit class may fall
below 98.0% in the final ensemble evaluation.

**4. All successfully trained ONNX models export cleanly** (opset 17,
dynamic batch axis, validated by `onnx.checker`) and produce correct top-1
predictions on a smoke-test set covering all 10 digits.

*The metric is MNIST test accuracy and real-world benchmark accuracy.
The definition of good is the thresholds above — specific, measurable, and
grounded in the predecessor project's actual benchmark and per-class data.*

---

## What we are NOT doing (scope guard)

- **No letter or character recognition.** This is digits 0–9 only. The EMNIST
  v4 62-class pipeline is a separate, complete body of work. Reintroducing
  letter classes is out of scope regardless of how naturally it might follow.

- **No knowledge distillation in this phase.** EMNIST v4 confirmed that
  distilled models trained exclusively on clean EMNIST data generalize worse
  to real-world photos than base models trained on the full multi-source
  dataset (Finding 3 above). Distillation dataset selection requires analysis
  before another distillation phase is run. Base models only.

- **No inference pipeline overhaul.** The EMNIST v4 pipeline (`ocr_pipeline.py`)
  already handles digit recognition modes. Extending it for MNIST outputs is
  a separate project. ONNX export is the delivery artifact for this phase.

- **No automated hyperparameter search.** Optimizer hyperparameters are set
  from literature values established during EMNIST v4 development and held
  fixed. Optuna or Ray Tune are not part of this iteration.

- **No deployment.** Serving, containerization, and API endpoints are out of
  scope. The ONNX files are the deliverable.

---

## Team & roles

**Solo — William Beckham**

Self-review: each PR receives a written self-review documenting what changed,
what was tested, what the output confirmed, and what (if anything) differed
from expected before merge.

AI partnership (Claude): used for code generation, debugging, document
drafting, and architecture analysis. All AI output is reviewed, tested, and
explicitly accepted or rejected before commit. See `agent-guardrails.md`.
