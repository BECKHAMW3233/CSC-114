# Four Optimizers. One Digit. Watch Them Vote.

**CSC-114 · AI Fundamentals I · Final Project**

Eight CNNs — trained with four different optimizers at two resolutions — each read every digit in an image independently. Most of the time they agree instantly. When they don't, that disagreement is the interesting part.

**Author:** William Edward Beckham III
**Hardware:** RTX 4080 16GB · Ryzen 9 7900X
**Best single model:** SOAP 128×128 — 99.66%

---

## Example Ensemble Read

*Ground truth: `50368` — 8 models voting per digit — 4 / 5 correct (80%)*

| Digit | Result |
|---|---|
| 1 | **5** — all 8 models agree |
| 2 | **0** — all 8 models agree |
| 3 | **3** — all 8 models agree |
| 4 | **??** — genuine split, no majority |
| 5 | **8** — all 8 models agree |

**Voting models, in order:** Lion 64×64 · AdamW 64×64 · SGD 64×64 · SOAP 64×64 · Lion 128×128 · AdamW 128×128 · SGD 128×128 · SOAP 128×128

> *Illustrative example — the digits and the `??` split shown here demonstrate how the vote works. A real disagreement like this hasn't been captured in a test run yet; see "The Inference Pipeline" below.*

---

## Why This Project Only Reads Digits

This is a deliberate narrowing from an earlier 62-class letters+digits ensemble. Four specific findings from that project's own per-class accuracy data are why.

**01 — The lowercase ambiguity cluster was structural, not fixable.**
Letters like o, s, c, u, l, f scored 0–28% across all six prior distilled models. That's a resolution and stroke-endpoint problem, not something more training solves.

**02 — SGD broke the "higher resolution helps" assumption.**
One model scored worse at 64×64 than at 32×32 — the only model in the prior ensemble where that happened. This project's SGD runs test whether that's an SGD trait or a 62-class artifact.

**03 — Distillation hurt real-world generalization.**
Models distilled on clean EMNIST overfit to that distribution and read messy real handwriting worse than their un-distilled base versions did.

**04 — Digits were competing against letters for accuracy.**
7↔9 and 8↔9 confusion sat near 90% in the combined ensemble. Removing letters removes that competing objective entirely.

### Term: What "Distillation" Means Here — and Why None of These 8 Models Use It

Distillation trains a smaller **student** model to copy the outputs of a larger, already-trained **teacher** model, instead of training the student straight from labeled data. It's normally a way to shrink a model while keeping most of its accuracy. The prior 62-class project ran this step; this project doesn't.

The prior project's distillation set was **clean EMNIST byclass only** — 697,932 samples, one narrow source. Distilled models overfit to that clean distribution and read messy real-world handwriting *worse* than their own un-distilled base versions. Distilling class `o` even made it worse outright — 49.4% base → 0.2% distilled at 32×32.

**No distillation phase runs in this project until dataset selection is resolved** — every one of the 8 ONNX exports here is a base model, trained directly on the full 439,148-sample five-source dataset, not a compressed copy of one.

---

## Eight Models, Four Optimizers, Two Resolutions

Same ten digit classes every time. What changes is the update rule, the architecture it's paired with, and the input size — each combination trained and exported independently.

### Lion — 99.49% (64×64 · 143 epochs)
*Evolved Sign Momentum · Chen et al. (Google), Feb 2023*
Uses the sign of a gradient interpolation instead of adaptive per-parameter rates. Converges to smoother minima than Adam — generally better real-world generalization, and lighter on memory since it only stores one momentum buffer.

### AdamW — 99.46% (64×64 · 119 epochs)
*Schedule-Free · Defazio et al. (Meta), May 2024*
Schedule-Free AdamW eliminates learning-rate scheduler tuning entirely. Paired with Squeeze-Excitation channel attention (Hu, Shen & Sun, 2017), which learns to amplify the feature detectors that matter most for each individual input.

### SGD — 99.49% (64×64 · 104 epochs)
*+ Nesterov momentum · Nesterov, 1983*
The classic optimizer, paired with a feature-pyramid architecture. Slowest to start — 84.97% after epoch 1 vs Lion's 94.41% — but the one whose resolution behavior this project is specifically built to interrogate.

### SOAP — 99.66% (128×128 · 49 epochs)
*Shampoo + Adam hybrid · Vyas et al., Sept 2024*
A Kronecker-factored second-order optimizer that approximates the full curvature matrix per weight tensor. Best-performing model in the entire ensemble at both resolutions.

### The Three Architectures

| Architecture | Params | Technique / provenance | Description |
|---|---|---|---|
| **OCRConvNet** | 2.46M | Depthwise-separable convs — Howard et al. (MobileNets), 2017 | Narrow, depthwise-separable convolutions split spatial and cross-channel learning into two cheaper operations — the smallest memory footprint of the three architectures. |
| **OCRConvNetWide** | 9.71M | SE attention — Hu, Shen & Sun, 2017 | Adds Squeeze-Excitation blocks after each stage for per-channel feature recalibration — the largest parameter count, built for handling structurally ambiguous classes. |
| **OCRConvNetTriple** | 4.58M (SGD) · 7.57M (SOAP) | Feature Pyramid Network — Lin et al., 2017 | Fuses pooled features from three stages before classifying, so the model sees stroke geometry, part structure, and whole-digit identity all at once. SGD and SOAP each instantiate a different-sized version of this architecture — same design, different capacity. |

---

## Tried & Dropped: AdaHessian, the 5th Optimizer

**A 5th optimizer — AdaHessian — couldn't clear both resolutions, so it didn't ship.**

Standing rule for this ensemble: **a model only ships if both its 64×64 and 128×128 versions can be trained effectively** — that's the whole premise of the resolution comparison below. AdaHessian, a second-order optimizer that estimates loss curvature (via a Hutchinson Hessian approximation) instead of just gradient direction, actually reached **99.53% test accuracy at 64×64** — competitive with, and slightly ahead of, both Lion and AdamW. But its 128×128 run couldn't clear the bar, so the whole optimizer was cut — not just the resolution that failed.

| | Time per epoch |
|---|---|
| Lion 64×64 (fastest) | ~1m 34s |
| AdaHessian 64×64 | ~18m 42s |
| AdaHessian 128×128 | ~74–75m |

Computing the Hessian estimate requires a second backward pass through the network on every step — roughly doubling (or worse) the cost of a normal training step. At 128×128 that meant a single epoch took over an hour; the 10-hour wall-clock cap every other model trains under would only fit **~8 epochs** of AdaHessian at that resolution — every other model's 128×128 run got 33 to 107 epochs, so 8 isn't enough to trust the result. It also needed a smaller batch size than every other model to avoid running out of VRAM — `512 → 256` at 64×64 after an out-of-memory failure, and a hardcoded override down to `64` at 128×128.

**Verdict:** 64×64 alone was good enough to include — but it doesn't ship without a working 128×128 counterpart, and 128×128 wasn't trainable in the same conditions every other model was held to.

---

## Why Every Model Trained at Two Resolutions

64×64 and 128×128 aren't just "more pixels for luck" — this project runs every optimizer through both as a controlled experiment, because the prior 62-class ensemble found resolution doesn't affect every optimizer the same way.

**→ SGD's two resolutions are a specific hypothesis test.**
The prior ensemble found one model — SGD-optimized — actually scored *worse* at 64×64 (76.93%) than at 32×32 (78.62%), the only optimizer where higher resolution hurt. This project reruns SGD at 64×64 and 128×128 specifically to see if that inversion is an SGD trait or was an artifact of the old 62-class problem.

**✓ The inversion reproduced.**
SGD 64×64 hit 99.49% — matching Lion exactly. SGD 128×128 came in lower, at 98.86%. The same resolution drop showed up again on a clean 10-class problem, which points toward it being a real SGD behavior rather than a 62-class artifact.

**×2 Every other optimizer ran the same 64/128 pair for comparison.**
Lion, AdamW, and SOAP all trained at both resolutions too — not because their behavior was in question, but so SGD's result has a baseline. Lion and AdamW both improved or held steady moving to 128×128; only SGD dropped, which is what makes the SGD finding a real contrast instead of noise.

---

## Full Results

| Model | Resolution | Test acc. | Epochs | Avg / epoch | Wall clock | Stop reason |
|---|---|---|---|---|---|---|
| Lion | 64×64 | 99.49% | 143 | 1m 34s | 3.76h | patience |
| Lion | 128×128 | 99.45% | 104 | 5m 48s | 10.07h | **wall-clock cap** |
| AdamW | 64×64 | 99.46% | 119 | 4m 09s | 8.24h | patience |
| AdamW | 128×128 | 99.42% | 42 | 14m 33s | 10.19h | **wall-clock cap** |
| SGD | 64×64 | 99.49% | 104 | 4m 49s | 8.35h | patience |
| SGD | 128×128 | 98.86% | 33 | 18m 29s | 10.17h | **wall-clock cap** |
| SOAP | 64×64 | 99.65% | 107 | 2m 47s | 4.96h | patience |
| SOAP | 128×128 | **99.66% (best overall)** | 49 | 10m 37s | 8.68h | patience |

Three of the eight runs — Lion 128, AdamW 128, SGD 128 — never triggered early stopping. They were still improving when the script's **10-hour wall-clock cap** ended the run at end-of-epoch. Total GPU time across the 8 shipped models: **64.42 hours**. This excludes AdaHessian — the dropped 5th optimizer above logged real GPU time of its own, but since it never made the ensemble, that time isn't counted here.

---

## From Photo to Verdict

`ocr_pipeline_mnist.py` does no post-processing and no remapping — raw inference, character by character, with disagreement made visible instead of hidden.

**Step 1 — Detect boxes.** Adaptive threshold and contours find candidate characters, clustered into lines by proximity — not a fixed grid.

**Step 2 — Rescue pass.** A box rejected only for being too wide — a stray stroke fused on — gets a second look if it fits a gap in its line.

**Step 3 — Normalize.** Each character is cropped, centered, padded, and resized to match each model's expected input size.

**Step 4 — Vote.** Every model predicts top-3. Unanimous or majority wins outright; a close split falls back to confidence-weighted scoring.

### Output Markers

- **`??`** — A character was detected and classified by every model, but they genuinely split with no majority or weighted winner.
- **`[NON-DIGIT?]`** — Every model's top confidence fell below the trust floor — likely not a digit at all, not just a hard one to read.

> Both markers are fully implemented and wired into the voting logic — neither has fired in a real test run yet. They should trigger correctly on a hard or ambiguous enough input; that just hasn't happened in testing so far.

---

*CSC-114 AI Fundamentals I · Summer 2026 · FTCC — William Edward Beckham III*
