# Issue Backlog — Module 5 (Walking Skeleton for Module 6)

These issues frame the first fully working version of the MNIST OCR ensemble
end-to-end — training completes across the full script set, every model
exports to ONNX, and the ensemble pipeline runs against a real-world image
set with a documented result. This backlog reflects where the project
actually stands as of this module: the 64×64 tier is complete for three of
the five optimizer families, and the remaining work is what's left to reach
the walking skeleton, not a from-scratch plan.

Context: the EMNIST v4 predecessor project already proved the training and
distillation pipeline works end-to-end. These issues are not exploratory —
they confirm the adapted MNIST pipeline runs correctly across the remaining
resolutions and optimizer families before the ensemble is considered feature
complete.

---

## Issue 1 — Complete remaining 64×64 models (AdaHessian, SOAP)

**Goal:** Finish the 64×64 tier. Lion, AdamW, and SGD are complete
(99.49%, 99.46%, 99.49% test accuracy respectively). AdaHessian 64×64 and
SOAP 64×64 are the two remaining models needed before the 64×64 ensemble is
feature complete.

**Acceptance criteria:**
- `ocr_adahessian_64.py` completes a full training run (wall clock or
  patience exit), produces a valid 10-class ONNX export, and logs a
  documented test accuracy.
- `ocr_soap_64.py` completes a full training run, produces a valid ONNX
  export, and logs a documented test accuracy.
- Both checkpoints load correctly in `ocr_pipeline_mnist.py` alongside the
  existing three 64×64 models.

**Definition of done:** Five 64×64 models present, all loading and voting
correctly in the ensemble pipeline.

---

## Issue 2 — Train the 128×128 tier across all optimizer families

**Goal:** Determine whether accuracy improves at 128×128, and specifically
resolve whether SGD's non-monotonic resolution behavior (observed in EMNIST
v4) is a 62-class artifact or an SGD-fundamental behavior.

**Acceptance criteria:**
- `mnist_lion_128.py`, `mnist_adamw_128.py`, `mnist_sgd_128.py`,
  `ocr_adahessian_128.py`, and `ocr_soap_128.py` each complete a full run.
- Each produces a valid ONNX export and a documented test accuracy.
- SGD 128×128 accuracy is explicitly compared against the SGD 64×64 baseline
  (99.49%) to answer the non-monotonic resolution question.

**Definition of done:** All five 128×128 models trained and exported; SGD
resolution finding documented one way or the other.

---

## Issue 3 — Resolve 256×256 viability for Lion and SOAP

**Goal:** Lion 256×256 is architecturally confirmed viable on paper (batch
128, 15.6/16.0GB VRAM in hardware testing), so it's worth attempting — but it
is a bonus result, not a requirement. SOAP 256×256 viability is untested.
AdamW 256×256 is already confirmed infeasible and is out of scope. This
issue completes the resolution-coverage picture without blocking on it.

**Acceptance criteria:**
- `mnist_lion_256.py --batch-size 128` is attempted. If it completes and
  exports cleanly, that's a genuine win for the ensemble's resolution
  coverage. If it fails for any reason (OOM, instability, wall-clock with no
  usable checkpoint), that's documented as a finding and the walking
  skeleton proceeds without it — it does not block Issue 4 or 5.
- `ocr_soap_256.py` is attempted under the same terms; if it OOMs, that
  result is documented as a finding (per the charter's stated approach to
  256×256 risk), not treated as a failure.

**Definition of done:** Both 256×256 attempts are documented either way —
trained and exported, or documented as infeasible with the data that shows
why. Neither outcome blocks the rest of the backlog. A successful Lion
256×256 model is an ensemble bonus; its absence does not lower the project
below "good enough" as defined in the charter.

---

## Issue 4 — Real-world benchmark validation across the full ensemble

**Goal:** Run the complete set of test images (test1–test9, test21–test24,
test30–test32, and any remaining) through `ocr_pipeline_mnist.py` against
the full ensemble once Issues 1–3 are complete, and produce a documented
accuracy figure against the charter's ≥93.7% / ≥200-character target.

**Acceptance criteria:**
- All test images processed through the pipeline with the expanded ensemble.
- Ensemble accuracy calculated and compared against ground truth for every
  image.
- The test30.jpg 7→9 misread (documented in the README's Real-World Testing
  section) is specifically re-checked: does the expanded ensemble (AdaHessian
  and SOAP added, 128×128 models added) resolve it, or does it persist?
- Any phantom-detection issues (extra lines picked up by `get_boxes()`, as
  seen on test31.jpg and test32.jpg) are noted but not fixed under this issue
  — that is explicitly out of scope per the charter's "no pipeline overhaul"
  scope guard, unless a separate issue is opened for it.

**Definition of done:** Documented real-world accuracy figure for the full
ensemble; the 7→9 hypothesis confirmed or refuted with data, not assumption.

---

## Issue 5 — Repository structure and module5/ artifacts committed

**Goal:** Stand up the Module 5 repository with all deliverables committed
under `module5/` and the project README updated to reflect the MNIST
pipeline's actual current state.

**Acceptance criteria:**
- `module5/charter.md` committed.
- `module5/agent-guardrails.md` committed.
- `module5/reflection.md` committed.
- `module5/issues.md` (this document) committed.
- Root `README.md` describes the MNIST OCR project accurately: all trained
  models and their measured results, the 5 digit supplementary datasets,
  resolution coverage, normalization convention, output paths, and the
  real-world testing results to date.
- All commits are on a branch; merged to main via PR with a written
  self-review documenting what was reviewed and confirmed before merge.

**Definition of done:** All files present under `module5/`; README reflects
current, accurate project state; PR merged with documented self-review.
