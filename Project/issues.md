# Issue Backlog — Module 5 (Walking Skeleton for Module 6)

These five issues define the first working version of the MNIST OCR ensemble
end-to-end. The goal of Module 6 (Iteration 1) is not a good model — it is
a pipeline that runs completely from raw data to ONNX output, even if the
accuracy is poor. Each issue is one branch, one PR, one self-review, one merge.

Context: the EMNIST v4 predecessor project proved the training and distillation
pipeline works end-to-end. Per-class accuracy was established across all 12 v4
ONNX models on 2026-07-06. These issues are confirmation that the MNIST-adapted
pipeline runs correctly on the new dataset configuration before committing to
full multi-day training runs.

---

## Issue 1 — Verify digit dataset pipeline end-to-end

**Goal:** Confirm that all digit supplementary datasets load correctly via the
updated `supplementary_data.py` (NUM_CLASSES=10, DIGIT_BOOST=1.0) and that the
combined training set assembles without errors.

**Acceptance criteria:**
- `download_datasets.py` completes without errors.
- `supplementary_data.py` loads all digit sources (MNIST, EMNIST Digits, USPS,
  SVHN, ARDIS IV) and prints correct sample counts matching documented totals.
- Letter-only datasets (Kaggle A-Z, Chars74K, PG-HWLD, EMNIST Balanced) are
  confirmed excluded — flags set to False, confirmed by print output showing
  they are not loaded.
- `NUM_CLASSES = 10` and `DIGIT_BOOST = 1.0` confirmed in
  `supplementary_data.py` — no 62-class weighting, no letter boost.
- Any missing dataset produces a graceful skip warning, not a crash.
- Console output captured to log file via `_Tee` logging.

**Definition of done:** Script runs to completion; sample counts printed and
match expectations; no letter datasets loaded; no crashes.

---

## Issue 2 — Smoke test: Model 1 (Lion, 64×64) for 3 epochs

**Goal:** Confirm that `ocr_pytorch_model.py` trains for 3 epochs at 64×64
on the combined digit dataset without crashing, producing a checkpoint, CSV
log, and ONNX export.

**Acceptance criteria:**
- Batch size auto-detection runs and selects a valid batch size.
- 3 training epochs complete; train loss and val loss print each epoch with
  hardware stats (VRAM, CUDA %, GPU temp, CPU %, RAM).
- `v4_model1_best_model_64.pt` exists and is non-zero.
- `v4_model1_ocr_mod_64.onnx` exports and passes `onnx.checker`.
- `v4_model1_training_log_64.csv` contains 3 rows with all hardware stat
  columns populated.
- Val accuracy after 3 epochs is a real number (not NaN). For reference:
  EMNIST v4 M1 64×64 epoch-1 val_acc was ~70% on 62 classes. MNIST at 10
  classes with 430,000 training samples should converge substantially faster.
  If 3-epoch val_acc is below 90%, investigate dataset loading and class
  weighting before proceeding.

**Definition of done:** All output files exist; no runtime errors; ONNX checker
passes; hardware monitoring columns populated in CSV.

---

## Issue 3 — Smoke test: Experimental scripts (AdaHessian 64, SOAP 64)

**Goal:** Confirm that both experimental optimizer scripts run end-to-end for
3 epochs on the MNIST dataset without crashing. These scripts were debugged
during EMNIST v4 development — two specific bugs were found and fixed:
(1) VRAM reference cycle in AdaHessian from retained computation graph, fixed
by explicitly nulling `.grad` after each step; (2) CPU-bound preconditioning
in SOAP at `precondition_frequency=10`, fixed by raising to 100. This issue
confirms both fixes hold on the MNIST dataset configuration.

**Acceptance criteria:**
- `ocr_adahessian_64.py` completes 3 epochs. VRAM does not grow
  monotonically across steps — confirmed by comparing VRAM at start of epoch 1
  vs end of epoch 3. ONNX export passes `onnx.checker`.
- `ocr_soap_64.py` completes 3 epochs. CUDA utilization confirmed above 80% —
  not CPU-bound at `precondition_frequency=100`. ONNX export passes checker.
- Both scripts produce timestamped log files capturing stdout and stderr.

**Definition of done:** Both scripts complete 3 epochs; ONNX files exist and
validate; VRAM stable for AdaHessian; CUDA utilization above 80% for SOAP.

---

## Issue 4 — Baseline accuracy: 1-epoch full run, Model 1 64×64

**Goal:** Establish a documented baseline — what accuracy does Model 1 achieve
after one full training epoch on the complete ~430,000-sample combined dataset?
This is the "make it run, badly" milestone for Module 6. It also provides the
first evidence on whether MNIST at 10 classes converges faster than EMNIST at
62 classes, and whether the digit-only training distribution produces cleaner
class separation on the per-digit accuracy breakdown.

**Acceptance criteria:**
- One full epoch completes on the combined dataset.
- Train accuracy, val accuracy, and epoch time logged to CSV.
- Hardware stats (VRAM, GPU temp, CPU %, RAM) logged for the full epoch.
- Val accuracy reported and recorded. If val_acc is below 85% at epoch 1,
  investigate dataset loading and class weighting — EMNIST v4 M1 reached
  ~70% at epoch 1 on 62 classes; MNIST at 10 classes is expected to exceed
  that substantially.
- Per-digit breakdown not required at this stage — overall val_acc is
  sufficient for the baseline.

**Definition of done:** Full epoch completes without interruption; CSV log row
is present with all columns populated; val accuracy is a real number with
documented value for project records.

---

## Issue 5 — Repository structure and module5/ artifacts committed

**Goal:** Stand up the project repository with all Module 5 deliverables
committed under `module5/` and the project README updated to reflect the MNIST
pipeline accurately.

**Acceptance criteria:**
- `module5/charter.md` committed — includes project context section
  documenting the pivot from EMNIST v4 and all four empirical findings that
  motivated it, with specific accuracy figures from the v4 per-class baseline
  (established 2026-07-06, all 12 models).
- `module5/agent-guardrails.md` committed — includes EMNIST v4 context,
  v4-specific checking procedures for AdaHessian VRAM stability and SOAP
  CUDA utilization, and the rule prohibiting AI from contradicting documented
  v4 findings without flagging the conflict.
- `module5/reflection.md` committed — includes all four v4 findings with
  specific accuracy figures, the SGD cross-resolution inversion finding, and
  the distillation generalization regression finding.
- `module5/issues.md` (this document) committed.
- Root `README.md` describes the MNIST OCR project accurately: all 8 training
  scripts, the 5 digit supplementary datasets, resolution coverage table
  (13 models target, 256×256 OOM caveat documented), normalization convention,
  and output paths.
- All commits on a branch; merged to main via PR with written self-review
  documenting what was reviewed and confirmed before merge.

**Definition of done:** All files present under `module5/`; README reflects
current project state; PR merged with documented self-review.
