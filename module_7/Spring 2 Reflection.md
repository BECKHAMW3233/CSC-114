# Spring 2 Reflection — Check-In 2: Is It Getting Better?

**Project:** MNIST digit OCR — multi-optimizer, multi-resolution CNN ensemble
**Course:** CSC-114, FTCC

---

## Before / after

**Before (Check-In 1):** Ran `ocr_pipeline_mnist.py --model-dir E:\CSC-114\project`
to test the ensemble against real photos. The scan recursively walked the
entire project folder with no exclusions, which swept up my Python venv --
including hundreds of ONNX's own internal unit-test files (many literally
named `model.onnx`) plus unrelated model-zoo files that had ended up under
that tree. Result: it tried to load **1925 "models"**, most of them garbage
(random tiny shapes, non-digit classifiers, a few that flat-out failed to
load), and the ensemble output was meaningless.

**After:** Added an exclusion list (`venv`, `.venv`, `site-packages`,
`__pycache__`, `.git`, etc.) that prunes those directories from the walk
before it ever descends into them. Re-ran the exact same command. It now
correctly finds and loads **exactly the 8 models I actually trained**, and
the ensemble produces real, readable per-character predictions and
consensus votes instead of noise.

## Written reflection

**What I changed:** Fixed a bug in `ocr_pipeline_mnist.py`'s `--model-dir`
scanner so it skips `venv`/`site-packages`/etc. instead of recursively
loading everything under the project folder.

**Why I thought it would help:** The 1925-model run made it obvious the
scanner had no folder exclusions at all -- it was doing exactly what I told
it to do (walk everything, load every `.onnx` file it finds), just not what
I meant. Excluding the folders that aren't part of my actual model output
was the direct fix for the direct cause.

**Did it actually help? How do I know:** Yes, confirmed by comparing the two
log files directly. The first log shows 1925 models loaded with dozens of
load failures and garbage predictions mixed into the ensemble vote. The
second log shows exactly 8 models loaded (`adamw_64`, `adamw_128`, `lion_64`,
`lion_128`, `sgd_64`, `sgd_128`, `soap_64`, `soap_128` -- the full set), and
for the first time I was able to actually score model output against real
ground truth: across 7 test photos with known-correct answers (69 digits
total), the ensemble's individual models scored 73.9%-97.1% accuracy. That
scoring wasn't even possible before the fix, since the output was drowned in
noise from 1917 irrelevant models.

## What's next

Now that real ground-truth scoring is working, the clearest next step is
figuring out why `lion_128` is the one model that scored meaningfully worse
than the rest (73.9% vs. 91-97% for everyone else) despite having fine
test-set numbers during training -- that's a real, specific problem the fix
this week made visible for the first time.

---

## Talking points

| | |
|---|---|
| **What changed** | Fixed `--model-dir` to exclude `venv`/`site-packages`/etc. instead of recursively scanning everything under the project folder. |
| **What happened** | Went from loading 1925 garbage "models" (mostly ONNX's own test fixtures) to correctly loading exactly my 8 trained models. |
| **Was it worth it** | Yes -- it's what made real ground-truth scoring possible at all this week (69 digits scored, 73.9%-97.1% accuracy range across the 8 models). |
| **What's next** | Investigate why lion_128 underperforms the other 7 models by ~20 points on real-world images despite normal test-set accuracy. |
