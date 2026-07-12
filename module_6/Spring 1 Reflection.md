# Spring 1 Reflection — Check-In 1: Is It Alive?

**Project:** MNIST digit OCR — multi-optimizer, multi-resolution CNN ensemble
**Course:** CSC-114, FTCC

---

## What I'm building

A digits-only (0-9) handwritten OCR pipeline that trains the same recognition
task across four different optimizer families (Lion, Schedule-Free AdamW,
SGD, SOAP) at two resolutions each, to compare how optimizer choice and input
resolution affect handwriting recognition accuracy.

## What runs today

- All 8 training scripts (Lion 64/128, AdamW 64/128, SGD 64/128, SOAP 64/128)
  run independently, each pulling from a combined ~439K-sample digit dataset
  (MNIST + EMNIST Digits + USPS + SVHN + ARDIS IV), training with automatic
  batch-size detection, hardware telemetry logging, checkpointing, and
  resume-from-crash support.
- Every model has completed training, exported to ONNX, and been validated
  on its held-out test split. Test accuracies range from 98.86% (SGD
  128x128) to 99.66% (SOAP 128x128).
- `ocr_pipeline_mnist.py` takes a real input (a photo of handwritten digits)
  through to a real output: it segments characters using classical CV
  (adaptive thresholding, contour detection, box merging), runs them through
  all 8 trained ONNX models, and returns a per-character ensemble vote with
  confidence scores.
- I ran the full 8-model pipeline against 14 real photos of handwritten
  digit sheets (not test-set images) taken on my phone. Seven of those
  photos (test1-test7) are reference sheets where I actually know the
  correct answer, so I scored each of the 8 models individually against
  ground truth, character by character (69 known digits total):

  | Model | Real-world accuracy |
  |---|---|
  | soap_128 | 97.1% |
  | soap_64 | 97.1% |
  | adamw_64 | 95.7% |
  | sgd_128 | 95.7% |
  | sgd_64 | 95.7% |
  | adamw_128 | 92.8% |
  | lion_64 | 91.3% |
  | lion_128 | 73.9% |

So: input to output runs end to end, on both the held-out test split and on
real photos I took myself, with actual scored accuracy numbers, not just
"the pipeline ran."

## What's still missing or broken

- Found a real, scored problem: `lion_128` is a clear outlier at 73.9%
  real-world accuracy, roughly 20 points behind every other model (which
  cluster at 91-97%). It never once posted the best score of the 8 models on
  any of the 7 scored images, and dropped as low as 50% on one pencil-written
  sheet. Its test-set accuracy during training was fine, so this looks like
  a real generalization gap specific to Lion at 128x128, not a training
  failure -- next step is figuring out why.
- The other 7 real-world photos (test21-test24, test30-test32) don't have an
  obvious known-correct answer to score against yet, so I only have
  model-agreement (consensus) numbers for those, not real accuracy. Those
  ranged from 32% to 95% agreement depending on how dense/cluttered the
  sheet was, but I can't yet say whether low agreement means wrong answers
  or just messier box detection on cluttered sheets.
- I also caught and fixed a real bug this week: `--model-dir` was scanning
  recursively with no exclusions, which swept up my entire Python venv
  (including hundreds of onnx's own internal test files) into the "ensemble"
  -- 1925 models instead of my actual 8. Fixed by excluding `venv`,
  `site-packages`, and similar folders from the scan.
- I originally scoped a 5th optimizer (AdaHessian, a second-order method)
  into the comparison. I trained it at 64x64 (99.53% test accuracy) but cut
  it from the final rotation -- noted under scope below.

## Am I still on track with my charter?

Yes. The core comparison (optimizer x resolution) is fully trained,
evaluated on the test split, and now also scored against real-world ground
truth for part of the data. The one real scope change: I dropped AdaHessian
from the final comparison after training it partially, to keep the
comparison to optimizer families I could fully complete at both resolutions
within the time I have.

---

## Talking points

| | |
|---|---|
| **What I'm building** | A digit-OCR pipeline comparing 4 optimizers x 2 resolutions to see how optimizer choice affects handwriting recognition accuracy. |
| **What runs today** | All 8 models are trained, ONNX-exported, and validated (98.86%-99.66% test accuracy). Scored against real-world photos: SOAP leads at 97.1%, lion_128 lags badly at 73.9%. |
| **What's broken/missing** | lion_128 has a real ~20-point real-world accuracy gap I don't understand yet. Also still need ground truth for 7 of the 14 real-world photos. |
| **What's next** | Dig into why lion_128 specifically underperforms in the real world despite fine test-set numbers, and get ground truth for the remaining photos. |
