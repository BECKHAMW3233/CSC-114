# Reflection — Module 5: Inception

## Which workflow step were you on when Housing stopped?

The California Housing work stopped partway through the *scale up and
regularize* step — Module 7 territory in the current arc. The provided script
ran past the validation turnaround epoch, which is the Module 4 hook: the code
did not decide when to stop, and without a written definition of done there was
no agreed stopping point to enforce. The work was technically complete, but it
stopped because the exercise ended, not because a pre-defined success signal
was reached.

---

## What is different this time

### Where this project actually started

This project did not begin at Module 5. It began in June 2026 as a
self-directed research effort: a 12-model deep learning ensemble for
handwritten character recognition across 62 classes — digits, uppercase
letters, and lowercase letters. That predecessor project (EMNIST v4) completed
training, distillation, ONNX export, per-class accuracy analysis, and initial
real-world benchmark testing before this module began. The per-class baseline
was fully established on 2026-07-06 when `ocr_class_accuracy.py` was run
against all 12 v4 ONNX models — 6 base and 6 distilled, both resolutions —
on the school machine (i7-10700, CPU-only inference). Module 5 is the point
where I am retrofitting formal process scaffolding onto work that was already
in motion, and using what that work taught me to scope the next one correctly.

### The pivot — and why it matters methodologically

The MNIST digits-only ensemble documented in this charter is not the same
project as EMNIST v4. It is a deliberate pivot driven by four specific
empirical findings from v4, all confirmed with measured data.

**Finding 1 — The lowercase ambiguity cluster is an architecture-level failure,
not a training-level failure.**

The same six classes fail across every model and every resolution: o, s, c, u,
l, f. At 64×64, M1 distilled reaches o=0.9%, s=1.4%, c=1.2%. M3 distilled
reaches o=0.0%, s=0.0%, c=10.9%. M2 distilled — the widest architecture at
9.7M parameters — produces the best results and still only reaches o=4.9%,
s=11.7%, c=23.4%. The post-processing pipeline (q→a remap, split rescue,
spatial override) was specifically built to compensate for these failures and
achieved 15/15 on the mixed benchmark image — but fails under the stress test
scenario where all characters are drawn from the failure cluster simultaneously.
The averaged soft labels from teachers who both misclassify o reinforced rather
than corrected the confusion: M1 base 49.4% → M1 distilled 0.2% at 32×32.
Distillation made o *worse* for M1. This is a visual ambiguity problem
requiring higher resolution and stroke-endpoint detection, not more training data
or better post-processing. It cannot be fixed within the 64×64 constraint.

**Finding 2 — SGD produces a fundamentally different error geometry at 64×64,
and not a better one.**

M3 base is the only model where 64×64 overall accuracy (76.93%) is lower than
32×32 (78.62%). At 64×64, M3 base routes S to incorrect classes at 1.9%
accuracy — down from 7.8% at 32×32. O drops from 8.3% to 2.1%. M1 and M2
both improve at every resolution increase; M3 regresses on its hardest classes.
The M3 distilled models partially compensate via soft label transfer from M1+M2
teachers, but o and s remain at 0.0% in M3 distilled 64×64 — SGD's learned
weight geometry for those classes cannot be corrected by distillation after the
fact. This finding carries directly into the MNIST project: Model 3 uses SGD,
and SGD has a documented tendency to find 64×64 minima that generalize worse
than its 32×32 minima on structurally ambiguous classes. It is the
highest-risk optimizer for the 128×128 runs and its per-class accuracy will be
monitored against this baseline specifically.

**Finding 3 — Distilled models trained on clean data generalize worse to
real-world photos than base models trained on noisy multi-source data.**

On every real-world benchmark image, the distilled models produced more `?`
outputs and more off-label predictions (N, D, U, P) than the base models at
the same resolution. The base models — particularly M2 and M3 at 64×64 — were
the cleanest raw readers on real handwriting. Distillation trained on EMNIST
byclass alone (697,932 clean, controlled, normalized samples) produces models
that have overfit to that distribution. The base models trained on 11 sources
including SVHN, Chars74K, and USPS generalize better to real photographs.
This is a documented finding for the paper and directly gates the MNIST project:
no distillation phase until the distillation dataset selection problem is solved.
Building a 13-model distilled ensemble on top of MNIST base models would
reproduce the same generalization regression at higher resolution.

**Finding 4 — Digit accuracy within the 62-class ensemble is constrained by
letter recognition demands.**

DIGIT_BOOST=3.0x weighting, the 11-source dataset composition, and architecture
choices were all calibrated to balance 62 classes simultaneously. v4 stress
tests confirmed 7→9 confusion on hooked 7 variants (~91% accuracy) and 8→9/d
confusion on open-top 8 variants (~90.5%) at 64×64. These are not model-level
ceiling effects — they are constraints imposed by the 62-class competing
objective. A digits-only pipeline removes those constraints entirely and trains
on ~430,000 samples all labeled 0–9, with no competing letter class demands.

### What I skipped — and what the charter does now

The core process failure in both Housing and EMNIST v4 was identical: no
written definition of done before building. Housing stopped when the exercise
ended. EMNIST v4 grew from a 3-model ensemble at one resolution to a 12-model
ensemble at two resolutions with distillation, five optimizer families, four
experimental scripts, a per-class diagnostic tool, and a 202-image benchmark
suite in progress — because each addition was individually justifiable and
nothing bounded the scope in advance.

This charter writes the finish line before any MNIST training runs begin:
ensemble test accuracy ≥ 99.2%, real-world benchmark ≥ 93.7% on ≥ 200
characters, per-class accuracy ≥ 98.0% on every digit class, all successfully
trained ONNX models export and validate. Those numbers are specific,
falsifiable, and grounded in measured v4 benchmark data — not aspirational
targets. The 93.7% figure is the EMNIST v4 real-world benchmark result. The
98.0% per-class floor is calibrated to close the specific 7→9 and 8→9
failure modes documented in v4 stress testing.

The "What we are NOT doing" section exists because every excluded item —
letter recognition, distillation, pipeline overhaul, hyperparameter search,
deployment — would have been individually justifiable additions to EMNIST v4
with no scope guard to stop them. The guard is what prevents this project from
becoming EMNIST v5 by accident.

### What the formal scaffolding does

The issue backlog, guardrails doc, and this reflection are not bureaucracy
retrofitted onto finished work. They are the professional documentation layer
that turns a collection of training scripts and benchmark results into a project
someone else could understand, reproduce, and build on. Instructors Milstead
and Norris are directly aware of the work and have engaged with it as a
potential publication-track project targeting arXiv and ICDAR. The formal
structure this module requires is exactly what that publication track needs: a
clear problem statement with predecessor context, measurable success criteria
grounded in prior results, explicit scope boundaries with reasons, and a
documented record of how decisions were made.

### What I would do differently from the start

Write the charter first — specifically the scope guard — before opening the
first issue. The EMNIST v4 project has a complete README with version history,
errata, per-class accuracy tables, benchmark data, and a hardware feasibility
analysis. All of it was earned by doing the work. But the scope was never
bounded in advance, and the result is a project that is complete and defensible
but open-ended by construction: there is always another resolution, another
optimizer, another dataset source, another post-processing fix. A charter
written before the first training run would have drawn the line earlier and let
the project close on its own terms.

That is the lesson Module 5 is actually teaching, and I understand it now in
a way I did not when EMNIST v4 began.
