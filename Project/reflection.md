# Reflection — Module 5: Inception

## Which workflow step were you on when Housing stopped?

The California Housing work stopped partway through the *scale up and
regularize* step — Module 7 territory in the current arc. The provided script
ran past the validation turnaround epoch, which is exactly the Module 4 hook:
the code did not decide when to stop, and without a written definition of done
there was no agreed stopping point to enforce. The work was technically complete,
but it stopped because the exercise ended, not because a pre-defined success
signal was reached.

---

## What is different this time

### Where this project actually started

This project did not begin at Module 5. It began in June 2026 as a self-directed
research effort: a 12-model deep learning ensemble for handwritten character
recognition across 62 classes — digits, uppercase letters, and lowercase letters.
That predecessor project (EMNIST v4) completed training, distillation, ONNX
export, and initial real-world benchmark testing before this module began. It
has its own full README, version history, per-class accuracy tables, and
benchmark results. Module 5 is not the starting line. It is the point where I
am retrofitting formal process scaffolding onto work that was already in motion,
and using what that work taught me to scope the next one correctly.

### The pivot — and why it matters methodologically

The MNIST digits-only ensemble documented in this charter is not the same
project as EMNIST v4. It is a deliberate pivot, driven by specific empirical
findings from v4 that I could not have known before building it.

The EMNIST v4 project found three things that directly shaped the scope of
this one:

**Finding 1 — The lowercase ambiguity cluster fails systematically at 64×64.**
Across all six distilled models, per-class accuracy on o, s, c, u, l, and f
ranged from 0.0% to 28.3%. These classes fail because their distinguishing
stroke features — the internal horizontal stroke in e, the descending tail on
a vs q, the dot on i — occupy fewer pixels than the convolutional filters can
reliably detect at 64×64. Post-processing cannot fix a resolution problem.
This is documented in the EMNIST v4 README under Limitations.

**Finding 2 — Distilled models trained on clean EMNIST data generalize worse
to real-world photos than base models trained on the full 11-source dataset.**
The distilled models produced more `?` outputs and off-label predictions on
every real-world benchmark image tested. The base models — particularly M2 and
M3 at 64×64 — were the cleanest raw readers. This is a finding about
distillation dataset selection that I did not anticipate and cannot fix without
a different approach.

**Finding 3 — Digit recognition accuracy within the 62-class ensemble is
constrained by letter recognition demands.** The DIGIT_BOOST=3.0x weighting,
the 11-source dataset composition, and the architecture choices were all
calibrated to balance 62 classes. A digits-only pipeline removes those
constraints. The EMNIST v4 stress tests showed 91% accuracy on isolated
digit classes — that ceiling is not acceptable for a purpose-built digit
recognizer.

The MNIST project exists because v4 provided enough evidence to scope it
correctly. Without v4, I would not have known which resolutions to prioritize,
which optimizers to test, which datasets to include, or where the real failure
modes live.

### What I skipped — and what the charter does now

The core process failure in both Housing and EMNIST v4 was the same: no
written definition of done before building. Housing stopped when the exercise
ended. EMNIST v4 grew from a 3-model ensemble at one resolution to a 12-model
ensemble at two resolutions with distillation, five optimizer families, and
four experimental scripts — because each addition was individually justifiable
and nothing bounded the scope in advance.

This charter writes the finish line before any training runs for the MNIST
project:

- Ensemble MNIST test accuracy ≥ 99.2%
- Real-world benchmark accuracy ≥ 93.7% on ≥ 200 characters
- Per-class accuracy ≥ 98.0% on every digit class
- All 13 ONNX models export and validate

Those numbers are specific, measurable, and grounded in actual v4 benchmark
data — not aspirational targets pulled from the air. The 93.7% figure is the
EMNIST v4 real-world benchmark result. The 99.2% figure is calibrated to what
a 13-model ensemble at 64/128/256×256 should be able to achieve on clean MNIST
given v4's 100% on structured digit grids with 6 models at 32/64×64.

The "What we are NOT doing" section is the other half of that discipline. It
lists five specific things that would be individually justifiable additions —
letter recognition, distillation, pipeline overhaul, hyperparameter search,
deployment — and explicitly excludes them. Every one of those would have been
added to EMNIST v4 without a scope guard. The guard is what prevents this
project from becoming EMNIST v5 by accident.

### What the formal scaffolding does

The issue backlog, guardrails doc, and this reflection are not bureaucracy
retrofitted onto finished work. They are the professional documentation layer
that turns a collection of training scripts into a project someone else could
understand, reproduce, and build on. Instructors Milstead and Norris are
directly aware of the work and have engaged with it as a potential
publication-track project. The formal structure this module requires is exactly
what that publication track needs: a clear problem statement, a measurable
definition of success, explicit scope boundaries, and a documented record of
how decisions were made.

### What I would do differently from the start

Write the charter first. Specifically, write the "What we are NOT doing"
section before opening the first issue. The EMNIST v4 project has a complete
README with version history, errata, and benchmark data — all of it earned by
doing the work — but the scope was never bounded in advance. The result is a
project that is complete and defensible but open-ended by construction: there
is always another resolution, another optimizer, another dataset source, another
post-processing fix to try. A charter written before the first training run
would have drawn the line earlier and let the project close on its own terms
rather than accumulating indefinitely.

That is the lesson Module 5 is actually teaching, and I understand it now in
a way I did not when EMNIST v4 began.
