# Reflection — Module 5: Inception

## Which workflow step were you on when Housing stopped?

The California Housing work stopped partway through the *scale up and
regularize* step. The provided script ran past the validation turnaround
epoch — the point where val MAE stopped improving. That was the Module 4
hook: the code did not decide when to stop, and without a written definition
of done there was no agreed stopping point to enforce. The work was
technically complete, but it stopped because the exercise ended, not because
a pre-defined success signal was reached.

## What's different this time

The short answer: this time the finish line was written down before any
training run, and it was written using real numbers instead of guesses —
because a predecessor project had already generated those numbers.

**Where the numbers came from.** This project didn't start from zero. It
follows EMNIST v4, a 62-class character recognition ensemble I built and
completed before this module began. That project taught me three things I
could not have known in advance, and each one shaped a specific line in this
project's charter:

- v4 showed the lowercase letter cluster (o, s, c, u, l, f) failing at
  0.0–28.3% accuracy at 64×64 — a resolution problem, not a training problem.
  That's why this project's charter commits to per-class accuracy ≥98% and
  treats resolution as a first-class variable instead of an afterthought.
- v4 showed digit accuracy being dragged down by the competing demands of
  62-class letter recognition (7→9 confusion around 91% at 64×64). That's the
  direct justification for going digits-only here, and it's the reason the
  charter's 99.2% ensemble accuracy target is set where it is instead of
  being an arbitrary round number.
- v4 showed distilled models generalizing worse to real-world photos than
  base models. That's why "no distillation" is explicitly in this project's
  scope guard rather than something I'd otherwise have been tempted to add
  partway through.

**Where the discipline was different.** The actual process failure in both
Housing and EMNIST v4 was the same one: no written definition of done before
building. Housing stopped when the exercise ended. EMNIST v4 grew from a
smaller plan into a 12-model ensemble with distillation and four
experimental optimizer scripts, because each addition was individually
justifiable and nothing bounded the scope in advance. It's complete and
defensible work, but it grew that way by accident, not by design.

This project's charter writes the finish line first: specific accuracy
thresholds grounded in v4's actual measured results, and a "what we are NOT
doing" section that names five things — letter recognition, distillation,
pipeline overhaul, hyperparameter search, deployment — that would each have
been an easy, individually-reasonable thing to add. Excluding them up front
is the only thing standing between this project and becoming "EMNIST v5" by
accident, the same way v4 grew past its original plan.

**What I'd do differently from the start.** Write the "what we are NOT
doing" section before opening the first issue, not after. EMNIST v4 has a
complete, well-documented result, but the scope was never bounded until
after most of the growth had already happened. A charter written on day one
would have let that project close on its own terms instead of accumulating
indefinitely. That's the habit this module is actually teaching, and it's
the one I'm applying going forward rather than retrofitting after the fact.
