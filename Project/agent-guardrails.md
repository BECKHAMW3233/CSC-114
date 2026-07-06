# Agent Guardrails — MNIST OCR Ensemble

## What this document is

This document defines the rules governing AI assistance (Claude) on this
project. It exists because an AI partner with no guardrails optimizes for
producing output, not for producing *correct* output that fits the actual
project. These rules keep the assistant on task and keep me in the driver's
seat.

This project has a predecessor — the EMNIST v4 62-class ensemble — whose
codebase, benchmark data, per-class accuracy tables, and findings directly
inform the MNIST work. The AI assistant has context on both. The guardrails
apply equally to both codebases: nothing in the EMNIST v4 pipeline is modified
by the AI without a specific, scoped issue authorizing it.

---

## What the AI assistant is allowed to do

- **Generate code** for training scripts, data loaders, utility functions,
  and export pipelines — provided I review and test every output before commit.
- **Debug existing code** — identify the cause of an error and propose a fix.
  I apply the fix; I do not paste unreviewed output directly into production.
- **Draft documents** — README, charter, reflection, guardrails, issue text.
  All drafts are reviewed and edited before commit.
- **Explain concepts** — optimizer behavior, architectural tradeoffs, PyTorch
  internals, second-order optimization theory. The explanation is for my
  understanding; I verify claims I am unsure about before acting on them.
- **Review code on request** — identify logic errors, missing edge cases, or
  style issues in code I wrote or code it generated.
- **Analyze benchmark and per-class accuracy results** — interpret per-class
  accuracy tables, identify failure patterns across models and resolutions,
  and suggest architectural or training changes. Analysis is advisory;
  decisions are mine. The v4 per-class baseline (all 12 ONNX models,
  established 2026-07-06) is the authoritative reference for any comparative
  analysis.

---

## What the AI assistant must never do

- **Rewrite a working file unprompted.** If a script runs correctly, the AI
  does not touch it unless I open a specific issue requesting a change. This
  applies equally to MNIST scripts and EMNIST v4 scripts.
- **Change more than one thing at a time.** Every change is scoped to a single
  issue. If the AI identifies five problems, we fix one per PR. This is the
  One Change Rule — it keeps blame isolated and rollback clean. This rule was
  the primary discipline that kept the EMNIST v4 experimental optimizer bugs
  (VRAM reference cycles in AdaHessian, CPU-bound preconditioning in SOAP)
  from contaminating other scripts when they were fixed.
- **Remove existing functionality without explicit instruction.** The AI may
  not delete dataset sources, training configurations, logging, or monitoring
  code because it judges them unnecessary.
- **Decide the project direction.** Architecture choices, dataset decisions,
  resolution targets, and optimizer selection are my decisions. The AI provides
  options and tradeoffs; it does not decide. This includes decisions about
  whether to add distillation, expand to letter classes, or change resolution
  targets — those are out of scope for this project per the charter.
- **Fabricate benchmark results or accuracy figures.** If the AI does not know
  a value, it says so. The v4 per-class accuracy tables are measured data —
  any figures cited in project documents must be traceable to those tables or
  to actual training logs. Invented numbers are not acceptable.
- **Contradict documented v4 findings without flagging it.** If the AI's
  analysis of MNIST results conflicts with a v4 finding (e.g. suggests SGD
  will scale cleanly to 128×128 when v4 documented SGD's non-monotonic
  resolution behavior), it must flag the conflict explicitly rather than
  asserting a position that contradicts the measured baseline.
- **Modify the EMNIST v4 codebase** without an explicit, scoped issue. The
  predecessor project is complete. Changes to it are not part of this
  project's scope and require their own issue and PR.

---

## How I check the AI's work

**For code:**
1. Read the diff before running anything.
2. Run the script on a short smoke test (2–3 epochs, small batch) before
   committing a full training run.
3. Confirm all output files (checkpoints, ONNX exports, CSV logs, hardware
   monitoring columns) exist and are non-zero before closing the issue.
4. For any script touching dataset loading: verify sample counts printed at
   startup match expected totals from dataset documentation.
5. For AdaHessian scripts specifically: verify VRAM is stable across epochs
   (not growing monotonically), confirming the `.grad = None` reference cycle
   fix is holding.
6. For SOAP scripts specifically: verify CUDA utilization is above 80%,
   confirming `precondition_frequency=100` is preventing the CPU-bound
   eigendecomposition issue documented in v4 development.

**For documents:**
1. Read the full draft line by line.
2. Verify any specific claims — sample counts, accuracy figures, file paths,
   benchmark results — against the actual codebase or the EMNIST v4 README
   before committing. The v4 per-class accuracy tables established 2026-07-06
   are the authoritative reference for any figures cited in comparative context.
3. Edit before commit — no document goes in verbatim.

**For benchmark and per-class accuracy analysis:**
1. Cross-reference AI interpretations against the raw per-class accuracy tables
   from the v4 README before accepting any comparative claim.
2. Do not accept causal claims about optimizer behavior (e.g. "SGD will
   perform better at 128×128") without checking them against the v4 finding
   that M3 base 64×64 accuracy was *lower* than M3 base 32×32 — SGD's
   non-monotonic resolution behavior is a documented baseline fact, not a
   hypothesis.
3. Treat AI analysis as a hypothesis to test, not a conclusion to document.

**For explanations:**
1. If the explanation contradicts my understanding or the documented v4
   findings, I ask for a source or verify independently before acting on it.
2. I treat AI explanations as a starting point for understanding, not as
   authoritative documentation.

---

## Sacred Flow compliance

The AI assistant operates within the Sacred Flow:

```
ISSUE → BRANCH → PR → SELF-REVIEW → MERGE
```

- Every piece of AI-assisted work originates from a GitHub Issue.
- AI-generated code is committed to a feature branch, never directly to main.
- The PR description documents what the AI contributed and what I changed.
- The self-review section of the PR explicitly addresses whether the AI output
  was used as-is, modified, or rejected, and why.
- Main is never committed to directly, regardless of how small the change.

---

## When the AI goes off-task

If the AI produces output that exceeds the scope of the current issue —
rewrites adjacent files, adds unrequested features, suggests expanding to
letter recognition, proposes distillation, modifies the EMNIST v4 pipeline
without authorization, or contradicts a documented v4 finding without flagging
the conflict — I:

1. Do not commit any of it.
2. Restate the issue scope explicitly and ask for a narrower response.
3. If the problem repeats, break the issue into smaller pieces until the
   output scope is controllable.

The AI works on what I asked for. Everything else gets its own issue, or does
not happen at all if it is out of scope per the charter.
