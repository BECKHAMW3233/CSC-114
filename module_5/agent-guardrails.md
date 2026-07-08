# Agent Guardrails — MNIST OCR Ensemble

## What this document is

This document defines the rules governing AI assistance (Claude) on this
project. It exists because an AI partner with no guardrails optimizes for
producing output, not for producing *correct* output that fits the actual
project. These rules keep the assistant on task and keep me in the driver's
seat.

This project has a predecessor — the EMNIST v4 62-class ensemble — whose
codebase, benchmark data, and findings directly inform the MNIST work. The AI
assistant has context on both. The guardrails apply equally to both codebases:
nothing in the EMNIST v4 pipeline is modified by the AI without a specific,
scoped issue authorizing it.

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
- **Analyze benchmark results** — interpret per-class accuracy tables, identify
  failure patterns, and suggest architectural or training changes. Analysis is
  advisory; decisions are mine.

---

## What the AI assistant must never do

- **Rewrite a working file unprompted.** If a script runs correctly, the AI
  does not touch it unless I open a specific issue requesting a change. This
  applies equally to MNIST scripts and EMNIST v4 scripts.
- **Change more than one thing at a time.** Every change is scoped to a single
  issue. If the AI identifies five problems, we fix one per PR. This is the
  One Change Rule — it keeps blame isolated and rollback clean.
- **Remove existing functionality without explicit instruction.** The AI may
  not delete dataset sources, training configurations, logging, or monitoring
  code because it judges them unnecessary.
- **Decide the project direction.** Architecture choices, dataset decisions,
  resolution targets, and optimizer selection are my decisions. The AI provides
  options and tradeoffs; it does not decide. This includes decisions about
  whether to pivot back to 62 classes, add distillation, or expand scope —
  those are out of scope for this project per the charter.
- **Fabricate benchmark results or accuracy figures.** If the AI does not know
  a value, it says so. Invented accuracy numbers, parameter counts, or dataset
  sizes are not acceptable in any project document — especially given that this
  project has real measured benchmark data from EMNIST v4 and from the MNIST
  training runs themselves that serve as the baseline.
- **Modify the EMNIST v4 codebase** without an explicit, scoped issue. The
  predecessor project is complete. Changes to it are not part of this project's
  scope and require their own issue and PR.

---

## How I check the AI's work

**For code:**
1. Read the diff before running anything.
2. Run the script on a short smoke test (2–3 epochs, small batch) before
   committing a full training run.
3. Confirm all output files (checkpoints, ONNX exports, CSV logs) exist and
   are non-zero before closing the issue.
4. For any script touching dataset loading: verify sample counts printed at
   startup match expected totals from dataset documentation.

**For documents:**
1. Read the full draft line by line.
2. Verify any specific claims — sample counts, accuracy figures, file paths,
   benchmark results — against the actual codebase, training logs, or the
   EMNIST v4 README before committing.
3. Edit before commit — no document goes in verbatim.

**For benchmark analysis:**
1. Cross-reference AI interpretations against the raw per-class accuracy
   tables and pipeline logs.
2. Do not accept causal claims (e.g. "SGD underperforms because of X") without
   checking them against documented errata.
3. Treat AI analysis as a hypothesis to test, not a conclusion to document —
   confirmed in practice with the AdaHessian/SOAP/128×128 hypothesis for the
   test30.jpg 7→9 misread, which is documented as a prediction to verify, not
   a settled fact.

**For explanations:**
1. If the explanation contradicts my understanding or documented findings, I
   ask for a source or verify independently before acting on it.
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
letter recognition, proposes distillation, or modifies the EMNIST v4 pipeline
without authorization — I:

1. Do not commit any of it.
2. Restate the issue scope explicitly and ask for a narrower response.
3. If the problem repeats, break the issue into smaller pieces until the
   output scope is controllable.

The AI works on what I asked for. Everything else gets its own issue, or does
not happen at all if it is out of scope per the charter.
