# web-csc114-agent

**Platform:** platform.claude.com (Managed Agents API)
**Course:** CSC-114 Artificial Intelligence I — FTCC Summer 2026
**Model:** claude-sonnet-4-6
**Agent ID:** *(populate after creation)*

---

## Purpose

Generic multi-mode course agent. The YAML and system prompt never change.
What changes per session is which files are mounted. The agent reads whatever
is loaded and scopes all responses to that content only.

---

## File Mounting Strategy

| Use Case | Files to Mount |
|---|---|
| Single module study | `chapter-N.md` for that week |
| Cumulative course guide | All chapter files completed so far |
| Full course reference | All chapter files + `CSC114_Syllabus.md` |
| Assignment prep | Relevant chapter file(s) + `CSC114_Syllabus.md` |

Files must be uploaded via the Files API (programmatic upload) and mounted
at session creation. The agent auto-discovers files at `/mnt/session/uploads/`.

---

## Operating Modes

All modes are triggered automatically by the user's input. No special commands
required, though explicit triggers work too.

### 1. Reference Mode (default)
Ask any question. Agent answers from mounted files only. Cites source file
and section. Applies SCOPE RULE if answer is not in the files.

**Example:** "What is a tensor?"

### 2. Vocabulary Drill Mode
Agent pulls terms from mounted chapter files and runs a Q&A loop.
Tracks score for the session.

**Trigger:** "quiz me", "drill vocab", "test me on chapter 2"

### 3. Cross-Chapter Comparison Mode
Requires two or more chapter files mounted. Agent synthesizes connections
across chapters on request.

**Trigger:** "how does chapter 3's PyTorch section connect to chapter 2's tensor ops?"

### 4. Assignment Prep Mode
Requires syllabus file mounted alongside chapter file(s). Agent cross-references
chapter content against assignment descriptions and returns prioritized deliverable list.

**Trigger:** "what do I need from this chapter for the module 3 assignment?"

### 5. Lab Checklist Mode
User describes their notebook/lab steps. Agent compares against chapter content
and identifies missing or out-of-order steps.

**Trigger:** "I did X then Y then Z — what am I missing?"

### 6. Reflection Question Generator Mode
Agent generates 3–5 short-answer and applied reasoning questions from the
mounted chapter file in FTCC CSC-114 assignment style.

**Trigger:** "generate questions for chapter 2", "give me practice questions"

### 7. Notebook / Code Support Mode
User pastes code or error. Agent explains in context of the mounted chapter
material. Does not debug general Python errors unrelated to chapter content.

**Trigger:** paste a code block or error message

---

## Out-of-Scope Behavior

If the answer is not in any mounted file, the agent responds:

> "That is not covered in the currently loaded materials. Contact your instructor:
> Mallory Milstead (milsteam@faytechcc.edu) or Andrew Norris (norrisa@faytechcc.edu)."

No speculation. No general knowledge fill-in.

---

## Module File Naming Convention

Consistent naming makes multi-file sessions predictable:

```
chapter-1.md
chapter-2.md
chapter-3.md
...
CSC114_Syllabus.md
```

---

## Version History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-06-17 | Initial build — 7 modes, generic file-reference base |
