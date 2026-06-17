# web-csc114-agent

**Platform:** platform.claude.com — Managed Agents API
**Course:** CSC-114 Artificial Intelligence I — FTCC Summer 2026
**Model:** claude-sonnet-4-6
**Agent ID:** *(populate after creation on platform)*

---

## Architecture

The YAML and system prompt are fixed for the entire course. Nothing changes
between modules. What changes per session is which files are mounted.

**Embedded in the agent (no file required):**
- Full assignment schedule with due dates and point values
- Instructor and escalation chain contacts with office locations
- Late work and academic integrity policies
- Chapter-to-assignment mapping for all modules
- Out-of-scope redirect behavior

**Supplied via mounted files:**
- Chapter content, vocabulary, technical concepts (Ch 1–9 only)

The agent is functional with zero files mounted — it answers any policy,
schedule, contact, or assignment-mapping question without a file. Content
modes activate when the relevant chapter file is loaded.

---

## Chapter-to-Assignment Mapping

| Assignment | Due | Chapter(s) |
|---|---|---|
| Create a Custom Agent | 6/07 | Ch 1 |
| Apply/Assess AI Frameworks | 6/21 | Ch 3 |
| Apply/Assess Classification & Regression | 6/28 | Ch 4 |
| Apply/Assess Machine Learning Workflow | 7/05 | Ch 5 + Ch 6 |
| Apply/Assess Computer Vision | 7/12 | Ch 8 + Ch 9 |
| Apply/Assess NLP and LLMs | 7/19 | Ch 13 + Ch 14 |
| Final Project | 7/20 | Ch 1–9 cumulative |

Ch 2 (tensors, gradient descent, backpropagation) is foundational to all
modules. Chapters 7, 10–20 are out of scope for this course.

---

## File Mounting Strategy

| Session Purpose | Mount These Files |
|---|---|
| Single module study | `chapter-N.md` for that week |
| Cumulative course guide | All chapter files completed to date |
| Assignment prep | Relevant chapter file(s) — mapping is embedded |
| Full course reference | chapter-1.md through chapter-9.md |

Files must be uploaded via the Files API (Python script).
Platform UI does not support direct session file upload.
Agent auto-discovers files at `/mnt/session/uploads/`.

**File naming convention:**
```
chapter-1.md
chapter-2.md
chapter-3.md
chapter-4.md
chapter-5.md
chapter-6.md
chapter-7.md   ← deep Keras dive; in book but low course priority
chapter-8.md
chapter-9.md
```

---

## Operating Modes

| Mode | Trigger | Requires |
|---|---|---|
| Reference | Any question | None (embedded) or chapter file |
| Vocabulary Drill | "quiz me", "drill vocab", "test me" | Chapter file |
| Cross-Chapter Comparison | Compare concepts across chapters | 2+ chapter files |
| Assignment Prep | "what do I need for [assignment]" | Chapter file (mapping is embedded) |
| Lab Checklist | Describe completed steps / "what am I missing" | Chapter file |
| Reflection Question Generator | "generate questions for chapter N" | Chapter file |
| Notebook / Code Support | Paste code or error message | Chapter file |

---

## Out-of-Scope Behavior

> "That is not covered in the currently loaded materials. Contact Milstead
> (milsteam@faytechcc.edu) or Norris (norrisa@faytechcc.edu) directly."

---

## Version History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-06-17 | Initial build — 7 modes, syllabus + Ch 1–9 mapping embedded |
