# Module 1 — Claude Projects as a Knowledge Platform

**Course:** CSC-114 Artificial Intelligence I
**Student:** William Beckham
**Institution:** Fayetteville Technical Community College
**Due:** 2026-06-07

---

## Overview

This module covers building and configuring AI agents using Claude Projects.
Two agents were built: a course information bot (CSC-114-InfoBot) using
claude.ai, and a Security+ exam prep agent (SecPlus-Bot) using the
platform.claude.com Managed Agents API.

---

## Folder Structure

```
module_1/
├── README.md                        ← this file
├── projects-vs-platform.md          ← comparison of claude.ai vs platform.claude.com
├── system-prompt-v1.md              ← CSC-114-InfoBot system prompt versions
├── testing-log.md                   ← CSC-114-InfoBot session test results
└── Project/
    ├── web-secplus-bot.yaml         ← agent configuration (deployed to platform.claude.com)
    ├── custom-instructions.md       ← three-question setup per lab spec
    ├── system-prompt-v1.md          ← SecPlus-Bot prompt version history and deployment notes
    ├── testing-log.md               ← SecPlus-Bot test results + setup notes
    ├── session-events-sesn_01CGfjAYcjAR1hwyZZoN8fEn.json  ← raw session log for verification
    └── notes/
        ├── Section 1_ Introduction.md
        ├── Section 2_ Fundamentals of Security.md
        ├── Section 3_ Threat Actors.md
        ├── Section 4_ Physical Security.md
        ├── Section 5_ Social Engineering.md
        ├── Section 6_ Malware.md
        ├── Section 7_ Data Protection.md
        └── Section 8_ Cryptographic Solutions.md
```

---

## Session 0 — CSC-114-InfoBot (claude.ai)

A Claude Project loaded with the CSC-114 syllabus, configured to answer
student questions using only that material and to refuse to fabricate
information not covered by the syllabus.

**Files:**
- `projects-vs-platform.md` — builder comparison between claude.ai and platform.claude.com
- `system-prompt-v1.md` — system prompt versions 1 and 2 with change notes
- `testing-log.md` — three test results including token count data from the Debug tab

---

## Session 1 — SecPlus-Bot (platform.claude.com)

A Managed Agent configured for CompTIA Security+ SY0-701 exam preparation,
loaded with 8 study note files covering the five exam domains. Files were
uploaded via the Files API (programmatically — the console UI does not
support direct upload) and mounted into the session at `/uploads/`.

**Agent:** `web-secplus-bot`
**Model:** claude-sonnet-4-6
**Knowledge base:** 8 section files covering Domains 1.0–5.0

**Files:**
- `web-secplus-bot.yaml` — full agent configuration including system prompt
- `custom-instructions.md` — three-question setup (who are you / who am I / what will we accomplish)
- `system-prompt-v1.md` — YAML version history with notes on what broke and what fixed it
- `testing-log.md` — three test results, setup notes, and self-recovery observation from Test 3
- `session-events-*.json` — raw event log from the live session for verification
- `notes/` — the 8 source files uploaded to the Files API as the agent knowledge base

---

## Key Takeaways

The file upload process on platform.claude.com required programmatic API
access, critical thinking to diagnose filename mismatches and missing toolset
configuration, and multiple iterations to get working. Once the setup was
correct the agent performed reliably across all three test types. The
`agent_toolset_20260401` tool must be enabled for file resources to be
readable in a session — without it the session fails with a missing tool error.
