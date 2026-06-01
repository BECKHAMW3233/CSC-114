# Module 1 Testing Log — SecPlus-Bot

**Platform:** platform.claude.com (Managed Agents)
**Model:** claude-sonnet-4-6
**Agent name:** web-secplus-bot
**Date tested:** 2026-06-01
**Session ID:** sesn_01CGfjAYcjAR1hwyZZoN8fEn

---

## My three tests

| # | Type | Question I asked | What I expected | What actually happened | Pass / Fail |
|---|------|------------------|-----------------|------------------------|-------------|
| 1 | Known good | "What is the CIA Triad and which SY0-701 objective does it fall under?" | Correct answer from Section 2 notes with objective citation | Correctly read from uploaded notes, cited Objective 1.2, provided a formatted table with all three pillars, real-world examples, and bonus context on Non-Repudiation and AAA | Pass |
| 2 | Known bad | "What are the requirements to pass the CISSP exam?" | Out-of-scope refusal — not in uploaded notes | Refused correctly: "That's outside my scope. I'm SecPlus-Bot... specifically designed to help students prepare for the CompTIA Security+ SY0-701 exam only." Did not fabricate CISSP information. | Pass |
| 3 | Edge case | "What is the difference between a threat vector and an attack vector when it comes to ransomware?" | Uncertain — topic spans Section 3 and Section 6 | Agent began reading files, appeared to pause mid-response, then automatically retried the tool call and completed the response on its own without any intervention | n/a |

---

## My one change

- **Which test prompted the change:** Test 3 — agent appeared to stall mid-response after saying "Let me pull up the relevant study notes for you"
- **The one thing I changed in my instructions:** Nothing — no manual change was required
- **What happened when I re-ran that test:** The agent retried automatically and completed the response, correctly distinguishing threat vector (method of infiltration, e.g. phishing campaign delivering ransomware) from attack vector (means of gaining access and executing the infection), citing Section 6 Objective 2.4

---

## Observations

Test 1 confirmed file mounting and agent_toolset_20260401 are working correctly — the agent read from /uploads/Section 2_ Fundamentals of Security.md and returned accurate scoped content. Test 2 confirmed the scope constraint in the system prompt is enforced. Test 3 showed the agent self-recovering — it retried the file read automatically and completed the response without any prompt changes or manual intervention.

---

## Setup Notes

The file upload and attachment process was the most challenging part of this lab. The platform.claude.com console UI does not support direct file uploads — files must be uploaded programmatically via the Files API using the `files-api-2025-04-14` beta header. Getting the upload script working required multiple attempts: the initial script failed because the filenames on disk used spaces and single underscores rather than the double-underscore format assumed in the script. Running `dir` to get the exact filenames and correcting the script resolved it.

Once files were uploaded and File IDs were obtained, attaching them to the session under Resources required entering each File ID and mount path manually one at a time. An additional error — "Missing required tool: file resources require the read tool to be usable" — appeared because the `agent_toolset_20260401` block had been removed from the YAML during cleanup. Re-adding it fixed the session creation.

The process required critical thinking to diagnose each failure point and systematic testing to resolve them. Once the setup was correct the agent itself performed as expected on all three tests.

---

## Session Verification

The raw session event log (`session-events-sesn_01CGfjAYcjAR1hwyZZoN8fEn.json`) is included in this project folder as verification of the live agent session. It contains the full event stream including user messages, agent responses, tool calls, and file read results confirming the agent read from the uploaded knowledge files during testing.
