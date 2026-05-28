# CSC-114-InfoBot — Session 1 Testing Log

**Platform:** platform.claude.com (Managed Agents)
**Model:** claude-sonnet-4-6
**Agent name:** web-csc114-infobot
**Date tested:** 2026-05-27

## Test Results

| Test # | Input | Expected | Actual | Pass / Fail | Notes |
|---|---|---|---|---|---|
| 1 | "What are the office hours for the CSC-114 instructors?" | "Upon request (Summer)" per Instructors section | "Per the Instructors section of the syllabus, both instructors hold office hours upon request during the Summer term: Mallory Milstead — ATC-113H, 910-678-8572, milsteam@faytechcc.edu; Andrew Norris — ATC-113C, 910-486-3967, norrisa@faytechcc.edu. Since hours are by appointment, just reach out via email or phone to set something up!" | Pass | Correctly cited the Instructors section and gave accurate contact info |
| 2 | "What's the GPA requirement to enroll in CSC-114?" | Fallback message — not in syllabus | "That isn't in the CSC-114 syllabus. Please contact your instructor — Mallory Milstead (milsteam@faytechcc.edu) or Andrew Norris (norrisa@faytechcc.edu) — for clarification. What I can tell you from the syllabus is that the listed prerequisite for CSC-114 is CSC-113." | Pass | Correctly refused to guess; noted the actual prerequisite from the syllabus |
| 3 | "Is the final exam on July 15th?" | Should reject — final project is 7/20 per Class Schedule | "Per the Class Schedule section of the syllabus, there is no final exam listed. Instead, there is a Final Project due on July 20, 2026 (worth 100 points). There is no assignment or exam listed for July 15th." | Pass | Correctly rejected the false date and pointed to the actual 7/20 final project |

## Token Counts (Test 1)

| Metric | Value |
|---|---|
| Input tokens | 3 |
| Output tokens | 165 |
| Cache creation input tokens | 7,233 |
| Cache read input tokens | 0 |
| Total tokens (input + output + cache) | 7,401 |

## Debug Observations

The Debug tab gave more granular data. On the transcript I could see the
tokens in the form of cache read, cache creation, uncached, and output.
On the Debug tab I could break it down further and see when the message
was sent to the model, the model's response, the token data, and then the
session end — or at least that part of the conversation ending.
