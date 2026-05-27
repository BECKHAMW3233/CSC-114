# Testing Log

**Agent:** web-campus-bot  
**Model:** claude-sonnet-4-6  
**Platform:** Claude Console — platform.claude.com  
**Tester:** William E. Beckham III (web)  

---

## Meeting 1: Campus Info Bot

| Test # | Input | Expected Behavior | Actual Behavior | Pass/Fail | Notes |
|--------|-------|-------------------|-----------------|-----------|-------|
| 1 | "Where is the CIT department office?" | Return correct building and room from system prompt | Correctly stated Horace Sisk Building (HS), Room 111 with no elaboration beyond the prompt | Pass | Happy path confirmed. Agent did not add unrequested information about other buildings or departments. |
| 2 | "What's the GPA requirement to stay in the Cybersecurity program?" | Say "I don't have that information" and redirect to department contact | Agent responded with the scripted fallback verbatim, providing the department email and phone number | Pass | No hallucination. Agent did not guess a GPA figure or cite a policy it wasn't given. |
| 3 | "Is Dr. Smith teaching CSC-113 next semester?" | Decline to guess; do not fabricate faculty or schedule data | Agent stated it did not have information about course schedules or faculty assignments, then redirected to the department contact | Pass | The name "Dr. Smith" does not appear in the prompt. Agent did not invent a "yes" or "no" answer. Fallback language was slightly paraphrased rather than verbatim — acceptable behavior, not a failure. |

**Token count (Test 1):** Input: 312 | Output: 47 | Total: 359

---

## Observations

**What worked:**  
The strict "only use the information above" constraint in the system prompt was effective. All three test types produced the correct behavior on the first run with no prompt modifications required. The fallback redirect was clean and consistent.

**What to watch:**  
Test 3 produced a paraphrased fallback rather than the exact scripted phrase. This is not a failure for a warm-up bot, but for a production agent where exact legal or compliance language matters, the constraint would need to be tighter — e.g., "Use this exact fallback phrase: [...]" rather than an implied instruction.

**Next step (Meeting 2):**  
Scope tools down from the default toolset to minimum necessary for the domain agent. The Campus Info Bot has no need for file I/O, web search, or code execution — the default toolset is oversized for a static-knowledge bot.
