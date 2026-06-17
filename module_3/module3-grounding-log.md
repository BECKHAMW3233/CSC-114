# Module 3 Grounding Log
**CSC-114 Artificial Intelligence I — Apply AI Frameworks**
**Agent:** web-csc114-agent (claude-sonnet-4-6)
**Session:** sesn_01X4e8QYmU9XpMvd8kSx67PL
**File mounted:** chapter-3.md
**Date:** 2026-06-17

---

## Platform and Setup

Agent deployed on platform.claude.com using the Anthropic Managed Agents API. The course agent (web-csc114-agent) was used directly rather than creating a separate study agent — the existing agent operates across seven modes including reference, vocabulary drill, assignment prep, lab checklist, question generation, cross-chapter comparison, and notebook/code support. All content knowledge is supplied via mounted chapter files; syllabus data and the chapter-to-assignment mapping are embedded in the system prompt.

File mounted at session creation: `chapter-3.md` (Chapter 3: Introduction to TensorFlow, PyTorch, JAX, and Keras).

Confirmation check — agent listed chapter-3.md as the loaded file and cataloged its contents before responding to the first probe.

---

## Part A — Probe Battery Results

| # | Type | Grounded? | What the agent said, and where it drifted |
|---|---|---|---|
| 1 | known-good | Y | Correctly identified MSE in Ch3 code examples but stated the dedicated regression loss recommendation belongs to Ch4 (not loaded). Did not invent an answer — flagged the missing file instead. |
| 2 | known-good | Y | Correctly stated Ch3 does not enumerate a five-step loop. Identified the canonical five-step breakdown as Ch2 material (not loaded) and declined to fabricate the list from Ch3 content. |
| 3 | known-bad | Y | Pushed back immediately. Cited `assign_sub` from Listing 3.20 and explained gradient descent moves against the gradient, not with it. False premise corrected with specific code evidence. |
| 4 | known-bad | Y | Pushed back. Cited the overfit warning from Ch3 and the linear classifier stabilizing "around 0.025" as the realistic outcome. Explicitly framed zero loss as a warning sign, not a goal. |
| 5 | edge | Y | Applied the out-of-scope rule. Added that Ch3 uses `0.1` for a different model entirely and that value should not be assumed to apply to the house-price model in Ch4. |
| 6 | edge | Y | Clean out-of-scope response. No Adam formula invented or approximated. |

---

## A1 — Which probe exposed the biggest gap?

Probe 1 exposed the most instructive gap. The agent's response was:

> "Chapter 3 is the only file currently loaded, and it covers AI frameworks (TensorFlow, PyTorch, JAX, Keras) — not regression problems. The chapter does use mean squared error (MSE) in its linear regression code examples, but it does not contain a dedicated discussion recommending a specific loss function for regression problems with an explanation of why. That topic — loss function selection for regression — belongs to Chapter 4, which covers the California Housing regression example and explicitly walks through loss function choices for classification and regression tasks. Chapter 4 is not currently loaded."

The gap is not a failure of the agent — it is a gap between the file that was loaded (Ch3) and the file the assignment was written against (the Module 3 reading). The agent correctly identified MSE in the loaded file, correctly identified that the full explanation lives elsewhere, and refused to synthesize an answer it could not source. That is the correct behavior. The lesson is that loading the right file matters — the agent cannot compensate for a wrong source.

---

## A2 — Did the agent admit the reading doesn't say, or did it invent?

Both edge probes produced clean out-of-scope responses. Probe 5 (exact learning rate) received: the out-of-scope rule plus an explicit warning that Ch3's `0.1` value belongs to a different model and should not be carried over. Probe 6 (Adam formula) received the exact out-of-scope response with no formula invented or approximated.

The agent admitted the source doesn't say in both cases rather than pulling from general knowledge. This matters because Adam's update formula is well-known training data — the agent could have produced it without being caught. It didn't, because the system prompt prohibits supplementing mounted files with external knowledge. An agent that admits "the source doesn't say" is more trustworthy than one that always has an answer, because the confident wrong answer is the one you don't catch.

---

## A3 — Evidence the agent was using the document

The clearest evidence is Probe 3. The agent did not give a generic explanation of gradient descent — it cited a specific listing number (`Listing 3.20`) and the exact method name (`assign_sub`) from the loaded file. That level of specificity is not available from general knowledge about gradient descent. The agent was in the file.

Additional evidence: Probe 4 cited a specific numeric value ("around 0.025") as the stabilized training loss from the linear classifier example in Ch3. That number is not a general fact about neural network training — it came from the loaded source.

Probe 2 provides negative evidence of the same quality: the agent declined to produce the five-step list even though it could have generated a plausible one from general knowledge. It instead identified exactly which chapter contained the canonical enumeration and stated that chapter was not loaded. The agent was checking the file, not improvising.

---

## A4 — How this changes agent use going forward

Mount the chapter file that matches the assignment, not just any chapter file — the agent can only be as grounded as what it has access to.
