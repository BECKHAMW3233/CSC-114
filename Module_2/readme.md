# Module 2 — Deep Learning Basics

**Course:** CSC-114 Artificial Intelligence I · Summer 2026
**Due:** June 14, 2026
**Status:** Complete

---

## Overview

This module covers Chapter 2 of *Deep Learning with Python* (3rd ed.) — the mathematical building blocks of neural networks. Topics include tensors, tensor operations, gradient descent, and backpropagation. Practical work includes building a course-specific study agent, working through vocabulary and reflection questions, and an in-class Teachable Machines exercise.

---

## Files

| File | Description |
|------|-------------|
| `m2-practice-chat.md` | Chat transcript from CSC114Bot session covering vocabulary definitions and three reflection questions |
| `system-prompt-v1.md` | CSC114Bot system prompt documentation — design decisions, file upload process, and comparison to Module 1 SecPlus-Bot |
| `web-csc114-bot.yaml` | Agent YAML configuration file |
| `session-events-sesn_01RPSkh33rgk4LvfTMYNXjzZ.json` | Raw session event log from platform.claude.com verifying the live agent session |

**notes/**

| File | Description |
|------|-------------|
| `chapter2_neural_network_math.md` | Full Chapter 2 text from *Deep Learning with Python* (3rd ed.) — uploaded to the Files API as the agent knowledge base |

**teachable-machine/**

| File | Description |
|------|-------------|
| `metadata.json` | Model metadata — labels, version, timestamp |
| `model.json` | TensorFlow.js model architecture |
| `weights.bin` | Trained model weights |

---

## Agent Built This Module

**CSC114Bot** (`web-csc114-bot`) — deployed on platform.claude.com (Managed Agents)

- **Model:** claude-sonnet-4-6
- **Knowledge base:** `chapter2_neural_network_math.md` uploaded via Files API
- **File ID:** `file_011Cbqtt5oGEU3GMRJpzdarg`
- **Session ID:** `sesn_01RPSkh33rgk4LvfTMYNXjzZ`

The agent reads the full Chapter 2 text to answer questions, explain vocabulary, and guide students through reflection questions using the Socratic method rather than giving direct answers.

---

## Reflection Questions Covered

**Q1 — Hot/Cold and Gradient Descent**
How is the Hot/Cold game related to gradient descent?

**Q2 — Teachable Machines Three States**
When teaching Teachable Machines to recognize Image A or Image B, why do we need three states instead of two?

**Q3 — Python Notebook Workflows**
What skills and workflows do I need to develop to work with Python notebooks, assuming my instructor provides the .ipynb file?

---

## Teachable Machines

- **Model type:** Image recognition
- **Labels:** `neutral`, `TWO`, `FIVE`
- **Task:** Classify handwritten digits — distinguishing a written 2 from a written 5, with a neutral/background class
- **Framework:** Google Teachable Machines (TensorFlow.js)
- **Exported:** `teachable-machine/` — metadata, model architecture, and weights

The `neutral` class directly demonstrates the answer to Q2: a two-class model forced to choose between TWO and FIVE would misfire on any input that is neither. The background class gives the model a valid output for unrecognized input.

---

## Key Vocabulary (Chapter 2)

| Term | Definition |
|------|-----------|
| Scalar | Single number — rank-0 tensor |
| Vector | 1D array of numbers — rank-1 tensor |
| Matrix | 2D grid of numbers — rank-2 tensor |
| Tensor | N-dimensional container for numerical data |
| Rank | Number of axes a tensor has (`ndim`) |
| Slope | Rate of change of a function at a point (derivative) |
| Gradient | Multidimensional generalization of slope; used to update model weights during training |
