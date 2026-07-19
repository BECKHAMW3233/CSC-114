# CSC-114 · Artificial Intelligence I

**Fayetteville Technical Community College — Summer 2026**

> AAS · Information Technology / Systems Security & Analysis · Graduated with Highest Honors · 4.000 GPA

---

## About This Repository

Course work, labs, and projects for CSC-114 Artificial Intelligence I (Summer 2026, May 26 – July 20). This course is the direct continuation of CSC-113 (Artificial Intelligence Fundamentals), completed Spring 2026 with an A, and moves from generative AI tooling into the underlying mechanics — deep learning, model training, computer vision, and NLP.

**Instructors:** Mallory Milstead · Andrew Norris
**Meeting:** Monday/Wednesday · ATC 115 · 10:00–11:50

---

## About the Author

William E. Beckham III · BECKHAMW3233

U.S. Army veteran — 13B Field Artillery, 38B Civil Affairs. Two combat deployments to Afghanistan (2001–2009). Eight years of residency in Okinawa, Japan.

Graduated May 2026 with an AAS in Information Technology / Systems Security & Analysis (Highest Honors, 4.000 GPA, President's List every eligible semester since Fall 2023). Currently pursuing a second AAS in Information Technology / Computer Programming & Development. Former AI Data Analyst on the DoD Pathfinder Project (2024–2025).

---

## What This Course Builds On

CSC-113 established the operational foundation — local AI infrastructure for federal cybersecurity workflows, validated model selection, and data sovereignty architecture. The findings that carried forward:

- A purpose-built 4.7GB cybersecurity model outperforms a general 32B model on threat intelligence tasks — model selection sets the quality ceiling before prompt engineering begins
- Native Windows outperforms Docker and WSL for single-user GPU inference; abstraction layer overhead is not theoretical
- System prompts that define role, framework, and output format produce consistently better results than per-query instruction
- Cloud AI is eliminated from federal cybersecurity workflows by data handling requirements regardless of performance — quality was equivalent in head-to-head testing

CSC-114 moves from using these tools to understanding and building the models underneath them.

---

## Course Outcomes

- Design and deploy a custom intelligent agent
- Apply AI frameworks: Keras and PyTorch
- Implement classification and regression algorithms
- Execute the machine learning workflow end-to-end
- Apply computer vision techniques
- Apply Natural Language Processing and LLM optimization

**Textbook:** Deep Learning with Python, 3rd Edition — François Chollet & Matthew Watson (Manning, 2026)

---

## Hardware & Environment

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 7900X @ 5.30GHz — 12C/24T |
| GPU | Zotac RTX 4080 16GB VRAM |
| RAM | 64GB DDR5-5600 |
| OS | Windows 11 Pro (Build 26200.7623) |

**Software:** Python 3.x · PyTorch · Keras · TensorFlow
**Local inference:** Ollama v0.21.2 · AnythingLLM v1.10.0 · 40+ deployed models (~150GB)
**Additional:** Kali Linux · NCL Spring 2026 (team L8_Arrivals, rank 142/3,638)

Predecessor repo: [CSC-113: AI Fundamentals](https://github.com/BECKHAMW3233/CSC-113)

---

## Repository Structure

```
CSC-114/
├── README.md
├── .gitkeep
│
├── module_1/                        ← Module 1: Claude Projects as a Knowledge Platform
│   ├── README.md
│   ├── projects-vs-platform.md
│   ├── system-prompt-v1.md
│   ├── testing-log.md
│   └── Project/
│       ├── web-secplus-bot.yaml
│       ├── custom-instructions.md
│       ├── system-prompt-v1 (1).md
│       ├── testing-log (1).md
│       ├── session-events-sesn_01CGfjAYcjAR1hwyZZoN8fEn.json
│       └── notes/
│           └── (8 Security+ SY0-701 study note files, Sections 1–8)
│
├── Module_2/                        ← Module 2: Deep Learning Basics
│   ├── readme.md
│   ├── chapter02_mathematical_building_blocks.ipynb
│   ├── m2_practice-chat.md
│   ├── system-prompt-v1.md
│   ├── web-csc114-bot.yaml
│   ├── session-events-sesn_01RPSkh33rgk4LvfTMYNXjzZ.json
│   ├── notes/
│   │   └── chapter2_neural_network_math.md
│   └── teachable-machine/
│       ├── metadata.json
│       ├── model.json
│       └── weights.bin
│
├── module_3/                        ← Module 3: AI Frameworks in Python
│   ├── chapter-3.md
│   ├── clasifier.png
│   ├── module3-grounding-log.md
│   ├── session-events-sesn_01X4e8QYmU9XpMvd8kSx67PL.json
│   ├── web-csc114-agent.yaml
│   └── ai_assess/
│       ├── Beckham_AssessAIFrameworks_Response.md
│       ├── mnist_convnet.keras
│       ├── mnist_convnet.py
│       └── Untitled0.ipynb
│
├── module_4/                        ← Module 4: Classification & Regression
│   ├── california_housing_regression.py
│   ├── chapter04_classification-and-regression.ipynb
│   ├── readme.md
│   ├── Apply_Classification_&_Regression/
│   │   ├── california_housing_regression.py
│   │   ├── readme.md
│   │   ├── setup_environment.py
│   │   └── validation_mae_curve.png
│   └── Assess_Classification_&_Regression/
│       ├── module4_assess_answers.md
│       └── validation_mae_curve.png
│
├── module_5/                        ← Module 5: Machine Learning Workflow (Project Charter)
│   ├── agent-guardrails.md
│   ├── charter.md
│   ├── issues.md
│   └── reflection.md
│
├── module_6/                        ← Module 6: Project Sprint 1 — Check-In 1
│   └── Spring 1 Reflection.md
│
├── module_7/                        ← Module 7: Project Sprint 2 — Check-In 2
│   └── Spring 2 Reflection.md
│
├── Project/                          ← Module 8: Final Project — Multi-Optimizer MNIST Ensemble
│   ├── 01_install_cuda.bat
│   ├── 02_install_python_packages.bat
│   ├── 03_verify_gpu.py
│   ├── install_deps.py
│   ├── mnist-benchmark-comparison.md
│   ├── mnist_adamw_128.py / mnist_adamw_64.py
│   ├── mnist_lion_128.py / mnist_lion_64.py
│   ├── mnist_sgd_128.py / mnist_sgd_64.py
│   ├── ocr_soap_128.py / ocr_soap_64.py
│   ├── ocr_pipeline_mnist.py
│   ├── mnist_demo.html / mnist_demo.md
│   ├── supplementary_data.py
│   ├── README.md
│   ├── adamw_128/ · adamw_64/
│   ├── lion_128/ · lion_64/
│   ├── sgd_128/ · sgd_64/
│   ├── pytorch_soap_128/ · pytorch_soap_64/
│   └── (each optimizer folder: best/final .pt checkpoints, .onnx export,
│        training curves, training log .csv, CLI run log)
│
├── emnist-model/                     ← Standalone: EMNIST OCR Ensemble Research
│   ├── 01_install_cuda.bat / 02_install_python_packages.bat / 03_verify_gpu.py
│   ├── download_datasets.py
│   ├── home_test_full.py
│   ├── ocr_distillation.py
│   ├── ocr_pipeline.py
│   ├── ocr_pytorch_model.py / model2.py / model3.py
│   ├── supplementary_data.py · test.py · README.md
│   ├── pytorch/ · pytorch2/ · pytorch3/         (3-model ensemble)
│   └── pytorch_distill1/ · distill2/ · distill3/ (knowledge-distilled variants)
│
└── temp-agent/                       ← Course reference agent (web-csc114-agent)
    ├── session-events-sesn_01QjCfwWMNBpXFpCvR4LHVVX.json
    ├── web-csc114-agent.md
    ├── web-csc114-agent.yaml
    └── chapters/
        └── chapter-1.md … chapter-9.md, temp.md
```

---

## Module Progress

### Module 1 — Claude Projects as a Knowledge Platform ✅
Built and deployed **SecPlus-Bot** (`web-secplus-bot`), a CompTIA Security+ SY0-701 exam prep agent on platform.claude.com using the Managed Agents API. Uploaded 8 study note files via the Files API, mounted them as a knowledge base, and validated the agent across three test cases covering known-good recall, out-of-scope refusal, and edge-case self-recovery. Documented in `module_1/testing-log.md` with raw session event log as verification.

### Module 2 — Deep Learning Basics ✅
Built and deployed **CSC114Bot** (`web-csc114-bot`), a Chapter 2 deep learning study assistant on platform.claude.com. Uploaded the full Chapter 2 text via the Files API as the agent's knowledge base. Used the agent to work through course vocabulary (Scalar, Tensor, Vector, Matrix, Rank, Slope, Gradient) and three reflection questions on gradient descent, Teachable Machines classification states, and Python notebook workflows. Built an image classification model in Google Teachable Machines distinguishing handwritten digits TWO and FIVE with a neutral background class. Exported model files committed to `teachable-machine/`.

### Module 3 — AI Frameworks in Python ✅
**Apply — Trust but Verify:** Loaded the Module 3 "How Machine Learning Works" reading into a Claude Project study agent and confirmed grounding with a "can it see it?" check. Ran a six-probe validation battery (two known-good, two known-bad/trap, two edge/not-in-the-doc) against the agent to test whether its answers were actually sourced from the reading or fabricated. Documented in `module3-grounding-log.md`.

**Assess — Option 2 (Build a Model with Claude Code):** Built and trained a Keras CNN (`mnist_convnet.py` / `.keras`) rather than the cheat-sheet option, using Claude Code with an Anthropic API key against a GPU-backed local/Colab environment. Documented dataset attributes, target, model type, optimizer choice, training epochs, and final accuracy/loss in `Beckham_AssessAIFrameworks_Response.md`.

### Module 4 — Classification & Regression ✅
**Apply:** Selected the Regression track from Chapter 4 (California Housing dataset, per Chollet & Watson's textbook example) over the IMDB classification track. Implemented the model with an AI coding agent in `california_housing_regression.py`, tracking validation MAE across training (`validation_mae_curve.png`).

**Assess:** Worked with the agent to generate three reflection questions on the build process and answered them together, including a required question on what was changed and how results shifted (e.g., k-fold value adjustments) — documented in `module4_assess_answers.md`.

### Module 5 — Machine Learning Workflow (Project Charter) ✅
**Apply:** Used the course agent — fed the Module 5 Inception Outline, Quickstart FAQ, and personal research — to prepare a formal **project charter** for the Module 8 final project (the multi-optimizer MNIST ensemble). Also produced supporting planning docs: `agent-guardrails.md` (AI partnership boundaries later referenced in the Module 8 rubric) and `issues.md`.

**Assess:** Consolidated and committed the charter and planning docs to `module_5/`, submitted as the graded deliverable.

### Module 6 — Project Sprint 1: "Is It Alive?" ✅
**Check-In 1** — proved the final project (MNIST multi-optimizer ensemble) runs start-to-finish, even in a rough state: takes input, produces a prediction. Documented in `Spring 1 Reflection.md`: what runs, what's missing or broken, and whether scope still matches the Module 5 charter.

### Module 7 — Project Sprint 2: "Is It Getting Better?" ✅
**Check-In 2** — documented a measurable before/after improvement over the Check-In 1 version (optimizer/config comparisons across AdamW, Lion, SGD, and SOAP at both 64 and 128 resolutions). Reflection in `Spring 2 Reflection.md` covers what was changed, why it was expected to help, and whether it actually did.

### Module 8 — Final Project: Multi-Optimizer MNIST Ensemble ✅
Committed to the top-level `Project/` folder rather than a `module_8/` folder. Final release-day deliverable: a from-scratch PyTorch training and benchmarking pipeline comparing four optimizers (AdamW, Lion, SGD, SOAP) at two input resolutions (64×64, 128×128) on MNIST, with full ONNX export, training curves, and CSV logs per configuration. Presented live per the Module 8 rubric: goal, data, approach, iterative improvement, and a working demo — traceable back to the Module 5 charter and `agent-guardrails.md`.

---

## Standalone Research

### EMNIST OCR Ensemble (`emnist-model/`)
Originated as course-adjacent exploration and grew into independent research beyond CSC-114 scope. A 6-model PyTorch ensemble (3 architecturally diverse base models — Lion, Schedule-Free AdamW, and SGD-trained — each paired with a knowledge-distilled variant) for handwritten character recognition across all 62 EMNIST classes (digits, uppercase, lowercase). Trained on 9 merged data sources (~1.44M samples), exported to ONNX, and deployed through a custom voting/inference pipeline with post-processing correction for known model bias patterns. Achieved 100% on structured digit-grid and mixed-content benchmark tests; full methodology, per-class accuracy breakdowns, and stress-test findings are documented in `emnist-model/README.md`. A v4 multi-resolution (32/64/128/256) expansion is planned pending cloud GPU compute.

See [`emnist-model/`](https://github.com/BECKHAMW3233/CSC-114/tree/main/emnist-model) for full documentation.

### Course Reference Agent (`temp-agent/`)
`web-csc114-agent` — a purpose-built Claude Sonnet 4.6 agent scoped strictly to CSC-114 course content (Chapters 1–9 of the textbook, syllabus data, and assignment schedule embedded directly in its system prompt). Operates across seven modes: reference Q&A, vocabulary drill, cross-chapter comparison, assignment prep, lab checklist review, reflection question generation, and notebook/code support — all strictly grounded in mounted chapter files, with a hard out-of-scope refusal rule for anything outside them.

---

## Academic Record

| Term | Courses | GPA |
|------|---------|-----|
| Fall 2023 | CTI-120, NET-125, NET-126, NOS-110, SEC-110 | 4.000 |
| Spring 2024 | CTI-110, NOS-120, SEC-150 | 4.000 |
| Fall 2024 | CCT-240, CSC-121, NOS-230, PSY-150 | 4.000 |
| Spring 2025 | COM-120, CTS-115, NOS-220, SEC-160 | 4.000 |
| Fall 2025 | ENG-111, MAT-143, SEC-175, SEC-210 | 4.000 |
| Spring 2026 | CCT-250, CIS-115, CSC-113, SEC-285 | 4.000 |
| Summer 2026 | CSC-114 (this repo) | complete — 4.000 |
| Fall 2026 | CSC-151, CSC-221, CTS-285, DBA-110 | registered |
| Spring 2027 | CSC-251, CSC-289, CSC-134, DBA-120 | planned |

President's List for every semester with 12+ credit hours (the required threshold). Program GPA: 4.000.

---

All work in this repository is original and completed in accordance with FTCC Academic Integrity policy.
CSC-114-1001 · Summer 2026 · Milstead / Norris · Fayetteville Technical Community College
