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
├── module_1/                        ← Module 1: Claude Projects as a Knowledge Platform
│   ├── README.md
│   ├── projects-vs-platform.md
│   ├── system-prompt-v1.md
│   ├── testing-log.md
│   └── Project/
│       ├── web-secplus-bot.yaml
│       ├── custom-instructions.md
│       ├── system-prompt-v1.md
│       ├── testing-log.md
│       ├── session-events-sesn_01CGfjAYcjAR1hwyZZoN8fEn.json
│       └── notes/
│           └── (8 Security+ study note files)
└── Module_2/                        ← Module 2: Deep Learning Basics
    ├── readme.md
    ├── m2-practice-chat.md
    ├── system-prompt-v1.md
    ├── web-csc114-bot.yaml
    ├── session-events-sesn_01RPSkh33rgk4LvfTMYNXjzZ.json
    ├── (Teachable Machines model file — pending 6/14 class)
    └── notes/
        └── chapter2_neural_network_math.md
```

*Additional module folders will be added as the course progresses.*

---

## Module Progress

### Module 1 — Claude Projects as a Knowledge Platform ✅
Built and deployed **SecPlus-Bot** (web-secplus-bot), a CompTIA Security+ SY0-701 exam prep agent on platform.claude.com using the Managed Agents API. Uploaded 8 study note files via the Files API, mounted them as a knowledge base, and validated the agent across three test cases covering known-good recall, out-of-scope refusal, and edge-case self-recovery. Documented in `module_1/testing-log.md` with raw session event log as verification.

### Module 2 — Deep Learning Basics 🔄
Built and deployed **CSC114Bot** (web-csc114-bot), a Chapter 2 deep learning study assistant on platform.claude.com. Uploaded the full Chapter 2 text (`chapter2_neural_network_math.md`) via the Files API and mounted it as the agent's knowledge base. Used the agent to work through course vocabulary (Scalar, Tensor, Vector, Matrix, Rank, Slope, Gradient) and three reflection questions on gradient descent, Teachable Machines classification states, and Python notebook workflows. Teachable Machines in-class component pending 6/14.

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
| Summer 2026 | CSC-114 (this repo) | in progress |
| Fall 2026 | CSC-151, CSC-221, CTS-285, DBA-110 | registered |
| Spring 2027 | CSC-251, CSC-289, CSC-134, DBA-120 | planned |

President's List every semester since Fall 2023. Program GPA: 4.000.

---

All work in this repository is original and completed in accordance with FTCC Academic Integrity policy.
CSC-114-1001 · Summer 2026 · Milstead / Norris · Fayetteville Technical Community College
