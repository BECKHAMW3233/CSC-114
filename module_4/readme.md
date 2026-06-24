# Module 4 — Classification and Regression

**Course:** CSC-114 Artificial Intelligence I — FTCC Summer 2026  
**Student:** William Edward Beckham III  
**Reference:** Chollet & Watson, *Deep Learning with Python*, 3rd Ed., Chapter 4

---

## Overview

Module 4 covers the two core supervised learning task types — classification and regression — using three real-world examples from Chapter 4: binary classification (IMDb movie reviews), multiclass classification (Reuters newswires), and scalar regression (California Housing prices). This module's work focuses on Option B: scalar regression.

---

## Assignments

| Assignment | Due | Points | Status |
|---|---|---|---|
| Apply — Classification & Regression | 6/28/26 | 100 | ✅ Complete |
| Assess — Classification & Regression | 6/28/26 | 100 | ✅ Complete |

---

## Contents

```
module_4/
├── README.md                                    # This file
├── Apply_Classification_&_Regression/
│   ├── README.md                                # Project documentation
│   ├── setup_environment.py                     # Install dependencies
│   ├── california_housing_regression.py         # Training script
│   └── validation_mae_curve.png                 # K-fold MAE curve output
└── Assess_Classification_&_Regression/
    ├── module4_assess_answers.md                # Assess submission
    └── validation_mae_curve.png                 # Training curve (attached)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Option | B — House Prices (Regression) |
| K-fold avg MAE @ 50 epochs | 0.282 (~$28,187 off) |
| Final test MAE | **0.302 (~$30,239 off)** |
| Book reference MAE | ~$31,000 |
| Overfitting turnaround | ~epoch 130 |

---

## Environment Note

Trained locally on personal hardware (Ryzen 9 7900X) using Keras with PyTorch CPU backend. TensorFlow has no Python 3.14 wheels on Windows as of June 2026. All dependencies managed via `setup_environment.py`.
