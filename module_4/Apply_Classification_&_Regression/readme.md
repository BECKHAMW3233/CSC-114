# CSC-114 Module 4 — California Housing Price Regression

**Course:** CSC-114 Artificial Intelligence I — FTCC Summer 2026  
**Student:** William Edward Beckham III  
**Reference:** Chollet & Watson, *Deep Learning with Python*, 3rd Ed., Chapter 4.3

---

## What This Is

A scalar regression model that predicts the median home price of a California neighborhood district using 8 numeric features drawn from the 1990 census. Built in Keras (PyTorch backend) following the book's Chapter 4.3 pipeline exactly, with a documented setup script for reproducible environment installation.

---

## Results

| Metric | Value |
|--------|-------|
| K-fold avg MAE @ 50 epochs | 0.282 (~$28,187 off) |
| K-fold fold scores | 0.295 / 0.298 / 0.237 / 0.298 |
| Final test MAE | **0.302 (~$30,239 off)** |
| Final test MSE | 0.2546 |
| Training time | < 2 minutes (CPU only) |

Book reference result: ~$31,000 MAE. This run: ~$30,239 — within expected run-to-run variance from random weight initialization.

---

## Dataset

**California Housing** (small version) — 1990 census, sourced via `keras.datasets.california_housing`

| Split | Samples | Shape |
|-------|---------|-------|
| Train | 480 | (480, 8) |
| Test | 120 | (120, 8) |

**8 input features per district:**
- Longitude, Latitude
- Median house age
- Population
- Number of households
- Median income
- Total rooms (across all homes)
- Total bedrooms (across all homes)

**Target:** Median home value in dollars (~$60,000–$500,000, 1990 prices)

The small version (600 districts total) is used intentionally — the book chooses it to demonstrate small-data techniques, specifically K-fold cross-validation.

---

## Pipeline

### 1. Feature Normalization
Each of the 8 input features is z-score normalized: subtract the column mean, divide by the column standard deviation. Mean and standard deviation are computed on the **training set only** and then applied to the test set. Using test statistics for normalization would be data leakage.

### 2. Target Scaling
All target values divided by 100,000 (e.g. $283,000 → 2.83). This keeps targets in the same range as the normalized inputs, allowing the model to learn reasonable weights quickly. Predictions are multiplied back by 100,000 to read in dollars.

### 3. Model Architecture
```
Input (8 features)
  → Dense(64, relu)
  → Dense(64, relu)
  → Dense(1)          ← no activation: linear output, any value
```
- **Loss:** `mean_squared_error` — squares the error, penalizing large misses harder
- **Metric:** `mean_absolute_error` — human-readable average dollar error
- **Optimizer:** `adam`
- Two hidden layers, 64 units each — deliberately small to limit overfitting on 480 samples

### 4. K-Fold Cross-Validation (K=4)
With only 480 training samples, a single train/validation split (~100 districts per fold) produces unreliable scores — which 100 districts you happen to pick swings the result by tens of thousands of dollars. K-fold solves this by training 4 separate models, each validated on a different quarter of the data, and averaging the 4 scores. Every district gets used for validation exactly once.

200 epochs are run per fold with per-epoch MAE logged. The averaged validation curve is plotted to find where improvement stalls.

### 5. Final Model
A fresh model is trained on all 480 training districts for 130 epochs (chosen from the validation curve), then evaluated once on the sealed 120-district test set.

---

## Validation MAE Curve

The curve shows the K-fold averaged validation MAE from epoch 11 onward (first 10 cropped — they're off-scale). The steep drop flattens around epoch 40–50 and remains stable through epoch 200 with no significant overfitting rise — consistent with the book's observation that small models on small datasets plateau rather than overfit sharply. The 130-epoch dashed line lands in the stable zone.

---

## Sample Predictions

```
District 1: predicted $251,270   actual $218,800
District 2: predicted $171,616   actual $218,400
District 3: predicted $120,423   actual $ 93,800
District 4: predicted $246,808   actual $173,400
District 5: predicted $215,243   actual $229,700
```

---

## File Structure

```
CSC-114/
└── module_4/
    └── Apply_Classification_&_Regression/
        ├── setup_environment.py           # Run once — installs all dependencies
        ├── california_housing_regression.py  # Main training script
        ├── validation_mae_curve.png       # Output: K-fold MAE curve
        └── README.md                      # This file
```

---

## Environment Setup

**Python:** 3.14 (system install)  
**Keras backend:** PyTorch CPU (`torch 2.12.1+cpu`)  
**Why PyTorch and not TensorFlow:** TensorFlow has no Python 3.14 wheels on Windows as of June 2026. Keras 3.x is backend-agnostic — swapping to PyTorch requires zero changes to the training code.

All cache and dataset files are directed to `E:\test projects\csc-114_tasks` to keep the C: drive clean.

### Install

```powershell
cd "E:\test projects\csc-114_tasks"
& "C:\Users\Will\AppData\Local\Python\pythoncore-3.14-64\python.exe" setup_environment.py
```

Installs: `keras`, `numpy`, `matplotlib`, `python-dotenv`, `torch` (CPU).  
Downloads and caches the California Housing dataset automatically.

### Run

```powershell
& "C:\Users\Will\AppData\Local\Python\pythoncore-3.14-64\python.exe" california_housing_regression.py
```

Expected runtime: under 2 minutes on modern CPU hardware.

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Scalar regression | Single linear output unit, no activation |
| Feature normalization | Z-score per column, train stats only |
| Target scaling | ÷ 100,000 for training, ×100,000 to read results |
| Small-data validation | K-fold (K=4) instead of single train/val split |
| Overfitting awareness | Validation curve used to select epoch count |
| Test set discipline | Sealed test set evaluated exactly once |
| Regression loss | MSE (training) + MAE (reporting) |

---

## Reference

Chollet, F. & Watson, M. (2026). *Deep Learning with Python*, 3rd Edition. Manning Publications. Chapter 4.3: Predicting house prices — a regression example.
