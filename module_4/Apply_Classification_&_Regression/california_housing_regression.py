"""
CSC-114 Module 4 — California Housing Price Regression
Follows Chollet & Watson, Deep Learning with Python 3rd Ed., Chapter 4.3

Pipeline:
  1. Load data (small version: 480 train / 120 test, 8 features)
  2. Normalize features (train stats only)
  3. Scale targets (÷ 100,000)
  4. K-fold cross-validation (K=4) to find best epoch count
  5. Train final model on all training data
  6. Evaluate on sealed test set
  7. Run a sample prediction

Run from E:\\test projects\\csc-114_tasks after running setup_environment.py first.
"""

# ── Load .env so all cache/dataset paths stay off C: ─────────────────────────
import os
from pathlib import Path

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
        print(f"[env]  Loaded {_env_file}")
    except ImportError:
        # dotenv not installed: parse manually
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[env]  Loaded {_env_file} (manual parse)")
else:
    print("[env]  No .env found — run setup_environment.py first.")

# ── Imports (after env vars are set so Keras sees KERAS_HOME etc.) ────────────
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import california_housing


# ── 1. Load data ──────────────────────────────────────────────────────────────

(train_data, train_targets), (test_data, test_targets) = (
    california_housing.load_data(version="small")
)

print("Train shape:", train_data.shape)   # (480, 8)
print("Test shape: ", test_data.shape)    # (120, 8)
print("Sample targets:", train_targets[:5])


# ── 2. Normalize features (mean & std from TRAIN only) ───────────────────────

mean = train_data.mean(axis=0)
std  = train_data.std(axis=0)

x_train = (train_data - mean) / std
x_test  = (test_data  - mean) / std   # reuse train stats — never peek at test


# ── 3. Scale targets (÷ 100,000 so they land in ~0.6–5.0) ───────────────────

y_train = train_targets / 100_000
y_test  = test_targets  / 100_000


# ── 4. Model factory ─────────────────────────────────────────────────────────
#    - Two hidden layers, 64 units, ReLU
#    - Final layer: Dense(1) with NO activation → linear output → any value

def get_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),                    # ← no activation: regression output
    ])
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",          # MSE: penalises big misses hard
        metrics=["mean_absolute_error"],    # MAE: human-readable (× $100k = dollars)
    )
    return model


# ── 5. K-fold cross-validation (K=4, 200 epochs) ────────────────────────────
#    Goal: find the epoch where validation MAE stops improving

K              = 4
num_val        = len(x_train) // K   # 120 districts per fold
NUM_EPOCHS     = 200
all_mae_hist   = []

print("\n── K-fold validation ──")
for i in range(K):
    print(f"  Fold {i + 1}/{K} …", end=" ", flush=True)

    # Carve out the validation slice for this fold
    fold_x_val   = x_train[i * num_val : (i + 1) * num_val]
    fold_y_val   = y_train[i * num_val : (i + 1) * num_val]

    # Everything else is training data
    fold_x_train = np.concatenate(
        [x_train[: i * num_val], x_train[(i + 1) * num_val :]], axis=0
    )
    fold_y_train = np.concatenate(
        [y_train[: i * num_val], y_train[(i + 1) * num_val :]], axis=0
    )

    model = get_model()
    history = model.fit(
        fold_x_train, fold_y_train,
        validation_data=(fold_x_val, fold_y_val),
        epochs=NUM_EPOCHS,
        batch_size=16,
        verbose=0,
    )

    mae_per_epoch = history.history["val_mean_absolute_error"]
    all_mae_hist.append(mae_per_epoch)

    best = min(mae_per_epoch)
    print(f"best val MAE = {best:.3f}  (~${best * 100_000:,.0f} off)")

# Average MAE across all folds for each epoch
average_mae_history = [
    np.mean([fold[i] for fold in all_mae_hist]) for i in range(NUM_EPOCHS)
]

# Quick 50-epoch check (matches book Listing 4.28 output)
print("\n── 50-epoch fold scores ──")
quick_scores = []
for i in range(K):
    val_mae_at_50 = all_mae_hist[i][49]
    quick_scores.append(val_mae_at_50)
    print(f"  Fold {i + 1}: {val_mae_at_50:.3f}")
print(f"  Average: {np.mean(quick_scores):.3f}  (~${np.mean(quick_scores)*100_000:,.0f} off)")


# ── 6. Plot the averaged validation MAE curve ─────────────────────────────────

CROP = 10   # first 10 epochs are on a different scale — drop them
truncated = average_mae_history[CROP:]
epochs    = range(CROP + 1, NUM_EPOCHS + 1)

plt.figure(figsize=(9, 4))
plt.plot(epochs, truncated, color="#34B3A0", linewidth=1.8)
plt.xlabel("Epoch")
plt.ylabel("Avg. validation MAE (×$100k)")
plt.title("K-fold averaged validation MAE (first 10 epochs cropped)")
plt.axvline(x=130, color="#C9603F", linestyle="--", linewidth=1, label="~130 epochs")
plt.legend()
plt.tight_layout()
plt.savefig("validation_mae_curve.png", dpi=120)
plt.show()
print("\nPlot saved → validation_mae_curve.png")


# ── 7. Final model: train on all 480 training districts, 130 epochs ───────────

FINAL_EPOCHS = 130
print(f"\n── Training final model ({FINAL_EPOCHS} epochs, all training data) ──")

final_model = get_model()
final_model.fit(
    x_train, y_train,
    epochs=FINAL_EPOCHS,
    batch_size=16,
    verbose=1,
)


# ── 8. Evaluate on the sealed test set (open ONCE) ───────────────────────────

test_mse, test_mae = final_model.evaluate(x_test, y_test, verbose=0)
print(f"\n── Test results ──")
print(f"  MSE : {test_mse:.4f}")
print(f"  MAE : {test_mae:.3f}  (~${test_mae * 100_000:,.0f} off on average)")


# ── 9. Sample prediction ──────────────────────────────────────────────────────

predictions = final_model.predict(x_test, verbose=0)

print("\n── First 5 predictions vs. actual ──")
for i in range(5):
    pred_dollars   = predictions[i][0] * 100_000
    actual_dollars = test_targets[i]
    print(f"  District {i+1}: predicted ${pred_dollars:>10,.0f}   actual ${actual_dollars:>10,.0f}")
