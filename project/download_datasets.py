"""
download_datasets.py
====================
Downloads and prepares supplementary datasets alongside EMNIST byclass.

Datasets:
  1. EMNIST Balanced  — 47 balanced classes, equal samples per class
                        Already on disk if EMNIST byclass was downloaded.
                        Zero extra download needed.

  2. Kaggle A-Z Handwritten Characters
                        28x28 grayscale, 372,450 samples across 26 uppercase
                        letter classes. Different writer pool than EMNIST —
                        genuinely different stroke variation for the same chars.
                        Directly addresses b/t, B/P, 2/6 confusion pairs.

Prerequisites:
  - pip install kaggle torchvision pillow numpy
  - Place kaggle.json at C:\\Users\\beckhamw3233\\.kaggle\\kaggle.json
    Get it from: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
    cd E:\\CSC-114\\emnist-model
    python download_datasets.py
"""

import os
import sys
import zipfile
import struct
import gzip
from pathlib import Path
import numpy as np

BASE_DIR     = Path(r"E:\CSC-114\emnist-model")
DATA_DIR     = BASE_DIR / "datasets"
PYTORCH_DIR  = DATA_DIR / "pytorch"
KAGGLE_DIR   = DATA_DIR / "kaggle"
KAGGLE_JSON = Path(r"C:\Users\Will\Downloads\kaggle.json")

print("=" * 60)
print("  Dataset Downloader — EMNIST OCR Supplementary Data")
print("=" * 60)

# =============================================================================
# 1. EMNIST Balanced — verify already downloaded
# =============================================================================
print("\n[1] EMNIST Balanced")
print("    Checking if already downloaded by torchvision...")

try:
    from torchvision.datasets import EMNIST
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    balanced = EMNIST(
        root=str(PYTORCH_DIR),
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )
    print(f"    [OK] EMNIST Balanced: {len(balanced):,} training samples")
    print(f"         Classes: {balanced.classes[:5]}... ({len(balanced.classes)} total)")

    balanced_test = EMNIST(
        root=str(PYTORCH_DIR),
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )
    print(f"    [OK] EMNIST Balanced test: {len(balanced_test):,} samples")

except Exception as e:
    print(f"    [FAIL] {e}")
    print("    Make sure torchvision is installed: pip install torchvision")

# =============================================================================
# 2. Kaggle A-Z Handwritten Characters
# =============================================================================
print("\n[2] Kaggle A-Z Handwritten Characters Dataset")

# Check for kaggle.json
if not KAGGLE_JSON.exists():
    print(f"    [SKIP] kaggle.json not found at {KAGGLE_JSON}")
    print()
    print("    To download this dataset:")
    print("      1. Go to https://www.kaggle.com/settings")
    print("      2. API section -> Create New Token -> downloads kaggle.json")
    print(f"      3. Place it at: {KAGGLE_JSON}")
    print("      4. Re-run this script")
    print()
    print("    Skipping Kaggle download — training will use EMNIST only.")
else:
    print(f"    [OK] kaggle.json found at {KAGGLE_JSON}")

    kaggle_csv = KAGGLE_DIR / "A_Z Handwritten Data.csv"
    kaggle_npy = KAGGLE_DIR / "az_images.npy"
    kaggle_labels_npy = KAGGLE_DIR / "az_labels.npy"

    if kaggle_npy.exists() and kaggle_labels_npy.exists():
        print(f"    [OK] Preprocessed Kaggle data already exists — skipping download")
        imgs   = np.load(str(kaggle_npy))
        labels = np.load(str(kaggle_labels_npy))
        print(f"         Images shape : {imgs.shape}")
        print(f"         Labels shape : {labels.shape}")
        print(f"         Classes      : A-Z (0-25)")
    else:
        print("    Downloading A-Z Handwritten Characters from Kaggle...")
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "sachinpatel21/az-handwritten-alphabets-in-csv-format",
                path=str(KAGGLE_DIR),
                unzip=True,
                quiet=False,
            )
            print("    [OK] Download complete")
        except Exception as e:
            print(f"    [FAIL] Kaggle download failed: {e}")
            print("    Check your kaggle.json is valid and you have internet access.")
            kaggle_csv = None

        # Preprocess CSV to numpy arrays
        if kaggle_csv and kaggle_csv.exists():
            print("    Preprocessing CSV to numpy arrays (this takes ~2 minutes)...")
            try:
                import pandas as pd
                df = pd.read_csv(str(kaggle_csv), header=None)
                labels = df.iloc[:, 0].values.astype(np.int64)
                pixels = df.iloc[:, 1:].values.astype(np.uint8)
                images = pixels.reshape(-1, 28, 28)
                np.save(str(kaggle_npy), images)
                np.save(str(kaggle_labels_npy), labels)
                print(f"    [OK] Saved {len(images):,} images to {kaggle_npy}")
                print(f"         Shape: {images.shape}  Labels: {np.unique(labels)}")
            except ImportError:
                print("    [FAIL] pandas not installed. Run: pip install pandas")
            except Exception as e:
                print(f"    [FAIL] Preprocessing failed: {e}")
        else:
            print("    [SKIP] CSV file not found — download may have failed")

# =============================================================================
# 3. Summary
# =============================================================================
print("\n" + "=" * 60)
print("  Dataset Summary")
print("=" * 60)

datasets = [
    ("EMNIST byclass (train)",   PYTORCH_DIR / "EMNIST" / "raw"),
    ("EMNIST Balanced (train)",  PYTORCH_DIR / "EMNIST" / "raw"),
    ("Kaggle A-Z images",        KAGGLE_DIR / "az_images.npy"),
    ("Kaggle A-Z labels",        KAGGLE_DIR / "az_labels.npy"),
]

for name, path in datasets:
    exists = Path(path).exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status}  {name}")
    if exists and str(path).endswith(".npy"):
        arr = np.load(str(path))
        print(f"           Shape: {arr.shape}")

print()
print("  Training files will automatically use available datasets.")
print("  If Kaggle data is missing, training falls back to EMNIST only.")
print("=" * 60)
