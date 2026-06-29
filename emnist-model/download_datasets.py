"""
download_datasets.py
====================
Downloads and prepares all supplementary datasets alongside EMNIST byclass.

Datasets (auto-download via torchvision):
  1. EMNIST Balanced   — 112,800 samples, 47 balanced classes
  2. EMNIST Digits     — 280,000 samples, digits 0-9 only
  3. MNIST             — 70,000 samples, digits 0-9, clean centered
  4. USPS              — 9,298 samples, digits 0-9, scanned envelopes
  5. SVHN              — 73,257 samples, digits 0-9, street sign photos
  6. Kaggle A-Z        — 372,450 samples, uppercase only (requires kaggle.json)

Datasets (manual download required):
  7. Chars74K EnglishHnd — ~3,410 handwritten samples, all 62 classes
  8. Chars74K EnglishImg — ~7,705 natural scene samples, all 62 classes
     Download from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

Prerequisites:
  pip install kaggle torchvision pillow numpy pandas
  Place kaggle.json at C:\\Users\\Will\\.kaggle\\kaggle.json

Usage:
    cd E:\\CSC-114\\emnist-model
    python download_datasets.py
"""

import os
from pathlib import Path
import numpy as np

BASE_DIR    = Path(r"E:\CSC-114\emnist-model")
DATA_DIR    = BASE_DIR / "datasets"
PYTORCH_DIR = DATA_DIR / "pytorch"
KAGGLE_DIR  = DATA_DIR / "kaggle"
KAGGLE_JSON = Path(r"C:\Users\Will\.kaggle\kaggle.json")

PYTORCH_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Dataset Downloader — EMNIST OCR Supplementary Data")
print("=" * 60)

# Shared transform for verification
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# =============================================================================
# Helper
# =============================================================================
def check(name, ds_train, ds_test=None):
    print(f"    [OK] {name} train: {len(ds_train):,} samples")
    if ds_test:
        print(f"    [OK] {name} test:  {len(ds_test):,} samples")

# =============================================================================
# 1. EMNIST Balanced
# =============================================================================
print("\n[1] EMNIST Balanced")
try:
    from torchvision.datasets import EMNIST
    tr = EMNIST(root=str(PYTORCH_DIR), split="balanced", train=True,  download=True, transform=transform)
    te = EMNIST(root=str(PYTORCH_DIR), split="balanced", train=False, download=True, transform=transform)
    check("EMNIST Balanced", tr, te)
except Exception as e:
    print(f"    [FAIL] {e}")

# =============================================================================
# 2. EMNIST Digits
# =============================================================================
print("\n[2] EMNIST Digits  ← counterbalances Kaggle A-Z uppercase flood")
try:
    from torchvision.datasets import EMNIST
    tr = EMNIST(root=str(PYTORCH_DIR), split="digits", train=True,  download=True, transform=transform)
    te = EMNIST(root=str(PYTORCH_DIR), split="digits", train=False, download=True, transform=transform)
    check("EMNIST Digits", tr, te)
    print(f"    [OK] Classes: {tr.classes}")
except Exception as e:
    print(f"    [FAIL] {e}")

# =============================================================================
# 3. MNIST
# =============================================================================
print("\n[3] MNIST  ← clean centered digits, different writer pool than EMNIST")
try:
    from torchvision.datasets import MNIST
    tr = MNIST(root=str(PYTORCH_DIR), train=True,  download=True, transform=transform)
    te = MNIST(root=str(PYTORCH_DIR), train=False, download=True, transform=transform)
    check("MNIST", tr, te)
except Exception as e:
    print(f"    [FAIL] {e}")

# =============================================================================
# 4. USPS
# =============================================================================
print("\n[4] USPS  ← scanned postal envelopes, real-world digit variation")
try:
    from torchvision.datasets import USPS
    tr = USPS(root=str(PYTORCH_DIR), train=True,  download=True, transform=transform)
    te = USPS(root=str(PYTORCH_DIR), train=False, download=True, transform=transform)
    check("USPS", tr, te)
except Exception as e:
    print(f"    [FAIL] {e}")

# =============================================================================
# 5. SVHN
# =============================================================================
print("\n[5] SVHN  ← street sign photos, real-world domain shift")
try:
    from torchvision.datasets import SVHN
    tr = SVHN(root=str(PYTORCH_DIR), split="train", download=True, transform=transform)
    te = SVHN(root=str(PYTORCH_DIR), split="test",  download=True, transform=transform)
    check("SVHN", tr, te)
    print(f"    Note: SVHN label 10 = digit 0 — remapped automatically in supplementary_data.py")
except Exception as e:
    print(f"    [FAIL] {e}")

# =============================================================================
# 6. Kaggle A-Z
# =============================================================================
print("\n[6] Kaggle A-Z Handwritten Characters")

if not KAGGLE_JSON.exists():
    print(f"    [SKIP] kaggle.json not found at {KAGGLE_JSON}")
    print("    To download:")
    print("      1. Go to https://www.kaggle.com/settings")
    print("      2. API -> Create New Token -> downloads kaggle.json")
    print(f"      3. Place at: {KAGGLE_JSON}")
    print("      4. Re-run this script")
else:
    os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_JSON.parent)
    print(f"    [OK] kaggle.json found")

    kaggle_npy  = KAGGLE_DIR / "az_images.npy"
    kaggle_lbls = KAGGLE_DIR / "az_labels.npy"
    kaggle_csv  = KAGGLE_DIR / "A_Z Handwritten Data.csv"

    if kaggle_npy.exists() and kaggle_lbls.exists():
        imgs   = np.load(str(kaggle_npy))
        labels = np.load(str(kaggle_lbls))
        print(f"    [OK] Already preprocessed: {imgs.shape}  Labels: {np.unique(labels)}")
    else:
        print("    Downloading from Kaggle...")
        try:
            import kaggle as kg
            kg.api.authenticate()
            kg.api.dataset_download_files(
                "sachinpatel21/az-handwritten-alphabets-in-csv-format",
                path=str(KAGGLE_DIR), unzip=True, quiet=False,
            )
            print("    [OK] Download complete")
        except Exception as e:
            print(f"    [FAIL] {e}")
            kaggle_csv = None

        if kaggle_csv and Path(kaggle_csv).exists():
            print("    Preprocessing CSV (~2 minutes)...")
            try:
                import pandas as pd
                df     = pd.read_csv(str(kaggle_csv), header=None)
                labels = df.iloc[:, 0].values.astype(np.int64)
                pixels = df.iloc[:, 1:].values.astype(np.uint8)
                images = pixels.reshape(-1, 28, 28)
                np.save(str(kaggle_npy),  images)
                np.save(str(kaggle_lbls), labels)
                print(f"    [OK] Saved {len(images):,} images")
            except ImportError:
                print("    [FAIL] pandas not installed: pip install pandas")
            except Exception as e:
                print(f"    [FAIL] {e}")

# =============================================================================
# 7 & 8. Chars74K — manual download
# =============================================================================
print("\n[7] Chars74K EnglishHnd + [8] EnglishImg")
chars_hnd = DATA_DIR / "EnglishHnd" / "English" / "Hnd" / "Img"
chars_img = DATA_DIR / "EnglishImg" / "English" / "Img" / "GoodImg" / "Bmp"

for name, path in [("EnglishHnd", chars_hnd), ("EnglishImg", chars_img)]:
    if path.exists():
        count = sum(1 for _ in path.rglob("*") if _.is_file())
        print(f"    [OK] Chars74K {name}: {count:,} files found")
    else:
        print(f"    [MISSING] Chars74K {name}")
        print(f"              Download from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/")
        print(f"              Extract to:    {path.parent.parent.parent.parent}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("  Dataset Summary")
print("=" * 60)
print()
print("  DIGIT DATASETS (primary fix for digit→letter misclassification):")
print("    EMNIST Digits:     280,000  samples  (digits 0-9)")
print("    MNIST:              70,000  samples  (digits 0-9)")
print("    USPS:                9,298  samples  (digits 0-9)")
print("    SVHN:               73,257  samples  (digits 0-9)")
print("    ─────────────────────────────────────")
print("    Digit subtotal:    432,555  samples")
print()
print("  MIXED/LETTER DATASETS:")
print("    EMNIST byclass:    814,255  samples  (62 classes — primary)")
print("    EMNIST Balanced:   112,800  samples  (47 classes)")
print("    Kaggle A-Z:        372,450  samples  (uppercase only)")
print("    Chars74K Hnd:        3,410  samples  (all 62 classes)")
print("    Chars74K Img:        7,705  samples  (all 62 classes)")
print()
print("  TOTAL:             1,822,475  samples")
print()
print("  With DIGIT_BOOST=3.0x in supplementary_data.py:")
print("    Digits have 3x effective weight over equal sampling.")
print("    Raw digit samples (432,555) now exceed Kaggle A-Z (372,450).")
print("    Combined effect: strongest anti-bias configuration yet.")
print()
print("  Training will skip any missing dataset automatically.")
print("=" * 60)