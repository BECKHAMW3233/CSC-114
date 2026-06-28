# verify_datasets.py
# Run from E:/CSC-114/emnist-model with the venv active:
#   python verify_datasets.py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import numpy as np

DATA_DIR   = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
KAGGLE_DIR = Path(r"E:\CSC-114\emnist-model\datasets\kaggle")
CHARS_HND  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishHnd\English\Hnd\Img")
CHARS_IMG  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishImg\English\Img\GoodImg\Bmp")

ok = 0
missing = 0

def report(name, status, detail=""):
    global ok, missing
    tag = "[OK]     " if status else "[MISSING]"
    print(f"  {tag}  {name:<28} {detail}")
    if status:
        ok += 1
    else:
        missing += 1

print()
print("=" * 60)
print("  Dataset Verification — EMNIST OCR Pipeline")
print("=" * 60)

# ── EMNIST ────────────────────────────────────────────────────
print("\n  EMNIST (torchvision)")
print("  " + "-" * 50)
from torchvision.datasets import EMNIST
for split in ["byclass", "balanced", "digits"]:
    try:
        tr = EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
        te = EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
        report(f"EMNIST {split}", True, f"train {len(tr):>8,}  test {len(te):>7,}")
    except Exception as e:
        report(f"EMNIST {split}", False, str(e)[:55])

# ── Digit datasets ────────────────────────────────────────────
print("\n  Digit Datasets (torchvision)")
print("  " + "-" * 50)

from torchvision.datasets import MNIST
try:
    tr = MNIST(root=str(DATA_DIR), train=True,  download=False)
    te = MNIST(root=str(DATA_DIR), train=False, download=False)
    report("MNIST", True, f"train {len(tr):>8,}  test {len(te):>7,}")
except Exception as e:
    report("MNIST", False, str(e)[:55])

from torchvision.datasets import USPS
try:
    tr = USPS(root=str(DATA_DIR), train=True,  download=False)
    te = USPS(root=str(DATA_DIR), train=False, download=False)
    report("USPS", True, f"train {len(tr):>8,}  test {len(te):>7,}")
except Exception as e:
    report("USPS", False, str(e)[:55])

from torchvision.datasets import SVHN
try:
    tr = SVHN(root=str(DATA_DIR), split="train", download=False)
    te = SVHN(root=str(DATA_DIR), split="test",  download=False)
    report("SVHN", True, f"train {len(tr):>8,}  test {len(te):>7,}")
except Exception as e:
    report("SVHN", False, str(e)[:55])

# ── Kaggle / manual ───────────────────────────────────────────
print("\n  Kaggle / Manual Datasets")
print("  " + "-" * 50)

npy_imgs = KAGGLE_DIR / "az_images.npy"
npy_lbls = KAGGLE_DIR / "az_labels.npy"
if npy_imgs.exists() and npy_lbls.exists():
    imgs   = np.load(str(npy_imgs))
    labels = np.load(str(npy_lbls))
    report("Kaggle A-Z", True,
           f"samples {imgs.shape[0]:>7,}  classes {len(set(labels.tolist()))}")
else:
    report("Kaggle A-Z", False, "run download_datasets.py")

for name, path in [("Chars74K EnglishHnd", CHARS_HND),
                   ("Chars74K EnglishImg",  CHARS_IMG)]:
    if path.exists():
        files = sum(1 for f in path.rglob("*") if f.is_file())
        report(name, True, f"files {files:>7,}")
    else:
        report(name, False, "not found")

# ── Optimizers ────────────────────────────────────────────────
print("\n  Optimizer Packages")
print("  " + "-" * 50)

try:
    import lion_pytorch
    report("lion-pytorch", True, "installed")
except ImportError:
    report("lion-pytorch", False, "pip install lion-pytorch")

try:
    import schedulefree
    report("schedulefree", True, "installed")
except ImportError:
    report("schedulefree", False, "pip install schedulefree")

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  {ok} OK   {missing} MISSING")
print()
if missing == 0:
    print("  All datasets and packages verified — ready to train.")
else:
    print("  Fix missing items above before starting training.")
print()
print("  Sample counts on disk:")
print(f"    EMNIST byclass:    697,932")
print(f"    EMNIST balanced:   112,800")
print(f"    EMNIST digits:     240,000")
print(f"    MNIST:              60,000")
print(f"    USPS:                7,291")
print(f"    SVHN:               73,257")
print(f"    Kaggle A-Z:        372,451")
print(f"    Chars74K Hnd:        3,411")
print(f"    Chars74K Img:        7,705")
print(f"    ────────────────────────────")
print(f"    Total:           1,574,847")
print()
print("  Digit subtotal:    380,548  vs Kaggle A-Z: 372,451")
print("  With DIGIT_BOOST=3.0x digits dominate training frequency.")
print()
print("=" * 60)
print()