"""
supplementary_data.py
=====================
Shared supplementary dataset loader for all three OCR training models.

Provides digit-only supplementary data sources for MNIST digit recognition:

  DIGIT-SPECIFIC (addresses systematic digit→letter misclassification):
  1. EMNIST Digits     — 280,000 samples, digits 0-9 only
  2. MNIST             — 70,000 samples, digits 0-9, clean centered
  3. USPS              — 9,298 samples, digits 0-9, scanned envelopes
  4. SVHN              — 73,257 samples, digits 0-9, street sign photos
  5. ARDIS IV          — 7,600 samples, digits 0-9, Swedish historical church records
                         Non-NIST writer population, 19th–20th century handwriting.
                         Requires download_datasets.py to have been run first.

  LETTER DATASETS (not used — digits-only project):
  6. EMNIST Balanced, Kaggle A-Z, Chars74K, PG-HWLD — excluded

Class index mapping (10 classes):
    0-9 : digits 0-9

Chars74K Sample folder mapping:
    Sample001-010 → digits 0-9     → byclass 0-9
    Sample011-036 → uppercase A-Z  → byclass 10-35
    Sample037-062 → lowercase a-z  → byclass 36-61

PG-HWLD folder mapping:
    Folder name = single uppercase letter (A, B, C ... Z)
    Maps directly to byclass 10-35 (A=10, B=11, ... Z=35)

Class balance rationale:
    Digits-only project — no letter counterbalancing needed.
    DIGIT_BOOST=1.0 (neutral). Weighting is pure inverse frequency across 10 classes.

    Raw digit samples:  ~440,155  (EMNIST Digits + MNIST + USPS + SVHN + ARDIS IV)
    All sources map to class indices 0-9 directly.
"""

from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import EMNIST, MNIST, USPS, SVHN
from PIL import Image

# Paths
DATA_DIR      = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
KAGGLE_DIR    = Path(r"E:\CSC-114\emnist-model\datasets\kaggle")
KAGGLE_IMGS   = KAGGLE_DIR / "az_images.npy"
KAGGLE_LBLS   = KAGGLE_DIR / "az_labels.npy"
CHARS74K_HND  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishHnd\English\Hnd\Img")
CHARS74K_IMG  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishImg\English\Img\GoodImg\Bmp")
ARDIS_DIR     = Path(r"E:\CSC-114\emnist-model\datasets\ardis")
ARDIS_IMGS    = ARDIS_DIR / "ardis_images.npy"
ARDIS_LBLS    = ARDIS_DIR / "ardis_labels.npy"
PGHWLD_DIR    = Path(r"E:\CSC-114\emnist-model\datasets\pg_hwld")

NUM_CLASSES = 10

# DIGIT_BOOST: no letter classes to counterbalance against in digits-only mode.
# Set to 1.0 — pure inverse frequency weighting across the 10 digit classes.
DIGIT_BOOST = 1.0

# Index mappings
BALANCED_TO_BYCLASS = {**{i: i for i in range(10)}, **{10+i: 10+i for i in range(26)}}
KAGGLE_TO_BYCLASS   = {i: i + 10 for i in range(26)}
CHARS74K_TO_BYCLASS = {f"Sample{i+1:03d}": i for i in range(62)}
# Digits 0-9 map directly to byclass 0-9 for MNIST, USPS, SVHN, EMNIST Digits
DIGIT_TO_BYCLASS    = {i: i for i in range(10)}


# =============================================================================
# Dataset wrappers
# =============================================================================

class EMNISTDigitsDataset(Dataset):
    """
    EMNIST Digits split — 280,000 training + 40,000 test samples, digits 0-9.
    Counterbalances Kaggle A-Z uppercase flood.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.base = EMNIST(
            root=str(DATA_DIR), split="digits", train=train,
            download=True, transform=None,
        )
        self.transform      = transform
        self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label  = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class MNISTDataset(Dataset):
    """
    MNIST — 70,000 samples, digits 0-9, clean centered 28x28.
    Different writer pool than EMNIST — additional stroke variation.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.base = MNIST(
            root=str(DATA_DIR), train=train,
            download=True, transform=None,
        )
        self.transform       = transform
        self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label  = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class USPSDataset(Dataset):
    """
    USPS — 9,298 training + 2,007 test samples, digits 0-9.
    Scanned from US Postal Service envelopes — real-world handwriting,
    different stroke characteristics than EMNIST/MNIST.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.base = USPS(
            root=str(DATA_DIR), train=train,
            download=True, transform=None,
        )
        self.transform       = transform
        self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label  = self.remapped_labels[idx]
        # USPS returns numpy arrays — convert to PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class SVHNDataset(Dataset):
    """
    SVHN (Street View House Numbers) — 73,257 training samples, digits 0-9.
    Photographed from real street signs — very different domain to EMNIST.
    Adds real-world domain shift robustness to digit recognition.
    Digits 0-9 map directly to byclass indices 0-9.
    Note: SVHN labels are 1-10 (10=0) — remapped to 0-9 byclass indices.
    """
    def __init__(self, train: bool = True, transform=None):
        split = "train" if train else "test"
        self.base = SVHN(
            root=str(DATA_DIR), split=split,
            download=True, transform=None,
        )
        self.transform = transform
        # SVHN uses label 10 for digit 0 — remap to standard 0-9
        self.remapped_labels = [int(label) % 10 for label in self.base.labels]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label  = self.remapped_labels[idx]
        # SVHN returns RGB — convert to grayscale to match other datasets
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class KaggleAZDataset(Dataset):
    """Kaggle A-Z — uppercase only (byclass 10-35)."""
    def __init__(self, transform=None):
        if not KAGGLE_IMGS.exists() or not KAGGLE_LBLS.exists():
            raise FileNotFoundError("Kaggle data not found. Run download_datasets.py first.")
        self.images    = np.load(str(KAGGLE_IMGS))
        raw_labels     = np.load(str(KAGGLE_LBLS))
        self.labels    = np.array([KAGGLE_TO_BYCLASS[l] for l in raw_labels], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = int(self.labels[idx])
        pil   = Image.fromarray(img, mode="L")
        if self.transform:
            img_t = self.transform(pil)
        else:
            img_t = transforms.ToTensor()(pil)
        return img_t, label


class BalancedEMNISTDataset(Dataset):
    """EMNIST Balanced — 47 classes remapped to byclass indices."""
    def __init__(self, train: bool = True, transform=None):
        self.base = EMNIST(
            root=str(DATA_DIR), split="balanced", train=train,
            download=True, transform=None,
        )
        self.transform = transform
        valid_indices   = []
        remapped_labels = []
        for i, (_, label) in enumerate(self.base):
            label = int(label)
            if label in BALANCED_TO_BYCLASS:
                valid_indices.append(i)
                remapped_labels.append(BALANCED_TO_BYCLASS[label])
        self.valid_indices   = valid_indices
        self.remapped_labels = remapped_labels

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        img, _   = self.base[orig_idx]
        label    = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


class Chars74KDataset(Dataset):
    """
    Chars74K dataset loader — works for both EnglishHnd and EnglishImg.
    Maps Sample001-062 directly to EMNIST byclass indices 0-61.
    Covers all 62 classes including both upper AND lowercase.
    """
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples   = []

        if not root.exists():
            raise FileNotFoundError(f"Chars74K not found at {root}")

        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            folder_name = folder.name
            if folder_name not in CHARS74K_TO_BYCLASS:
                continue
            label = CHARS74K_TO_BYCLASS[folder_name]
            for ext in ("*.png", "*.jpg", "*.bmp"):
                for img_path in sorted(folder.glob(ext)):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


class ARDISDataset(Dataset):
    """
    ARDIS IV — 7,600 handwritten digit images from Swedish church records.
    19th–20th century writer population, fully independent of NIST.
    28x28 grayscale MNIST-format. Digits 0-9 → byclass indices 0-9.
    Requires download_datasets.py to have been run first.
    """
    def __init__(self, transform=None):
        if not ARDIS_IMGS.exists() or not ARDIS_LBLS.exists():
            raise FileNotFoundError(
                f"ARDIS IV not found at {ARDIS_DIR}. "
                "Run download_datasets.py first."
            )
        self.images          = np.load(str(ARDIS_IMGS))  # (N, 28, 28) uint8
        raw_labels           = np.load(str(ARDIS_LBLS))  # (N,) int64 0-9
        self.remapped_labels = [int(l) for l in raw_labels]  # digits → byclass 0-9 directly
        self.transform       = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.remapped_labels[idx]
        pil   = Image.fromarray(img, mode="L")
        if self.transform:
            img_t = self.transform(pil)
        else:
            img_t = transforms.ToTensor()(pil)
        return img_t, label

    def get_labels(self):
        return self.remapped_labels


class PGHWLDDataset(Dataset):
    """
    PG-HWLD (Gdansk Tech Handwritten Letters Dataset) — 17,160 samples.
    26 uppercase letter classes (A-Z), 660 samples per class.
    Fully independent of NIST SD-19 — unique writer population and
    collection procedure. Maps to byclass indices 10-35 (uppercase A-Z).

    Expected content: exactly 17,160 total images across 26 single-letter
    (A-Z) folders, found ANYWHERE under PGHWLD_DIR at any nesting depth —
    e.g. both of these layouts are supported:
        PGHWLD_DIR/A/*.png                          (flat)
        PGHWLD_DIR/train-images/A/*.png             (nested under a split folder)
        PGHWLD_DIR/test-images/A/*.png

    This was previously top-level-only (PGHWLD_DIR.iterdir()), which silently
    skipped everything the moment the archive extracted into a nested
    train/test split layout instead of flat letter folders. Recursing with
    rglob() handles either layout without requiring manual reorganization.
    If train-images/A and test-images/A both exist, their contents are
    pooled together as one class (this dataset has no standard train/test
    split — see download_datasets.py).
    """
    # Map uppercase letter folder names to byclass indices 10-35
    PGHWLD_TO_BYCLASS = {chr(ord('A') + i): 10 + i for i in range(26)}

    def __init__(self, transform=None):
        self.transform = transform
        self.samples   = []  # list of (Path, byclass_label)

        if not PGHWLD_DIR.exists():
            raise FileNotFoundError(
                f"PG-HWLD not found at {PGHWLD_DIR}. "
                "See download_datasets.py for manual download instructions."
            )

        # Recurse through every subdirectory at any depth, not just top-level,
        # and collect any folder whose name is exactly one of A-Z (case-insensitive).
        seen_paths = set()  # guard against the same file being matched twice
        for folder in sorted(PGHWLD_DIR.rglob("*")):
            if not folder.is_dir():
                continue
            folder_name = folder.name.upper()
            if folder_name not in self.PGHWLD_TO_BYCLASS:
                continue
            label = self.PGHWLD_TO_BYCLASS[folder_name]
            for ext in ("*.png", "*.jpg", "*.bmp", "*.PNG", "*.JPG"):
                for img_path in sorted(folder.glob(ext)):
                    resolved = img_path.resolve()
                    if resolved in seen_paths:
                        continue  # skip exact duplicate path (defensive, shouldn't normally trigger)
                    seen_paths.add(resolved)
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {PGHWLD_DIR}. "
                "Check folder structure: pg_hwld/A/*.png, pg_hwld/B/*.png ... "
                "(letter folders can be nested under a split folder, e.g. "
                "pg_hwld/train-images/A/*.png)"
            )

        actual_count = len(self.samples)
        if actual_count != 17160:
            print(f"  [PG-HWLD WARNING] Loaded {actual_count:,} samples, "
                  f"expected 17,160. Check for duplicate folders or an "
                  f"incomplete extraction before trusting this run's results.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# =============================================================================
# get_class_weights
# =============================================================================

def get_class_weights(dataset) -> torch.Tensor:
    """
    Compute WeightedRandomSampler weights for any dataset type.
    Applies DIGIT_BOOST multiplier to digit classes after inverse frequency weighting.
    For digits-only training DIGIT_BOOST=1.0 (no boost needed).
    """
    print("[Dataset] Computing class weights for balanced sampling...")
    targets       = _extract_targets(dataset)
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES).float()
    class_counts  = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts

    print(f"[Dataset] Applying DIGIT_BOOST={DIGIT_BOOST}x weighting (1.0 = inverse frequency only)")
    class_weights[:10] *= DIGIT_BOOST

    sample_weights = class_weights[targets]
    print(f"[Dataset] Class weight range:  {class_weights.min():.6f} — {class_weights.max():.6f}")
    print(f"[Dataset] Digit weight range:  {class_weights[:10].min():.6f} — {class_weights[:10].max():.6f}")
    return sample_weights


def _extract_targets(dataset) -> torch.Tensor:
    """Recursively extract all labels from any dataset structure."""
    if isinstance(dataset, ConcatDataset):
        return torch.cat([_extract_targets(ds) for ds in dataset.datasets])

    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent = dataset.dataset
        if hasattr(parent, 'targets'):
            return torch.tensor(
                [int(parent.targets[i]) for i in dataset.indices], dtype=torch.long
            )
        return _extract_targets(parent)[list(dataset.indices)]

    if isinstance(dataset, (Chars74KDataset, PGHWLDDataset)):
        return torch.tensor(dataset.get_labels(), dtype=torch.long)

    if isinstance(dataset, KaggleAZDataset):
        return torch.tensor(dataset.labels.tolist(), dtype=torch.long)

    if isinstance(dataset, (EMNISTDigitsDataset, MNISTDataset, USPSDataset,
                             SVHNDataset, BalancedEMNISTDataset, ARDISDataset)):
        return torch.tensor(dataset.remapped_labels, dtype=torch.long)

    if hasattr(dataset, 'targets'):
        return torch.tensor([int(t) for t in dataset.targets], dtype=torch.long)

    raise ValueError(f"Cannot extract targets from dataset type: {type(dataset)}")


# =============================================================================
# Main loader
# =============================================================================

def load_supplementary(
    transform,
    use_balanced:  bool = True,
    use_digits:    bool = True,
    use_mnist:     bool = True,
    use_usps:      bool = True,
    use_svhn:      bool = True,
    use_kaggle:    bool = True,
    use_chars_hnd: bool = True,
    use_chars_img: bool = True,
    use_ardis:     bool = True,
    use_pghwld:    bool = True,
    train:         bool = True,
) -> Optional[ConcatDataset]:
    """
    Load all available supplementary datasets.
    Gracefully skips any dataset that is missing or fails to load.

    Loading order:
      Digit datasets first (EMNIST Digits, MNIST, USPS, SVHN)
      then mixed/letter datasets (Balanced, Kaggle, Chars74K)
    """
    datasets = []
    total    = 0

    # ── Digit datasets ────────────────────────────────────────────────────────
    if use_ardis:
        try:
            ds = ARDISDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] ARDIS IV:        {len(ds):,} samples (Swedish historical digits)")
        except FileNotFoundError:
            print("  [Supplementary] ARDIS IV skipped — run download_datasets.py first")
        except Exception as e:
            print(f"  [Supplementary] ARDIS IV skipped: {e}")

    if use_digits:
        try:
            ds = EMNISTDigitsDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] EMNIST Digits:   {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] EMNIST Digits skipped: {e}")

    if use_mnist:
        try:
            ds = MNISTDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] MNIST:           {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] MNIST skipped: {e}")

    if use_usps:
        try:
            ds = USPSDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] USPS:            {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] USPS skipped: {e}")

    if use_svhn and train:
        try:
            ds = SVHNDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] SVHN:            {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] SVHN skipped: {e}")

    # ── Mixed/letter datasets ─────────────────────────────────────────────────
    if use_balanced:
        try:
            ds = BalancedEMNISTDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] EMNIST Balanced: {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] EMNIST Balanced skipped: {e}")

    if use_kaggle and train:
        try:
            ds = KaggleAZDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Kaggle A-Z:      {len(ds):,} samples (uppercase only)")
        except FileNotFoundError:
            print("  [Supplementary] Kaggle A-Z skipped — run download_datasets.py first")
        except Exception as e:
            print(f"  [Supplementary] Kaggle A-Z skipped: {e}")

    if use_chars_hnd and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_HND, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K Hnd:    {len(ds):,} samples (all 62 classes)")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K Hnd skipped — not found at {CHARS74K_HND}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K Hnd skipped: {e}")

    if use_chars_img and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_IMG, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K Img:    {len(ds):,} samples (all 62 classes)")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K Img skipped — not found at {CHARS74K_IMG}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K Img skipped: {e}")

    if use_pghwld and train:
        try:
            ds = PGHWLDDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] PG-HWLD:         {len(ds):,} samples (non-NIST uppercase A-Z)")
        except FileNotFoundError:
            print(f"  [Supplementary] PG-HWLD skipped — manual download required (see download_datasets.py)")
        except Exception as e:
            print(f"  [Supplementary] PG-HWLD skipped: {e}")

    if not datasets:
        print("  [Supplementary] No supplementary data available — using EMNIST byclass only")
        return None

    print(f"  [Supplementary] Total supplementary samples: {total:,}")
    return ConcatDataset(datasets)


def get_combined_weights(byclass_dataset, supplementary_dataset) -> torch.Tensor:
    """Compute weights for combined MNIST + supplementary digit dataset."""
    byclass_targets = _extract_targets(byclass_dataset)
    supp_targets    = _extract_targets(supplementary_dataset) if supplementary_dataset \
                      else torch.tensor([], dtype=torch.long)
    all_targets     = torch.cat([byclass_targets, supp_targets])
    class_counts    = torch.bincount(all_targets, minlength=NUM_CLASSES).float()
    class_counts    = torch.clamp(class_counts, min=1)
    class_weights   = 1.0 / class_counts
    class_weights[:10] *= DIGIT_BOOST
    sample_weights  = class_weights[all_targets]
    print(f"  [Weights] Combined: {len(all_targets):,} samples")
    print(f"  [Weights] Class weight range:  {class_weights.min():.6f} — {class_weights.max():.6f}")
    print(f"  [Weights] Digit weight range:  {class_weights[:10].min():.6f} — {class_weights[:10].max():.6f}")
    return sample_weights