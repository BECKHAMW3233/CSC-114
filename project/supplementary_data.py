"""
supplementary_data.py
=====================
Shared supplementary dataset loader for all three OCR training models.

Provides four additional data sources alongside EMNIST byclass:
  1. EMNIST Balanced        — 47 balanced classes, equal samples per class
  2. Kaggle A-Z             — 372,450 samples, 26 UPPERCASE classes only
  3. Chars74K EnglishHnd    — 3,410 handwritten samples, all 62 classes (both cases)
  4. Chars74K EnglishImg    — natural scene characters, all 62 classes (both cases)

EMNIST byclass class index mapping (62 classes):
    0-9   : digits 0-9
    10-35 : uppercase A-Z
    36-61 : lowercase a-z

Chars74K Sample folder mapping:
    Sample001-010 → digits 0-9     → byclass 0-9
    Sample011-036 → uppercase A-Z  → byclass 10-35
    Sample037-062 → lowercase a-z  → byclass 36-61
"""

from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import EMNIST
from PIL import Image

# Paths
DATA_DIR      = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
KAGGLE_DIR    = Path(r"E:\CSC-114\emnist-model\datasets\kaggle")
KAGGLE_IMGS   = KAGGLE_DIR / "az_images.npy"
KAGGLE_LBLS   = KAGGLE_DIR / "az_labels.npy"
CHARS74K_HND  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishHnd\English\Hnd\Img")
CHARS74K_IMG  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishImg\English\Img\GoodImg\Bmp")

NUM_CLASSES = 62

# EMNIST Balanced → byclass mapping
BALANCED_TO_BYCLASS = {}
for i in range(10):
    BALANCED_TO_BYCLASS[i] = i
for i in range(26):
    BALANCED_TO_BYCLASS[10 + i] = 10 + i

# Kaggle A-Z → byclass (uppercase only, 0-25 → 10-35)
KAGGLE_TO_BYCLASS = {i: i + 10 for i in range(26)}

# Chars74K Sample folder → byclass index
# Sample001=0(digit 0), Sample002=1(digit 1)...Sample010=9(digit 9)
# Sample011=10(A)...Sample036=35(Z)
# Sample037=36(a)...Sample062=61(z)
CHARS74K_TO_BYCLASS = {f"Sample{i+1:03d}": i for i in range(62)}


# =============================================================================
# Dataset wrappers
# =============================================================================

class KaggleAZDataset(Dataset):
    """Kaggle A-Z — uppercase only (byclass 10-35)."""
    def __init__(self, transform=None):
        if not KAGGLE_IMGS.exists() or not KAGGLE_LBLS.exists():
            raise FileNotFoundError(f"Kaggle data not found. Run download_datasets.py first.")
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
        valid_indices = []
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

    Folder structure:
        <root>/Sample001/img001-001.png  (digit 0)
        <root>/Sample011/img011-001.png  (uppercase A)
        <root>/Sample037/img037-001.png  (lowercase a)

    Maps Sample001-062 directly to EMNIST byclass indices 0-61.
    Covers all 62 classes including both upper AND lowercase —
    which Kaggle A-Z does not provide.
    """
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples   = []  # list of (image_path, byclass_label)

        if not root.exists():
            raise FileNotFoundError(f"Chars74K not found at {root}")

        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            folder_name = folder.name  # e.g. "Sample001"
            if folder_name not in CHARS74K_TO_BYCLASS:
                continue
            label = CHARS74K_TO_BYCLASS[folder_name]
            for img_path in sorted(folder.glob("*.png")):
                self.samples.append((img_path, label))
            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append((img_path, label))
            for img_path in sorted(folder.glob("*.bmp")):
                self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        """Return all labels as a list — used by get_class_weights."""
        return [s[1] for s in self.samples]


# =============================================================================
# get_class_weights — handles ConcatDataset correctly
# =============================================================================

def get_class_weights(dataset) -> torch.Tensor:
    """
    Compute WeightedRandomSampler weights for any dataset type.
    Handles: Subset, ConcatDataset, EMNIST, Chars74K, Kaggle, Balanced.
    Fixed v2: ConcatDataset no longer causes AttributeError.
    """
    print("[Dataset] Computing class weights for balanced sampling...")
    targets = _extract_targets(dataset)
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES).float()
    class_counts  = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    print(f"[Dataset] Class weight range: {class_weights.min():.6f} — {class_weights.max():.6f}")
    return sample_weights


def _extract_targets(dataset) -> torch.Tensor:
    """Recursively extract all labels from any dataset structure."""
    # ConcatDataset — recurse into each sub-dataset
    if isinstance(dataset, ConcatDataset):
        all_targets = []
        for ds in dataset.datasets:
            all_targets.append(_extract_targets(ds))
        return torch.cat(all_targets)

    # Subset — index into parent dataset targets
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent = dataset.dataset
        if hasattr(parent, 'targets'):
            return torch.tensor(
                [int(parent.targets[i]) for i in dataset.indices], dtype=torch.long
            )
        # Parent is itself a custom dataset — recurse
        all_t = _extract_targets(parent)
        return all_t[list(dataset.indices)]

    # Chars74K
    if isinstance(dataset, Chars74KDataset):
        return torch.tensor(dataset.get_labels(), dtype=torch.long)

    # Kaggle A-Z
    if isinstance(dataset, KaggleAZDataset):
        return torch.tensor(dataset.labels.tolist(), dtype=torch.long)

    # EMNIST Balanced wrapper
    if isinstance(dataset, BalancedEMNISTDataset):
        return torch.tensor(dataset.remapped_labels, dtype=torch.long)

    # Standard torchvision dataset with .targets
    if hasattr(dataset, 'targets'):
        return torch.tensor([int(t) for t in dataset.targets], dtype=torch.long)

    raise ValueError(f"Cannot extract targets from dataset type: {type(dataset)}")


# =============================================================================
# Main loader
# =============================================================================

def load_supplementary(
    transform,
    use_balanced:  bool = True,
    use_kaggle:    bool = True,
    use_chars_hnd: bool = True,
    use_chars_img: bool = True,
    train:         bool = True,
) -> Optional[ConcatDataset]:
    """
    Load all available supplementary datasets.
    Gracefully skips any dataset that is missing or fails to load.
    Falls back to EMNIST byclass only if nothing is available.
    """
    datasets = []
    total    = 0

    if use_balanced:
        try:
            ds = BalancedEMNISTDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] EMNIST Balanced: {len(ds):,} samples added")
        except Exception as e:
            print(f"  [Supplementary] EMNIST Balanced skipped: {e}")

    if use_kaggle and train:
        try:
            ds = KaggleAZDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Kaggle A-Z (uppercase only): {len(ds):,} samples added")
        except FileNotFoundError:
            print("  [Supplementary] Kaggle A-Z skipped — run download_datasets.py first")
        except Exception as e:
            print(f"  [Supplementary] Kaggle A-Z skipped: {e}")

    if use_chars_hnd and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_HND, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K EnglishHnd (both cases): {len(ds):,} samples added")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K EnglishHnd skipped — not found at {CHARS74K_HND}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K EnglishHnd skipped: {e}")

    if use_chars_img and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_IMG, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K EnglishImg (both cases): {len(ds):,} samples added")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K EnglishImg skipped — not found at {CHARS74K_IMG}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K EnglishImg skipped: {e}")

    if not datasets:
        print("  [Supplementary] No supplementary data available — using EMNIST byclass only")
        return None

    print(f"  [Supplementary] Total supplementary samples: {total:,}")
    return ConcatDataset(datasets)


def get_combined_weights(byclass_dataset, supplementary_dataset) -> torch.Tensor:
    """Compute weights for combined byclass + supplementary dataset."""
    byclass_targets = _extract_targets(byclass_dataset)
    supp_targets    = _extract_targets(supplementary_dataset) if supplementary_dataset else torch.tensor([], dtype=torch.long)
    all_targets     = torch.cat([byclass_targets, supp_targets])
    class_counts    = torch.bincount(all_targets, minlength=NUM_CLASSES).float()
    class_counts    = torch.clamp(class_counts, min=1)
    class_weights   = 1.0 / class_counts
    sample_weights  = class_weights[all_targets]
    print(f"  [Weights] Combined: {len(all_targets):,} samples")
    print(f"  [Weights] Class weight range: {class_weights.min():.6f} — {class_weights.max():.6f}")
    return sample_weights
