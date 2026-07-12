"""
mnist_adamw_64.py
=================
MNIST OCR — OCRConvNetWide, Schedule-Free AdamW, 64×64
Pure PyTorch — Wider architecture with Squeeze-Excitation attention blocks.

Architecture — OCRConvNetWide:
  Filter progression: 32→128→256→512
  Squeeze-Excitation (SE) attention after each stage
  StochasticDepth (DropPath) regularization
  Classifier head: 512→256→10
  ~9.7M parameters, ~37.1 MB (float32)

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 3  — PyTorch nn.Module, tensors, backward(), optimizer.step()
  Ch. 5  — Dropout, weight decay, data augmentation as regularization
  Ch. 6  — Universal ML workflow
  Ch. 8  — ConvNet architecture, filter progression, GlobalAveragePooling
  Ch. 9  — BatchNormalization, residual connections, depthwise separable convs
  Ch. 18 — Mixed-precision (AMP), model ensembling, int8 quantization

OPTIMIZER — Schedule-Free AdamW:
  Won MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge.
  Eliminates the learning rate scheduler entirely through iterate averaging.
  Uses Polyak-Ruppert averaging to replace the warmup+decay schedule with
  a single constant LR that adapts internally.
  Requires: pip install schedulefree
  Key advantages:
    - No scheduler tuning required — no T_0, no pct_start, no warmup epochs
    - No warm restart disruptions
    - Anytime optimization — best checkpoint at any epoch, not just at end
    - Internal warmup built in via warmup_steps parameter
  Usage:
    - Must call optimizer.train() before each training epoch
    - Must call optimizer.eval() before validation/evaluation
    - These toggle between update mode and averaging mode
  BatchNorm requirement:
    - Before each eval pass, run 50-batch warm-up with model.train() +
      optimizer.eval() to update BatchNorm running stats at the averaged
      parameter point. Required per Schedule-Free docs for any BatchNorm model.

Hardware (confirmed):
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB  — batch=512 override, 13.6/14.4GB VRAM peak, 62–69°C

Output: E:\\CSC-114\\project\\adamw_64\\
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import sys
import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST

# Schedule-Free AdamW — pip install schedulefree
try:
    import schedulefree
    HAS_SCHEDULEFREE = True
except ImportError:
    HAS_SCHEDULEFREE = False
    print("[Warning] schedulefree not installed — falling back to AdamW+CosineAnnealingLR")
    print("          Run: pip install schedulefree")

# Hardware monitoring — pip install psutil
try:
    import psutil
    psutil.cpu_percent()  # verify it actually works, not just imports
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from supplementary_data import load_supplementary, get_combined_weights
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — digit supplementary data unavailable")


# =============================================================================
# 0b. AUTOMATIC CLI OUTPUT LOGGING
# =============================================================================
# Mirrors every console line to a persistent .txt file automatically, so the
# full raw CLI transcript is captured for every run without manually copying
# terminal output afterward. Appends across runs (with a timestamped session
# header) so restarts/reruns accumulate in one file rather than overwriting.

class _Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        self.log.write(f"\n{'='*70}\n[Session start] "
                        f"{datetime.now().isoformat(timespec='seconds')}\n{'='*70}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # flush immediately so a crash/kill doesn't lose the tail

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 10
# Schedule-Free AdamW uses similar LR range to standard AdamW
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
VALIDATION_SPLIT = 0.15
PATIENCE         = 15
import platform
NUM_WORKERS      = 4  # parallel prefetch for training throughput
USE_AMP          = True

DATA_DIR         = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
OUTPUT_ROOT      = Path(r"E:\CSC-114\project\adamw_64")

LABEL_MAP = list("0123456789")

# Resolutions trained automatically, in order, in a single run.
# 32x32 runs first (fast, full convergence room), then 64x64.
# Fully independent — no weights, optimizer state, or history carried
# between resolutions. Each is a complete retrain from random init.
RESOLUTIONS = [64]


def get_hw_stats() -> dict:
    """
    Returns a dict of hardware utilization metrics for per-epoch logging.
    GPU stats via torch.cuda + nvidia-smi. CPU/RAM via psutil if available.
    Falls back gracefully if any source is unavailable.
    """
    import subprocess
    stats = {
        "vram_peak_alloc_gb":    round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else -1,
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved()  / 1024**3, 3) if torch.cuda.is_available() else -1,
        "cuda_util_pct":    -1,
        "gpu_temp_c":       -1,
        "cpu_pct":          -1,
        "ram_used_gb":      -1,
        "ram_total_gb":     -1,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(",")
        stats["cuda_util_pct"] = int(parts[0].strip())
        stats["gpu_temp_c"]    = int(parts[1].strip())
    except Exception:
        pass
    if HAS_PSUTIL:
        try:
            stats["cpu_pct"]      = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats["ram_used_gb"]  = round(vm.used  / 1024**3, 2)
            stats["ram_total_gb"] = round(vm.total / 1024**3, 2)
        except Exception:
            pass
    return stats


def determine_batch_size(model_cls, img_size: int, device: torch.device,
                         candidates=(1024, 512, 256)) -> int:
    """
    Automatically determines the largest safe batch size by attempting a forward
    + backward pass at each candidate size. Steps down on OOM. The determined
    batch size is used for all dataloaders at this resolution.
    """
    print(f"[Batch] Auto-detecting batch size for {img_size}x{img_size}...")
    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            # Pre-flight: skip if free VRAM < 1.5GB to avoid shared memory spillover
            _free_vram = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_reserved()) / 1024**3
            if _free_vram < 1.5:
                print(f'[Batch] Batch size {bs} — insufficient free VRAM ({_free_vram:.1f}GB free), skipping')
                continue
            model_test = model_cls(NUM_CLASSES).to(device)
            dummy = torch.zeros(bs, 1, img_size, img_size, device=device)
            out   = model_test(dummy)
            loss  = out.sum()
            loss.backward()
            # Check actual VRAM via nvidia-smi — catches Windows shared memory spillover
            try:
                import subprocess
                _smi = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    timeout=3
                ).decode().strip()
                _used_mb = float(_smi.split('\n')[0].strip())
                if _used_mb > 15000:  # 15GB threshold on 16GB card
                    raise RuntimeError('out of memory')
            except subprocess.SubprocessError:
                if torch.cuda.max_memory_allocated() / 1024**3 > 15.0:
                    raise RuntimeError('out of memory')
            del model_test, dummy, out, loss
            torch.cuda.empty_cache()
            # Check for silent shared VRAM spillover (Windows)
            _alloc_mb = torch.cuda.memory_allocated() / 1024**2
            _total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            if _alloc_mb > _total_mb * 0.95:
                del model_test, dummy, out, loss
                torch.cuda.empty_cache()
                print(f'[Batch] Batch size {bs} — spilling to shared VRAM, stepping down')
                continue
            print(f"[Batch] Batch size {bs} — OK")
            return bs
        except RuntimeError as e:
            _emsg = str(e).lower()
            if "out of memory" in _emsg or "find was unable" in _emsg or "engine" in _emsg:
                torch.cuda.empty_cache()
                print(f"[Batch] Batch size {bs} — failed ({type(e).__name__}), trying next")
            else:
                raise
    print(f"[Batch] All candidates exhausted — using minimum {candidates[-1]}")
    return candidates[-1]


def build_config(img_size: int, batch_size: int) -> dict:
    """
    Builds all resolution-dependent settings and output paths for a single run.
    Batch size is passed in from determine_batch_size() rather than hardcoded.
    """
    return {
        "img_size":    img_size,
        "batch_size":  batch_size,
        "checkpoint_path":  str(OUTPUT_ROOT / f"v1_adamw_64_best_{img_size}.pt"),
        "final_model_path": str(OUTPUT_ROOT / f"v1_adamw_64_final_{img_size}.pt"),
        "onnx_path":        str(OUTPUT_ROOT / f"v1_adamw_64_{img_size}.onnx"),
        "log_path":         str(OUTPUT_ROOT / f"v1_adamw_64_log_{img_size}.csv"),
        "plot_path":        str(OUTPUT_ROOT / f"v1_adamw_64_curves_{img_size}.png"),
        "resume_path":      str(OUTPUT_ROOT / f"v1_adamw_64_resume_{img_size}.json"),
    }


# =============================================================================
# 2. DEVICE SETUP
# =============================================================================

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1024**3
        print(f"[Device] {props.name}  |  {vram:.1f} GB VRAM  |  "
              f"CUDA {torch.version.cuda}  |  AMP: {USE_AMP}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU — {torch.get_num_threads()} threads")
    return device


# =============================================================================
# 3. DATA PIPELINE
# =============================================================================

def get_transforms(img_size: int, augment: bool = False) -> transforms.Compose:
    """
    v3: Same augmentation as v2 — rotation ±5°, shear 5°, blur.
    Schedule-Free optimizer handles learning dynamics differently but
    augmentation strategy is retained from v2 analysis.
    v4: [0,1] normalization — Normalize(0.5, 0.5) removed.
    """
    aug_transforms = [
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=5,
        ),
        transforms.ColorJitter(contrast=0.3, brightness=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.3),
    ] if augment else []

    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(aug_transforms + base_transforms)


def get_class_weights(dataset) -> torch.Tensor:
    """Handles ConcatDataset correctly via supplementary_data."""
    print("[Dataset] Computing class weights for balanced sampling...")
    if HAS_SUPPLEMENTARY:
        from supplementary_data import _extract_targets
        targets = _extract_targets(dataset)
    else:
        if hasattr(dataset, "datasets"):
            all_t = []
            for ds in dataset.datasets:
                if hasattr(ds, "dataset") and hasattr(ds.dataset, "targets"):
                    all_t.extend([int(ds.dataset.targets[i]) for i in ds.indices])
                elif hasattr(ds, "targets"):
                    all_t.extend([int(t) for t in ds.targets])
                elif hasattr(ds, "labels"):
                    all_t.extend(ds.labels.tolist())
                elif hasattr(ds, "remapped_labels"):
                    all_t.extend(ds.remapped_labels)
            targets = torch.tensor(all_t, dtype=torch.long)
        elif hasattr(dataset, "dataset"):
            targets = torch.tensor(
                [int(dataset.dataset.targets[i]) for i in dataset.indices], dtype=torch.long
            )
        else:
            targets = torch.tensor([int(t) for t in dataset.targets], dtype=torch.long)

    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES).float()
    class_counts  = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    print(f"[Dataset] Class weight range: {class_weights.min():.6f} — {class_weights.max():.6f}")
    return sample_weights


def load_mnist(data_dir: Path, img_size: int):
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[Dataset] Loading MNIST...")

    train_full = MNIST(root=str(data_dir), train=True,
                        download=True, transform=get_transforms(img_size, augment=True))
    test_ds    = MNIST(root=str(data_dir), train=False,
                        download=True, transform=get_transforms(img_size, augment=False))

    total       = len(train_full)
    val_count   = int(total * VALIDATION_SPLIT)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full, train_indices.indices)
    val_base = MNIST(root=str(data_dir), train=True,
                      download=False, transform=get_transforms(img_size, augment=False))
    val_ds   = Subset(val_base, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")

    # Load digit-only supplementary datasets
    supp_ds = None
    if HAS_SUPPLEMENTARY:
        print("[Dataset] Loading digit supplementary data...")
        supp_ds = load_supplementary(
            transform=get_transforms(img_size, augment=True),
            use_digits=True,
            use_mnist=True,
            use_usps=True,
            use_svhn=True,
            use_ardis=True,
            use_balanced=False,
            use_kaggle=False,
            use_chars_hnd=False,
            use_chars_img=False,
            use_pghwld=False,
            train=True,
        )
        if supp_ds is not None:
            train_ds = ConcatDataset([train_ds, supp_ds])
            print(f"[Dataset] Combined training set: {len(train_ds):,} samples")

    return train_ds, val_ds, test_ds, supp_ds


def make_dataloader(dataset, batch_size: int, shuffle: bool = False,
                    use_weighted_sampler: bool = False,
                    num_workers_override: int = None) -> DataLoader:
    _nw = num_workers_override if num_workers_override is not None else NUM_WORKERS
    if use_weighted_sampler:
        sample_weights = get_class_weights(dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=_nw, pin_memory=torch.cuda.is_available(),
            persistent_workers=False, drop_last=False,
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=_nw, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, drop_last=False,
    )


# =============================================================================
# 4. MODEL ARCHITECTURE — identical to v2
# =============================================================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class SEResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch)
        self.drop_path = StochasticDepth(drop_path)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = self.drop_path(x)
        return F.relu(x + residual, inplace=True)


class OCRConvNetWide(nn.Module):
    """
    Wider OCR ConvNet with SE attention.
    Input:  (batch, 1, 64, 64)
    Output: (batch, 62)
    Filter progression: 32→128→256→512
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(SEResidualBlock(32, 128),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(SEResidualBlock(128, 256), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(SEResidualBlock(256, 512), nn.MaxPool2d(2))
        self.stage4 = SEResidualBlock(512, 512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# =============================================================================
# 5. EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int, path: str):
        self.patience  = patience
        self.path      = path
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save({"state_dict": model.state_dict(), "val_loss": val_loss}, self.path)
            print(f"  [Checkpoint] val_loss → {val_loss:.4f}  saved")
        else:
            self.counter += 1
            print(f"  [EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                print("  [EarlyStopping] Halting.")


# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> tuple:
    """
    v3: Schedule-Free AdamW requires optimizer.train() to be called before
    each training epoch to switch from averaging mode to update mode.
    This is handled in main() before calling this function.
    """
    model.train()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=USE_AMP and device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, per_class: bool = False) -> tuple:
    """
    v3: Schedule-Free AdamW requires optimizer.eval() before evaluation
    to switch to parameter averaging mode. This is handled in main().
    """
    model.eval()
    total_loss = total_correct = total_samples = 0

    if per_class:
        class_correct = torch.zeros(NUM_CLASSES)
        class_total   = torch.zeros(NUM_CLASSES)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if per_class:
            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    if per_class and class_total.sum() > 0:
        class_acc = class_correct / class_total.clamp(min=1)
        worst = class_acc.argsort()[:10]
        print("\n  [Per-Class] 10 worst-performing classes:")
        for idx in worst:
            print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
                  f"{class_acc[idx]*100:.1f}%  ({int(class_total[idx])} samples)")

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 7. LOGGING AND PLOTTING
# =============================================================================

def plot_history(history: dict, path: str, img_size: int):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"MNIST OCR ScheduleFree-AdamW {img_size}x{img_size} — Training History",
                 fontsize=12, fontweight="bold")
    ax1.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Saved to {path}")


def save_log(history: dict, path: str):
    fieldnames = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
        "epoch_time_s", "vram_peak_alloc_gb", "vram_peak_reserved_gb",
        "cuda_util_pct", "gpu_temp_c", "cpu_pct", "ram_used_gb",
    ]
    n = len(history["train_loss"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "epoch":            i + 1,
                "train_loss":       f"{history['train_loss'][i]:.6f}",
                "train_acc":        f"{history['train_acc'][i]:.6f}",
                "val_loss":         f"{history['val_loss'][i]:.6f}",
                "val_acc":          f"{history['val_acc'][i]:.6f}",
                "lr":               f"{history['lr'][i]:.8f}",
                "epoch_time_s":     history.get("epoch_time_s",    ["-1"]*n)[i],
                "vram_peak_alloc_gb":    history.get("vram_peak_alloc_gb",   ["-1"]*n)[i],
                "vram_peak_reserved_gb": history.get("vram_peak_reserved_gb",["-1"]*n)[i],
                "cuda_util_pct":    history.get("cuda_util_pct",   ["-1"]*n)[i],
                "gpu_temp_c":       history.get("gpu_temp_c",      ["-1"]*n)[i],
                "cpu_pct":          history.get("cpu_pct",         ["-1"]*n)[i],
                "ram_used_gb":      history.get("ram_used_gb",     ["-1"]*n)[i],
            })
    print(f"[Log] Saved to {path}")


# =============================================================================
# 8. SAVE / EXPORT
# =============================================================================

def save_model(model: nn.Module, path: str):
    torch.save({"state_dict": model.state_dict()}, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Save] {path}  ({size_mb:.1f} MB)")


def export_onnx(model: nn.Module, path: str, img_size: int):
    """
    IMPORTANT: inference normalization must be:
        arr = arr / 255.0
    (v4: [0,1] normalization, no mean/std shift)
    """
    model.eval()
    dummy = torch.zeros(1, 1, img_size, img_size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[ONNX] Exported to {path}  ({size_mb:.1f} MB)")



# =============================================================================
# 9. RUN TRAINING — ONE FULL RESOLUTION PASS
# =============================================================================

def run_training(img_size: int, batch_override: int = None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    if batch_override is not None:
        batch_size = batch_override
        print(f"[Batch] Override: using batch size {batch_size} (skipping auto-detect)")
    else:
        batch_size = determine_batch_size(OCRConvNetWide, img_size, device)
    cfg = build_config(img_size, batch_size)

    print("=" * 60)
    print(f"  MNIST OCR — ScheduleFree-AdamW  [{img_size}x{img_size}]")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Resolution: {img_size}x{img_size}  |  Batch: {cfg['batch_size']} {'(override)' if batch_override else '(auto-detected)'}")
    print(f"  Optimizer: {'Schedule-Free AdamW' if HAS_SCHEDULEFREE else 'AdamW+CosineAnnealingLR (fallback)'}")
    print(f"  Normalization: [0,1]  (v4 fix — no mean/std shift)")
    print("=" * 60)

    train_ds, val_ds, test_ds, supp_ds = load_mnist(DATA_DIR, img_size)
    train_loader = make_dataloader(train_ds, cfg["batch_size"], use_weighted_sampler=True)
    val_loader   = make_dataloader(val_ds,  cfg["batch_size"], num_workers_override=0)
    test_loader  = make_dataloader(test_ds, cfg["batch_size"], num_workers_override=0)

    model = OCRConvNetWide(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetWide (Model 2) — {img_size}x{img_size}")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Schedule-Free AdamW — no scheduler needed, LR managed internally
    if HAS_SCHEDULEFREE:
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=len(train_loader),  # 1 epoch warmup
        )
        print(f"[Optimizer] Schedule-Free AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
        print(f"            warmup_steps={len(train_loader)} (1 epoch)")
        scheduler = None
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
        print(f"[Optimizer] AdamW+CosineAnnealingLR (fallback — install schedulefree)")

    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=cfg["checkpoint_path"])

    print(f"\n[Train] Starting {img_size}x{img_size} — no epoch cap | batch: {cfg['batch_size']} | wall limit: 10h | patience: {PATIENCE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    # ── Resume from previous session ─────────────────────────────────────────
    start_epoch = 1
    resume_path = Path(cfg["resume_path"].replace(".json", ".pt"))
    if resume_path.exists() and Path(cfg["checkpoint_path"]).exists():
        try:
            _rs = torch.load(str(resume_path), map_location=device, weights_only=False)
            _ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt)
            optimizer.load_state_dict(_rs["optimizer_state"])
            if scheduler is not None and _rs.get("scheduler_state") is not None:
                scheduler.load_state_dict(_rs["scheduler_state"])
            scaler.load_state_dict(_rs["scaler_state"])
            early_stop.counter   = _rs["patience_counter"]
            early_stop.best_loss = _rs["best_val_loss"]
            history              = _rs["history"]
            start_epoch          = _rs["epoch"] + 1
            print(f"[Resume] Loaded from epoch {_rs['epoch']} "
                  f"(val_loss={_rs['best_val_loss']:.4f}, patience={_rs['patience_counter']}/{PATIENCE})")
            print(f"[Resume] Continuing from epoch {start_epoch}")
        except Exception as _e:
            print(f"[Resume] Could not load state: {_e} — starting fresh")
            start_epoch = 1
    else:
        print("[Resume] No prior checkpoint found — starting fresh")

    wall_start = time.time()
    for epoch in range(start_epoch, 10**6):
        t0 = time.time()

        # Schedule-Free requires train() call before training to switch to update mode
        if HAS_SCHEDULEFREE:
            optimizer.train()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Schedule-Free requires eval() call before validation to switch to averaging mode
        if HAS_SCHEDULEFREE:
            # BatchNorm warm-up: update running stats at averaged parameter point x
            model.train()
            optimizer.eval()
            with torch.no_grad():
                for _imgs, _ in itertools.islice(train_loader, 50):
                    model(_imgs.to(device))
            model.eval()
        else:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0
        hw         = get_hw_stats()

        print(f"[{img_size}x{img_size}] Epoch {epoch:3d}  "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  |  "
              f"lr: {current_lr:.2e}  [{elapsed:.0f}s]  |  "
              f"VRAM {hw['vram_peak_alloc_gb']:.1f}/{hw['vram_peak_reserved_gb']:.1f}GB  "
              f"CUDA {hw['cuda_util_pct']}%  {hw['gpu_temp_c']}°C  |  "
              f"CPU {hw['cpu_pct']}%  RAM {hw['ram_used_gb']:.1f}/{hw['ram_total_gb']:.1f}GB")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("lr", current_lr)]:
            history[k].append(v)
        history.setdefault("vram_peak_alloc_gb",    []).append(hw["vram_peak_alloc_gb"])
        history.setdefault("vram_peak_reserved_gb", []).append(hw["vram_peak_reserved_gb"])
        history.setdefault("cuda_util_pct",    []).append(hw["cuda_util_pct"])
        history.setdefault("gpu_temp_c",       []).append(hw["gpu_temp_c"])
        history.setdefault("cpu_pct",          []).append(hw["cpu_pct"])
        history.setdefault("ram_used_gb",      []).append(hw["ram_used_gb"])
        history.setdefault("epoch_time_s",     []).append(round(elapsed, 1))

        early_stop(val_loss, model)
        if early_stop.stop:
            break

        # Save resume state after every epoch
        try:
            import torch as _torch
            _rpath = cfg["resume_path"].replace(".json", ".pt")
            _torch.save({
                "epoch":            epoch,
                "patience_counter": early_stop.counter,
                "best_val_loss":    float(early_stop.best_loss),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict() if scheduler is not None else None,
                "scaler_state":     scaler.state_dict(),
                "history":          history,
            }, _rpath)
        except Exception as _re:
            print(f"  [Resume] Warning: could not save state: {_re}")

        wall_elapsed = time.time() - wall_start
        if wall_elapsed >= 36000:
            print(f"  [Wall clock] 10-hour limit reached after epoch {epoch} "
                  f"({wall_elapsed/3600:.2f}h) — stopping cleanly.")
            break

    print(f"\n[Train] [{img_size}x{img_size}] Loading best checkpoint...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print(f"\n[Eval] [{img_size}x{img_size}] Running per-class accuracy analysis on test set...")
    if HAS_SCHEDULEFREE:
        model.train()
        optimizer.eval()
        with torch.no_grad():
            for _imgs, _ in itertools.islice(train_loader, 50):
                model(_imgs.to(device))
        model.eval()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  [{img_size}x{img_size}] Model 2 Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  [{img_size}x{img_size}] Model 2 Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    plot_history(history, cfg["plot_path"], img_size)
    save_log(history, cfg["log_path"])
    save_model(model, cfg["final_model_path"])

    try:
        model_cpu = OCRConvNetWide(NUM_CLASSES)
        model_cpu.load_state_dict(
            torch.load(cfg["final_model_path"], map_location="cpu", weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu, cfg["onnx_path"], img_size)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")

    # Clean up resume state — training completed successfully
    resume_p = Path(cfg["resume_path"].replace(".json", ".pt"))
    if resume_p.exists():
        resume_p.unlink()
        print(f"  [Resume] State cleared — training complete.")
    print(f"\n[Done] [{img_size}x{img_size}] All files saved to {OUTPUT_ROOT}")
    return test_acc


# =============================================================================
# 10. MAIN — RUNS ALL RESOLUTIONS BACK TO BACK, FULLY AUTOMATIC
# =============================================================================

def main(batch_override=None):
    print("\n" + "#" * 60)
    print(f"  MNIST ScheduleFree-AdamW 64x64 — SINGLE RESOLUTION TRAINING")
    print(f"  Resolutions queued: {RESOLUTIONS}")
    print(f"  Each is a full independent retrain — no weight transfer")
    print("#" * 60 + "\n")

    results = {}
    for img_size in RESOLUTIONS:
        acc = run_training(img_size, batch_override=batch_override)
        results[img_size] = acc

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE — SCHEDULEFREE-ADAMW 64x64")
    for img_size, acc in results.items():
        print(f"  {img_size}x{img_size}: {acc*100:.2f}% test accuracy")
    print("#" * 60)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--batch-size', type=int, default=None,
                         help='Override auto batch detection with a fixed batch size')
    _args = _parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(OUTPUT_ROOT / f"v1_adamw_64_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    main(batch_override=_args.batch_size)