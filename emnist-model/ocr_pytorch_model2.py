"""
ocr_pytorch_model2.py
=====================
EMNIST OCR — Model 2 for Ensemble
Pure PyTorch — Wider architecture with Squeeze-Excitation attention blocks.

Architectural differences from Model 1 (intentional diversity for ensemble):
  - Wider filter progression: 32→128→256→512 vs 32→64→128→256
  - Squeeze-Excitation (SE) attention after each stage
  - StochasticDepth (DropPath) regularization
  - Larger classifier head: 512→256

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
    - Internal warmup built in via momentum ramp
  Usage difference from standard Adam:
    - Must call optimizer.train() before training loop
    - Must call optimizer.eval() before validation/evaluation
    - These toggle between update mode and averaging mode

CORRECTIONS APPLIED (v3):
  - Optimizer changed from AdamW+CosineAnnealingLR to Schedule-Free AdamW
  - No scheduler needed — Schedule-Free handles LR internally
  - PATIENCE increased to 15
  - RandomRotation ±5°, shear 5° (v2 fixes retained)
  - WeightedRandomSampler retained
  - label_smoothing=0.05 retained
  - Blur augmentation retained

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB  — batch=256, ~8GB VRAM

Output: E:\\CSC-114\\emnist-model\\pytorch2\\
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import csv
import time
from pathlib import Path

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
from torchvision.datasets import EMNIST

# Schedule-Free AdamW — pip install schedulefree
try:
    import schedulefree
    HAS_SCHEDULEFREE = True
except ImportError:
    HAS_SCHEDULEFREE = False
    print("[Warning] schedulefree not installed — falling back to AdamW+CosineAnnealingLR")
    print("          Run: pip install schedulefree")

# Supplementary datasets
try:
    from supplementary_data import load_supplementary, get_combined_weights
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — using EMNIST byclass only")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 62

IMG_SIZE         = 64
IMG_HEIGHT       = IMG_SIZE
IMG_WIDTH        = IMG_SIZE

BATCH_SIZE       = 512 if IMG_SIZE == 32 else 256
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS           = 50
# Schedule-Free AdamW uses similar LR range to standard AdamW
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
VALIDATION_SPLIT = 0.15
PATIENCE         = 15
NUM_WORKERS      = 8
USE_AMP          = True

BASE_DIR         = Path(r"E:\CSC-114\emnist-model\pytorch2")
DATA_DIR         = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
CHECKPOINT_PATH  = str(BASE_DIR / "best_model2.pt")
FINAL_MODEL_PATH = str(BASE_DIR / "final_model2.pt")
ONNX_PATH        = str(BASE_DIR / "ocr_model2.onnx")
LOG_PATH         = str(BASE_DIR / "training_log2.csv")
PLOT_PATH        = str(BASE_DIR / "training_curves2.png")

LABEL_MAP = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz")
)


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

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    v3: Same augmentation as v2 — rotation ±5°, shear 5°, blur.
    Schedule-Free optimizer handles learning dynamics differently but
    augmentation strategy is retained from v2 analysis.
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
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
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


def load_emnist(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[Dataset] Loading EMNIST byclass...")

    train_full = EMNIST(root=str(data_dir), split="byclass", train=True,
                        download=True, transform=get_transforms(augment=True))
    test_ds    = EMNIST(root=str(data_dir), split="byclass", train=False,
                        download=True, transform=get_transforms(augment=False))

    total       = len(train_full)
    val_count   = int(total * VALIDATION_SPLIT)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full, train_indices.indices)
    val_base = EMNIST(root=str(data_dir), split="byclass", train=True,
                      download=False, transform=get_transforms(augment=False))
    val_ds   = Subset(val_base, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")

    supp_ds = None
    if HAS_SUPPLEMENTARY:
        print("[Dataset] Loading supplementary data...")
        supp_ds = load_supplementary(
            transform=get_transforms(augment=True),
            use_balanced=True,
            use_digits=True,
            use_mnist=True,
            use_usps=True,
            use_svhn=True,
            use_kaggle=True,
            use_chars_hnd=True,
            use_chars_img=True,
            train=True,
        )
        if supp_ds is not None:
            train_ds = ConcatDataset([train_ds, supp_ds])
            print(f"[Dataset] Combined training set: {len(train_ds):,} samples")

    return train_ds, val_ds, test_ds, supp_ds


def make_dataloader(dataset, shuffle: bool = False,
                    use_weighted_sampler: bool = False) -> DataLoader:
    if use_weighted_sampler:
        sample_weights = get_class_weights(dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0), drop_last=False,
        )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0), drop_last=False,
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
        worst = class_acc.argsort()[:15]
        print("\n  [Per-Class] 15 worst-performing classes:")
        for idx in worst:
            print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
                  f"{class_acc[idx]*100:.1f}%  ({int(class_total[idx])} samples)")

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 7. LOGGING AND PLOTTING
# =============================================================================

def plot_history(history: dict):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR Model 2 (Schedule-Free AdamW) — Training History",
                 fontsize=12, fontweight="bold")
    ax1.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150); plt.close()
    print(f"[Plot] Saved to {PLOT_PATH}")


def save_log(history: dict):
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history.keys())
        w.writeheader()
        for row in zip(*history.values()):
            w.writerow(dict(zip(history.keys(), row)))
    print(f"[Log] Saved to {LOG_PATH}")


# =============================================================================
# 8. SAVE / EXPORT
# =============================================================================

def save_model(model: nn.Module, path: str = FINAL_MODEL_PATH):
    torch.save({"state_dict": model.state_dict()}, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Save] {path}  ({size_mb:.1f} MB)")


def export_onnx(model: nn.Module, path: str = ONNX_PATH):
    """
    IMPORTANT: inference normalization must be:
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5
    """
    model.eval()
    dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[ONNX] Exported to {path}  ({size_mb:.1f} MB)")


# =============================================================================
# 9. ENSEMBLE INFERENCE
# =============================================================================

def ensemble_predict_loader(model1, model2, loader, device) -> tuple:
    model1.eval(); model2.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            p1  = F.softmax(model1(images), dim=1)
            p2  = F.softmax(model2(images), dim=1)
            avg = (p1 + p2) / 2.0
            all_preds.append(avg.argmax(1).cpu())
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Model 2 (v3 — Schedule-Free AdamW)")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {BASE_DIR}")
    print(f"  Resolution: {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    print(f"  Optimizer: {'Schedule-Free AdamW' if HAS_SCHEDULEFREE else 'AdamW+CosineAnnealingLR (fallback)'}")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    train_ds, val_ds, test_ds, supp_ds = load_emnist(DATA_DIR)
    train_loader = make_dataloader(train_ds, use_weighted_sampler=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    model = OCRConvNetWide(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetWide (Model 2)")
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        print(f"[Optimizer] AdamW+CosineAnnealingLR (fallback — install schedulefree)")

    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=CHECKPOINT_PATH)

    print(f"\n[Train] Starting — max epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Schedule-Free requires train() call before training to switch to update mode
        if HAS_SCHEDULEFREE:
            optimizer.train()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Schedule-Free requires eval() call before validation to switch to averaging mode
        if HAS_SCHEDULEFREE:
            optimizer.eval()
        else:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  |  "
              f"lr: {current_lr:.2e}  [{elapsed:.0f}s]")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("lr", current_lr)]:
            history[k].append(v)

        early_stop(val_loss, model)
        if early_stop.stop:
            break

    print(f"\n[Train] Loading best checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print("\n[Eval] Running per-class accuracy analysis on test set...")
    if HAS_SCHEDULEFREE:
        optimizer.eval()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  Model 2 Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Model 2 Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    plot_history(history)
    save_log(history)
    save_model(model)

    try:
        model_cpu = OCRConvNetWide(NUM_CLASSES)
        model_cpu.load_state_dict(
            torch.load(FINAL_MODEL_PATH, map_location="cpu", weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")

    model1_path = Path(r"E:\CSC-114\emnist-model\pytorch\best_model.pt")
    if model1_path.exists():
        try:
            import sys
            sys.path.insert(0, str(Path(r"E:\CSC-114\emnist-model")))
            from ocr_pytorch_model import OCRConvNet
            model1 = OCRConvNet(NUM_CLASSES)
            ckpt1  = torch.load(str(model1_path), map_location="cpu", weights_only=False)
            model1.load_state_dict(ckpt1["state_dict"])
            model1 = model1.to(device)
            preds, labels = ensemble_predict_loader(model1, model, test_loader, device)
            ensemble_acc = (preds == labels).mean()
            print(f"\n{'='*40}")
            print(f"  Model 1 alone : see pytorch/training_log.csv")
            print(f"  Model 2 alone : {test_acc*100:.2f}%")
            print(f"  ENSEMBLE M1+M2: {ensemble_acc*100:.2f}%")
            print(f"{'='*40}")
        except Exception as e:
            print(f"[Ensemble] Could not load Model 1: {e}")

    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()