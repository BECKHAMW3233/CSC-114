"""
ocr_pytorch_model3.py
=====================
EMNIST OCR — Model 3 for Ensemble (Maximum Capacity)
Pure PyTorch — Triple-width channels + Multi-Scale feature fusion +
               Deep classifier ensemble head.

Architectural differences from Models 1 and 2:
  CHANNELS (3x Model 1):
    Stem:    1→96
    Stage 1: 96→192
    Stage 2: 192→384
    Stage 3: 384→768
    Stage 4: 768→768

  MULTI-SCALE FUSION:
    Feature pyramid: concatenates pooled outputs from stages 2+3+4
    before the classifier.

  DEEP CLASSIFIER (5 layers):
    768_fused→1024→512→256→128→62

  GELU activations in classifier (vs ReLU in Models 1+2)
  SGD + Momentum + CosineAnnealingWarmRestarts

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 3, 5, 6, 8, 9, 18

CORRECTIONS APPLIED (v2):
  - RandomRotation reduced from ±15° to ±5° — prevents L/7 and H/I confusion
  - Shear reduced from 10° to 5°
  - WeightedRandomSampler added — addresses EMNIST byclass class imbalance
  - Domain-shift augmentation added (perspective + blur + noise) — Model 3's
    data diversity contribution, most aggressive of the three models
  - Per-class accuracy logging added

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB  — batch=256 for triple-width model

Output: E:\\CSC-114\\emnist-model\\pytorch3\\
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

# ── Resolution toggle ─────────────────────────────────────────────────────────
IMG_SIZE         = 64       # switch to 32 to revert to original resolution
# IMG_SIZE       = 32       # original — uncomment to use
IMG_HEIGHT       = IMG_SIZE
IMG_WIDTH        = IMG_SIZE

# Triple-width model is most VRAM-hungry — reduce batch further at 64x64
BATCH_SIZE       = 128 if IMG_SIZE == 64 else 256
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS           = 50
LEARNING_RATE    = 0.01
WEIGHT_DECAY     = 3e-5
MOMENTUM         = 0.9
VALIDATION_SPLIT = 0.15
PATIENCE         = 15
NUM_WORKERS      = 8
USE_AMP          = True

BASE_DIR         = Path(r"E:\CSC-114\emnist-model\pytorch3")
DATA_DIR         = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
CHECKPOINT_PATH  = str(BASE_DIR / "best_model3.pt")
FINAL_MODEL_PATH = str(BASE_DIR / "final_model3.pt")
ONNX_PATH        = str(BASE_DIR / "ocr_model3.onnx")
LOG_PATH         = str(BASE_DIR / "training_log3.csv")
PLOT_PATH        = str(BASE_DIR / "training_curves3.png")

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
# 3. DATA PIPELINE — domain-shift augmentation
# =============================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    FIX v2: Rotation reduced from ±15° to ±5°, shear from 10° to 5°.
    FIX v2: Added domain-shift augmentation — perspective distortion +
    GaussianBlur + random sharpness adjustment.

    Model 3's data diversity strategy (most aggressive of the three):
    - Model 1: clean EMNIST only
    - Model 2: clean + blur/noise
    - Model 3: clean + blur/noise + perspective distortion + sharpness variation
    This maximizes ensemble diversity at the data level.
    """
    aug_transforms = [
        transforms.RandomRotation(degrees=5),            # FIX: was 15, now 5
        transforms.RandomAffine(
            degrees=0,
            translate=(0.12, 0.12),
            scale=(0.80, 1.20),
            shear=5,                                      # FIX: was 10, now 5
        ),
        transforms.ColorJitter(contrast=0.4, brightness=0.15),
        # FIX v2: domain-shift augmentation
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(sharpness_factor=0, p=1.0),
        ], p=0.2),
    ] if augment else []

    base_transforms = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
    return transforms.Compose(aug_transforms + base_transforms)


def get_class_weights(dataset) -> torch.Tensor:
    """Fixed v2: handles ConcatDataset correctly via supplementary_data."""
    print("[Dataset] Computing class weights for balanced sampling...")
    if HAS_SUPPLEMENTARY:
        from supplementary_data import _extract_targets
        targets = _extract_targets(dataset)
    else:
        # Fallback for when supplementary_data.py is not present
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
            use_kaggle=True,
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
# 4. MODEL ARCHITECTURE — Triple-width + Multi-Scale Fusion
# =============================================================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        return x * self.fc(scale).view(b, c, 1, 1)


class BottleneckResidualBlock(nn.Module):
    """
    Bottleneck residual block — 1x1 compress → 3x3 conv → 1x1 expand.
    More parameter-efficient than standard 3x3→3x3 at triple-width channel counts.
    """
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.1):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = nn.Conv2d(in_ch,  mid,    1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid,    mid,    3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid,    out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch)
        self.drop_path_prob = drop_path
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        # StochasticDepth inline
        if self.training and self.drop_path_prob > 0:
            keep = 1 - self.drop_path_prob
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
            out  = out * mask.float() / keep
        return F.relu(out + residual, inplace=True)


class OCRConvNetTriple(nn.Module):
    """
    Triple-width OCR ConvNet with multi-scale feature fusion.
    Input:  (batch, 1, 32, 32)
    Output: (batch, 62)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BottleneckResidualBlock(96, 192),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(BottleneckResidualBlock(192, 384), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(BottleneckResidualBlock(384, 768), nn.MaxPool2d(2))
        self.stage4 = BottleneckResidualBlock(768, 768)

        # Multi-scale fusion: pool stages 2, 3, 4 and concatenate
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        fused_dim = 384 + 768 + 768  # = 1920

        # Deep 5-layer GELU classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        # Multi-scale fusion
        f2 = self.pool2(s2).flatten(1)
        f3 = self.pool3(s3).flatten(1)
        f4 = self.pool4(s4).flatten(1)
        fused = torch.cat([f2, f3, f4], dim=1)
        return self.classifier(fused)


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
    """FIX v2: Added per_class accuracy logging."""
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
    fig.suptitle("EMNIST OCR Model 3 — Training History", fontsize=12, fontweight="bold")
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
# 9. THREE-MODEL ENSEMBLE
# =============================================================================

def three_model_ensemble(model1, model2, model3, loader, device) -> tuple:
    for m in [model1, model2, model3]:
        m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            p1  = F.softmax(model1(images), dim=1)
            p2  = F.softmax(model2(images), dim=1)
            p3  = F.softmax(model3(images), dim=1)
            avg = (p1 + p2 + p3) / 3.0
            all_preds.append(avg.argmax(1).cpu())
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Model 3 (Triple + MultiScale)  v2")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {BASE_DIR}")
    print(f"  Resolution: {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    print("  Changes: rotation ±5°, shear 5°, WeightedRandomSampler,")
    print("           perspective + blur + sharpness domain augmentation")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    train_ds, val_ds, test_ds, supp_ds = load_emnist(DATA_DIR)
    train_loader = make_dataloader(train_ds, use_weighted_sampler=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    model = OCRConvNetTriple(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetTriple (Model 3)")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")
    print(f"  Batch size : {BATCH_SIZE}")

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer  = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                           momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=35, T_mult=2, eta_min=1e-4
    )
    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=CHECKPOINT_PATH)

    print(f"\n[Train] Starting — max epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
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
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  Model 3 Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Model 3 Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    plot_history(history)
    save_log(history)
    save_model(model)

    try:
        model_cpu = OCRConvNetTriple(NUM_CLASSES)
        model_cpu.load_state_dict(
            torch.load(FINAL_MODEL_PATH, map_location="cpu", weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")

    print(f"\n[Ensemble] Loading Models 1 and 2...")
    try:
        import sys
        sys.path.insert(0, str(Path(r"E:\CSC-114\emnist-model")))
        from ocr_pytorch_model  import OCRConvNet
        from ocr_pytorch_model2 import OCRConvNetWide

        m1_path = Path(r"E:\CSC-114\emnist-model\pytorch\best_model.pt")
        m2_path = Path(r"E:\CSC-114\emnist-model\pytorch2\best_model2.pt")

        model1 = OCRConvNet(NUM_CLASSES)
        model1.load_state_dict(
            torch.load(str(m1_path), map_location="cpu", weights_only=False)["state_dict"]
        )
        model1 = model1.to(device)

        model2 = OCRConvNetWide(NUM_CLASSES)
        model2.load_state_dict(
            torch.load(str(m2_path), map_location="cpu", weights_only=False)["state_dict"]
        )
        model2 = model2.to(device)

        preds, labels = three_model_ensemble(model1, model2, model, test_loader, device)
        acc_ensemble  = (preds == labels).mean()

        print(f"\n{'='*55}")
        print(f"  FINAL ENSEMBLE RESULTS")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  Model 3 alone  : {test_acc*100:.2f}%")
        print(f"  3-Model ensemble (equal weights): {acc_ensemble*100:.2f}%")
        print(f"{'='*55}")

    except Exception as e:
        print(f"[Ensemble] Error: {e}")

    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()
