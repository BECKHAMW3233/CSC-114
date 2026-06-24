"""
ocr_pytorch_model.py
====================
EMNIST OCR — Handwritten + Printed Character Recognition
Pure PyTorch implementation — no Keras, no TensorFlow dependency.

Recognizes: digits 0-9, A-Z, a-z  (62 classes)
Dataset   : EMNIST byclass — 814,255 samples

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 2  — Mathematical building blocks: tensors, matrix multiply, gradient descent,
            backpropagation, loss functions
  Ch. 3  — PyTorch introduction: tensors, Parameter class, backward(),
            optimizer.step(), zero_grad(), nn.Module (Listings 3.22-3.27)
  Ch. 5  — Overfitting / underfitting; Dropout, weight decay (L2), data augmentation
            as regularization strategies; validation set methodology
  Ch. 6  — Universal ML workflow: define problem → measure success → prepare data
            → build model → tune → evaluate on test set
  Ch. 8  — ConvNet architecture: Conv2D blocks, MaxPooling, GlobalAveragePooling,
            filter progression 32→64→128→256, data augmentation (Listings 8.1-8.31)
  Ch. 9  — BatchNormalization, residual connections, depthwise separable convolutions,
            model depth vs. width tradeoffs
  Ch. 18 — Mixed-precision training (torch.autocast + GradScaler), model ensembling,
            int8 quantization for faster inference, hyperparameter optimization

CORRECTIONS APPLIED (v2):
  - RandomRotation reduced from ±8° to ±5° — prevents L/7 and H/I confusion at 32x32
  - Shear reduced from 5° to 3° — same reason
  - WeightedRandomSampler added — addresses EMNIST byclass class imbalance
  - Per-class accuracy logging added to evaluate() — identifies per-class failures
  - Inference normalization documented: input must be (arr - 0.5) / 0.5 at inference

RESOLUTION OPTION (v2):
  - IMG_SIZE configurable: 32 (original) or 64 (recommended)
  - At 64x64: b/t, B/P, 2/6, 3/W shape separations become visible features
  - BATCH_SIZE auto-adjusts: 512 at 32x32, 256 at 64x64
  - Training time ~2-3x longer per epoch at 64x64
  - ONNX export and inference pipeline read IMG_SIZE from model input shape automatically

Hardware target:
    AMD Ryzen 9 7900X  (24 threads — used by DataLoader workers)
    64 GB DDR5-5600    (full EMNIST dataset cached in RAM after epoch 1)
    RTX 4080 16 GB     (BATCH_SIZE=512, AMP float16 via torch.autocast)

All output: E:\\CSC-114\\emnist-model\\pytorch\\

Run:
    E:\\CSC-114\\emnist-model\\venv\\Scripts\\activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install torchmetrics matplotlib pillow optuna
    python ocr_pytorch_model.py
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import csv
import time
import json
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
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST

# Supplementary datasets — EMNIST Balanced + Kaggle A-Z
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
# 64 recommended: resolves b/t, B/P, 2/6, 3/W shape-similarity failures
# 32 original: faster training, use for quick iteration
IMG_SIZE         = 64       # switch to 32 to revert to original resolution
# IMG_SIZE       = 32       # original — uncomment to use
IMG_HEIGHT       = IMG_SIZE
IMG_WIDTH        = IMG_SIZE

# Batch size auto-adjusts: 64x64 images are 4x memory of 32x32
BATCH_SIZE       = 256 if IMG_SIZE == 64 else 512
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS           = 50
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 3e-5
VALIDATION_SPLIT = 0.15
PATIENCE         = 12
NUM_WORKERS      = 8
USE_AMP          = True

BASE_DIR         = Path(r"E:\CSC-114\emnist-model\pytorch")
DATA_DIR         = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
CHECKPOINT_PATH  = str(BASE_DIR / "best_model.pt")
FINAL_MODEL_PATH = str(BASE_DIR / "final_model.pt")
ONNX_PATH        = str(BASE_DIR / "ocr_model.onnx")
QUANTIZED_PATH   = str(BASE_DIR / "ocr_model_quantized.pt")
LOG_PATH         = str(BASE_DIR / "training_log.csv")
PLOT_PATH        = str(BASE_DIR / "training_curves.png")

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
        print(f"[Device] CPU — {torch.get_num_threads()} threads available")
    return device


# =============================================================================
# 3. DATA TRANSFORMS AND PIPELINE
# =============================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Ch. 8 augmentation strategy.
    Ch. 5: augmentation is a form of regularization.

    FIX v2: Rotation reduced from ±8° to ±5°, shear from 5° to 3°.
    At 32x32 resolution, ±8° rotation rotates L into a position visually
    indistinguishable from 7. ±5° preserves enough diversity without
    destroying directional class boundaries (L/7, H/I, 1/l).
    """
    aug_transforms = [
        transforms.RandomRotation(degrees=5),          # FIX: was 8, now 5
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=3,                                    # FIX: was 5, now 3
        ),
        transforms.ColorJitter(contrast=0.2),
    ] if augment else []

    base_transforms = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        # Normalize to [-1, 1] — IMPORTANT: inference must apply same normalization
        # arr = arr / 255.0; arr = (arr - 0.5) / 0.5
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
    """
    Ch. 6 workflow step: prepare data.
    Downloads EMNIST byclass via torchvision (~540 MB first run).
    """
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

    # Load supplementary datasets — EMNIST Balanced + Kaggle A-Z
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
    """
    FIX v2: Added WeightedRandomSampler support for training loader.
    When use_weighted_sampler=True, overrides shuffle and samples classes equally.
    Val/test loaders always use shuffle=False, use_weighted_sampler=False.
    """
    if use_weighted_sampler:
        # Use get_combined_weights if supplementary data provided, else per-class weights
        sample_weights = get_class_weights(dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,           # mutually exclusive with shuffle
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Ch. 9 depthwise separable convolution."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn         = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pointwise(self.depthwise(x))), inplace=True)


class ResidualBlock(nn.Module):
    """Ch. 9 residual block with BatchNormalization."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class OCRConvNet(nn.Module):
    """
    OCR ConvNet — Ch. 8 architecture with Ch. 9 residual/BN improvements.
    Input:  (batch, 1, 32, 32)
    Output: (batch, 62)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(DepthwiseSeparableConv(1, 32))
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64), nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128), nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256), nn.MaxPool2d(2),
        )
        self.stage4    = ResidualBlock(256, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
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
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


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
            torch.save({"state_dict": model.state_dict(),
                        "val_loss": val_loss}, self.path)
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

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    scheduler, device) -> tuple:
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

        if scheduler is not None:
            scheduler.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             per_class: bool = False) -> tuple:
    """
    FIX v2: Added per_class accuracy logging.
    When per_class=True, prints per-class accuracy sorted by worst performers.
    Call with per_class=True on the test set to identify specific class failures.
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
# 7. LEARNING RATE SCHEDULING
# =============================================================================

def build_scheduler(optimizer, train_loader):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )


# =============================================================================
# 8. LOGGING AND PLOTTING
# =============================================================================

def plot_history(history: dict, path: str = PLOT_PATH):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR (PyTorch) — Training History", fontsize=12, fontweight="bold")
    ax1.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Saved to {path}")


def save_log(history: dict, path: str = LOG_PATH):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"
        ])
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow({
                "epoch":      i + 1,
                "train_loss": f"{history['train_loss'][i]:.6f}",
                "train_acc":  f"{history['train_acc'][i]:.6f}",
                "val_loss":   f"{history['val_loss'][i]:.6f}",
                "val_acc":    f"{history['val_acc'][i]:.6f}",
                "lr":         f"{history['lr'][i]:.8f}",
            })
    print(f"[Log] Saved to {path}")


# =============================================================================
# 9. SAVE / LOAD / EXPORT
# =============================================================================

def save_model(model: nn.Module, path: str = FINAL_MODEL_PATH):
    torch.save({
        "state_dict":  model.state_dict(),
        "num_classes": NUM_CLASSES,
        "img_height":  IMG_HEIGHT,
        "img_width":   IMG_WIDTH,
        "label_map":   LABEL_MAP,
    }, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Save] {path}  ({size_mb:.1f} MB)")


def export_onnx(model: nn.Module, path: str = ONNX_PATH):
    """
    ONNX export.
    IMPORTANT: inference pipeline must normalize input as:
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5
    before passing to the model. This matches training normalization.
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


def export_quantized(model: nn.Module, path: str = QUANTIZED_PATH):
    """Ch. 18 int8 quantization for faster CPU inference."""
    model.eval().cpu()
    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={nn.Linear}, dtype=torch.qint8,
    )
    torch.save(quantized, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Quantize] int8 model saved to {path}  ({size_mb:.1f} MB)")
    return quantized


# =============================================================================
# 10. INFERENCE
# =============================================================================

def predict_image(model, image_path, device, top_k=5):
    from PIL import Image
    transform = get_transforms(augment=False)
    img = Image.open(image_path).convert("L")
    arr = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(arr), dim=1)[0].cpu().numpy()
    top_i = np.argsort(probs)[::-1][:top_k]
    return [(LABEL_MAP[i], float(probs[i])) for i in top_i]


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Pure PyTorch  (v2 — corrected augmentation)")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {BASE_DIR}")
    print(f"  Resolution: {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    print("  Changes: rotation ±5°, shear 3°, WeightedRandomSampler")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    train_ds, val_ds, test_ds, supp_ds = load_emnist(DATA_DIR)

    # FIX v2: use weighted sampler for training to address class imbalance
    train_loader = make_dataloader(train_ds, use_weighted_sampler=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    model = OCRConvNet(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNet")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, train_loader)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=CHECKPOINT_PATH)

    print(f"\n[Train] Starting — max epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device
        )
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

    # FIX v2: run per-class accuracy on test set to identify remaining failures
    print("\n[Eval] Running per-class accuracy analysis on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    plot_history(history)
    save_log(history)
    save_model(model)

    model_cpu = OCRConvNet(NUM_CLASSES)
    model_cpu.load_state_dict(
        torch.load(FINAL_MODEL_PATH, map_location="cpu",
                   weights_only=False)["state_dict"]
    )
    export_onnx(model_cpu)
    export_quantized(model_cpu)

    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()
