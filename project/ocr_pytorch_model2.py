"""
ocr_pytorch_model2.py
=====================
EMNIST OCR — Model 2 for Ensemble
Pure PyTorch — Wider architecture with Squeeze-Excitation attention blocks.

This is Model 2 in a two-model ensemble with ocr_pytorch_model.py.
Architectural differences from Model 1 (intentional diversity for ensemble):
  - Wider filter progression: 32→128→256→512 vs 32→64→128→256
  - Squeeze-Excitation (SE) attention after each stage (channel recalibration)
  - StochasticDepth (DropPath) regularization instead of SpatialDropout
  - Cosine annealing LR schedule instead of OneCycleLR
  - AdamW optimizer instead of Adam
  - Larger classifier head: 512→256 instead of 256→256

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 3  — PyTorch nn.Module, tensors, backward(), optimizer.step()
  Ch. 5  — Dropout, weight decay, data augmentation as regularization
  Ch. 6  — Universal ML workflow
  Ch. 8  — ConvNet architecture, filter progression, GlobalAveragePooling
  Ch. 9  — BatchNormalization, residual connections, depthwise separable convs
  Ch. 18 — Mixed-precision (AMP), model ensembling, int8 quantization

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB

Output: E:\\CSC-114\\emnist-model\\pytorch2\\
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import EMNIST


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 62
IMG_HEIGHT       = 32
IMG_WIDTH        = 32

BATCH_SIZE       = 512
EPOCHS           = 50
LEARNING_RATE    = 3e-4    # AdamW default — lower than Model 1's 1e-3
WEIGHT_DECAY     = 5e-4    # stronger L2 vs Model 1's 1e-4
VALIDATION_SPLIT = 0.15
PATIENCE         = 7
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
# 3. DATA PIPELINE (same as Model 1 — identical data, different model)
# =============================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    aug_transforms = [
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15), shear=8),
        transforms.ColorJitter(contrast=0.3),
    ] if augment else []

    base_transforms = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
    return transforms.Compose(aug_transforms + base_transforms)


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
    return train_ds, val_ds, test_ds


def make_dataloader(dataset, shuffle: bool = False) -> DataLoader:
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
# 4. MODEL ARCHITECTURE — Wider + Squeeze-Excitation attention (Ch. 9)
# =============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation (SE) block — channel attention mechanism.

    Ch. 9 concept: after a residual block learns spatial features, SE
    recalibrates which channels are most informative by:
      1. Squeeze: GlobalAveragePool → (batch, C) — collapses spatial dims
      2. Excitation: FC → ReLU → FC → Sigmoid → (batch, C) scale vector
      3. Scale: multiply each channel by its learned importance weight

    This is architecturally distinct from Model 1 which has no attention.
    Hu et al. 2018 "Squeeze-and-Excitation Networks" showed consistent
    accuracy improvements across all ConvNet architectures at minimal cost.

    reduction: bottleneck ratio for the FC layers (16 = standard SE paper)
    """
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
        scale = self.pool(x).view(b, c)        # squeeze: (B, C)
        scale = self.fc(scale).view(b, c, 1, 1)  # excitation: (B, C, 1, 1)
        return x * scale                        # channel-wise rescaling


class SEResidualBlock(nn.Module):
    """
    Residual block with integrated Squeeze-Excitation attention.

    Architecture: Conv→BN→ReLU→Conv→BN→SE→add_skip→ReLU

    Ch. 9 residual connection gives gradients a direct backward path.
    SE attention sits before the residual add — it recalibrates the
    learned features before they're combined with the identity skip.

    Wider than Model 1: uses larger channel counts throughout.
    """
    def __init__(self, in_ch: int, out_ch: int, drop_path_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch, reduction=16)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if in_ch != out_ch else nn.Identity()
        )

        # StochasticDepth (DropPath): randomly drops entire residual branch
        # during training — stronger regularization than SpatialDropout.
        # Ch. 5: another form of noise injection to prevent memorization.
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic depth: drop entire residual branch with probability p."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        # Random per-sample mask: shape (batch, 1, 1, 1) broadcasts over spatial dims
        mask = torch.rand(x.shape[0], 1, 1, 1,
                          device=x.device, dtype=x.dtype)
        mask = (mask < keep).float() / keep   # scale to preserve expectation
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)               # channel attention
        out = self.drop_path(out)        # stochastic depth regularization
        return F.relu(out + residual, inplace=True)


class OCRConvNetWide(nn.Module):
    """
    Wide OCR ConvNet with Squeeze-Excitation attention.

    Input:  (batch, 1, 32, 32)
    Output: (batch, 62)

    Architecture (Model 2 — wider than Model 1):
        Stem:    DepthwiseSep(1→32)
        Stage 1: SEResidualBlock(32→128)  + MaxPool   [Model1: 32→64]
        Stage 2: SEResidualBlock(128→256) + MaxPool   [Model1: 64→128]
        Stage 3: SEResidualBlock(256→512) + MaxPool   [Model1: 128→256]
        Stage 4: SEResidualBlock(512→512)             [Model1: 256→256]
        Pool:    AdaptiveAvgPool2d(1)
        Head:    Linear(512→512)→BN→ReLU→Drop→Linear(512→256)→BN→ReLU→Drop→Linear(256→62)

    Parameters: ~8.5M vs Model 1's ~2.4M
    Diversity rationale: wider channels capture more feature variety;
    SE attention focuses on informative channels; deeper head adds capacity.
    Ch. 18 ensemble: "use as different models as possible" for maximum benefit.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Depthwise separable stem — Ch. 9 efficient feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False),  # depthwise
            nn.Conv2d(1, 32, 1, bias=False),                       # pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Wider filter progression with SE attention
        self.stage1 = nn.Sequential(
            SEResidualBlock(32,  128, drop_path_rate=0.05),
            nn.MaxPool2d(2),    # 32×32 → 16×16
        )
        self.stage2 = nn.Sequential(
            SEResidualBlock(128, 256, drop_path_rate=0.1),
            nn.MaxPool2d(2),    # 16×16 → 8×8
        )
        self.stage3 = nn.Sequential(
            SEResidualBlock(256, 512, drop_path_rate=0.15),
            nn.MaxPool2d(2),    # 8×8 → 4×4
        )
        self.stage4 = SEResidualBlock(512, 512, drop_path_rate=0.2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Deeper classifier head — more capacity for 62-class discrimination
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
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
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


# =============================================================================
# 5. EARLY STOPPING + CHECKPOINT (same as Model 1)
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
            print(f"  [Checkpoint] val_loss → {val_loss:.4f}  saved to {self.path}")
        else:
            self.counter += 1
            print(f"  [EarlyStopping] {self.counter}/{self.patience} epochs without improvement")
            if self.counter >= self.patience:
                self.stop = True
                print("  [EarlyStopping] Halting training.")


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
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 7. PLOTTING AND LOGGING
# =============================================================================

def plot_history(history: dict):
    ep      = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR Model 2 (Wide + SE) — Training History",
                 fontsize=12, fontweight="bold")
    ax1.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)
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
# 9. ENSEMBLE INFERENCE (Ch. 18)
# =============================================================================

def ensemble_predict_loader(model1: nn.Module,
                             model2: nn.Module,
                             loader: DataLoader,
                             device: torch.device) -> tuple:
    """
    Ch. 18 ensemble: average softmax outputs from both models.
    Both models trained independently on same data but with different
    architectures — their errors are partially uncorrelated, so averaging
    cancels individual mistakes and improves overall accuracy.

    Returns (predictions, labels) as numpy arrays for accuracy calculation.
    """
    model1.eval(); model2.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            p1 = F.softmax(model1(images), dim=1)
            p2 = F.softmax(model2(images), dim=1)
            avg = (p1 + p2) / 2.0
            all_preds.append(avg.argmax(1).cpu())
            all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return preds, labels


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Model 2 (Wide + Squeeze-Excitation)")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {BASE_DIR}")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    # Step 1: Data
    train_ds, val_ds, test_ds = load_emnist(DATA_DIR)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    # Step 2: Model — wider architecture with SE attention
    model = OCRConvNetWide(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetWide (Model 2)")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")
    print(f"  Architecture: wider channels + SE attention + StochasticDepth")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW — weight decay applied correctly (decoupled from adaptive lr)
    # Ch. 5: proper L2 regularization decoupled from gradient scaling
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # Cosine annealing — smooth decay vs Model 1's OneCycleLR warmup/peak/decay
    # Different schedule = different optimization trajectory = more ensemble diversity
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    scaler     = torch.amp.GradScaler('cuda',
                     enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=CHECKPOINT_PATH)

    # Step 3: Train
    print(f"\n[Train] Starting — max epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    history = {k: [] for k in ["train_loss", "train_acc",
                                "val_loss",   "val_acc", "lr"]}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, None, device
        )
        # CosineAnnealingLR steps per epoch (not per batch)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  |  "
              f"lr: {current_lr:.2e}  [{elapsed:.0f}s]")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc),
                     ("lr", current_lr)]:
            history[k].append(v)

        early_stop(val_loss, model)
        if early_stop.stop:
            break

    # Step 4: Reload best weights
    print(f"\n[Train] Loading best checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # Step 5: Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*40}")
    print(f"  Model 2 Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Model 2 Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    # Step 6: Save artifacts
    plot_history(history)
    save_log(history)
    save_model(model)

    # Step 7: ONNX export
    try:
        model_cpu = OCRConvNetWide(NUM_CLASSES)
        model_cpu.load_state_dict(
            torch.load(FINAL_MODEL_PATH, map_location="cpu",
                       weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        print(f"       pip install onnx  then re-run export")

    # Step 8: Ensemble test — load Model 1 and average predictions
    model1_path = Path(r"E:\CSC-114\emnist-model\pytorch\best_model.pt")
    if model1_path.exists():
        print(f"\n[Ensemble] Loading Model 1 from {model1_path}")
        try:
            # Import Model 1 class
            import sys
            sys.path.insert(0, str(Path(r"E:\CSC-114\emnist-model")))
            from ocr_pytorch_model import OCRConvNet
            model1 = OCRConvNet(NUM_CLASSES)
            ckpt1  = torch.load(str(model1_path), map_location="cpu",
                                weights_only=False)
            model1.load_state_dict(ckpt1["state_dict"])
            model1 = model1.to(device)

            preds, labels = ensemble_predict_loader(
                model1, model, test_loader, device
            )
            ensemble_acc = (preds == labels).mean()
            print(f"\n{'='*40}")
            print(f"  Model 1 alone : 88.06%")
            print(f"  Model 2 alone : {test_acc*100:.2f}%")
            print(f"  ENSEMBLE      : {ensemble_acc*100:.2f}%")
            print(f"{'='*40}")
        except Exception as e:
            print(f"[Ensemble] Could not load Model 1: {e}")
    else:
        print(f"\n[Ensemble] Model 1 checkpoint not found at {model1_path}")
        print(f"           Run ocr_pytorch_model.py first.")

    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()
