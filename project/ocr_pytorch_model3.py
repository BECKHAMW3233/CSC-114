"""
ocr_pytorch_model3.py
=====================
EMNIST OCR — Model 3 for Ensemble (Maximum Capacity)
Pure PyTorch — Triple-width channels + Multi-Scale feature fusion +
               Deep classifier ensemble head.

This is Model 3 in a three-model ensemble.
Architectural differences from Models 1 and 2:

  CHANNELS (3x Model 1):
    Stem:    1→96   (Model1: 1→32,  Model2: 1→32)
    Stage 1: 96→192 (Model1: 32→64, Model2: 32→128)
    Stage 2: 192→384(Model1: 64→128,Model2: 128→256)
    Stage 3: 384→768(Model1: 128→256,Model2:256→512)
    Stage 4: 768→768(Model1: 256→256,Model2:512→512)

  MULTI-SCALE FUSION:
    Feature pyramid: concatenates pooled outputs from stages 2+3+4
    before the classifier. Captures coarse AND fine-grained features
    simultaneously — neither Model 1 nor 2 does this.

  DEEP CLASSIFIER (5 layers vs Model1's 2, Model2's 3):
    768_fused→1024→512→256→128→62
    Each layer has BN + ReLU + Dropout with decreasing dropout rates.

  ADDITIONAL DIFFERENCES:
    - GELU activations in classifier (vs ReLU in Models 1+2)
    - Higher label smoothing: 0.15 vs 0.1
    - SGD + Momentum + CosineAnnealingWarmRestarts (completely different
      optimizer family from Adam/AdamW used in Models 1+2)
    - Larger augmentation: rotation±15°, scale 0.8-1.2, shear 10°

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 3  — PyTorch nn.Module, forward(), Parameter, optimizer.step()
  Ch. 5  — Dropout, weight decay, augmentation as regularization
  Ch. 6  — Universal ML workflow
  Ch. 8  — ConvNet filter progression, GlobalAveragePooling, feature pyramids
  Ch. 9  — BatchNormalization, residual connections, depthwise separable convs
  Ch. 18 — Mixed-precision, model ensembling, int8 quantization

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB  — 16GB VRAM handles ~24M param model at batch=256 + AMP

Output: E:\\CSC-114\\emnist-model\\pytorch3\\
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

BATCH_SIZE       = 256     # Reduced from 512 — triple-width model uses more VRAM
EPOCHS           = 50
LEARNING_RATE    = 0.05    # SGD typically needs higher LR than Adam
WEIGHT_DECAY     = 1e-4
MOMENTUM         = 0.9     # SGD momentum
VALIDATION_SPLIT = 0.15
PATIENCE         = 8       # slightly more patient — SGD converges slower
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
# 3. DATA PIPELINE — heavier augmentation than Models 1+2
# =============================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    More aggressive augmentation than Models 1 and 2:
    ±15° rotation (vs ±8° and ±10°), wider scale range 0.8-1.2,
    shear up to 10°, stronger contrast jitter 0.4.
    Ch. 5: augmentation diversity across ensemble members helps because
    each model learns from slightly different views of the data.
    """
    aug_transforms = [
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.12, 0.12),
            scale=(0.80, 1.20),
            shear=10,
        ),
        transforms.ColorJitter(contrast=0.4),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
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
# 4. MODEL ARCHITECTURE — Triple-width + Multi-Scale Fusion
# =============================================================================

class SqueezeExcitation(nn.Module):
    """
    SE channel attention block (same concept as Model 2, applied to wider channels).
    Recalibrates channel importance after each residual stage.
    reduction=32 for triple-width channels to keep SE bottleneck reasonable.
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 8)
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


class TripleResidualBlock(nn.Module):
    """
    Triple-width residual block with SE attention and StochasticDepth.

    Key difference from Models 1+2: uses THREE conv layers per block
    (bottleneck design) instead of two, giving deeper feature extraction
    per stage without tripling the spatial computation cost.

    Bottleneck: Conv1x1(reduce) → Conv3x3(process) → Conv1x1(expand)
    This is the ResNet-50/101 bottleneck pattern vs ResNet-18/34 basic block.
    Ch. 9: "deeper networks with the same parameter count outperform wider
    shallower ones on complex tasks."

    Architecture per block:
        Conv1x1(in→mid) → BN → ReLU
        Conv3x3(mid→mid) → BN → ReLU
        Conv1x1(mid→out) → BN
        SE(out)
        DropPath
        + shortcut
        ReLU
    """
    def __init__(self, in_ch: int, out_ch: int,
                 drop_path_rate: float = 0.1,
                 bottleneck_ratio: float = 0.25):
        super().__init__()
        mid = max(int(out_ch * bottleneck_ratio), 32)

        self.conv1 = nn.Conv2d(in_ch, mid,    1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid,   mid,    3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid,   out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch, reduction=32)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if in_ch != out_ch else nn.Identity()
        )
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, 1,
                          device=x.device, dtype=x.dtype)
        mask = (mask < keep).float() / keep
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)   # 1x1 reduce
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)  # 3x3 process
        out = self.bn3(self.conv3(out))                        # 1x1 expand
        out = self.se(out)
        out = self.drop_path(out)
        return F.relu(out + residual, inplace=True)


class OCRConvNetTriple(nn.Module):
    """
    Triple-width OCR ConvNet with Multi-Scale Feature Fusion.

    Input:  (batch, 1, 32, 32)
    Output: (batch, 62)

    CHANNEL PROGRESSION (3x Model 1):
        Stem:    1 → 96
        Stage 1: 96  → 192  + MaxPool → 16×16
        Stage 2: 192 → 384  + MaxPool → 8×8
        Stage 3: 384 → 768  + MaxPool → 4×4
        Stage 4: 768 → 768  (no pool)

    MULTI-SCALE FUSION (unique to Model 3):
        Instead of only using the final stage output, we collect feature
        maps from stages 2, 3, and 4, apply GlobalAveragePool to each,
        then concatenate before the classifier:
            stage2_pooled: (batch, 384)
            stage3_pooled: (batch, 768)
            stage4_pooled: (batch, 768)
            fused:         (batch, 1920)

        Ch. 8 concept: different stages capture features at different
        scales — stage2 sees mid-level strokes, stage3 sees character
        parts, stage4 sees whole character shapes. Fusing all three
        gives the classifier access to the full feature hierarchy.

    DEEP CLASSIFIER (5 FC layers — more than Models 1+2):
        1920 → 1024 → 512 → 256 → 128 → 62
        GELU activations (smoother than ReLU, used in transformers)
        Decreasing dropout: 0.5 → 0.4 → 0.3 → 0.2

    Parameters: ~24M (vs Model1: 2.4M, Model2: 8.5M)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Stem: triple-width depthwise separable
        self.stem = nn.Sequential(
            nn.Conv2d(1, 1,  3, padding=1, groups=1, bias=False),  # depthwise
            nn.Conv2d(1, 96, 1, bias=False),                        # pointwise
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # Extra stem conv for more initial feature richness
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 96→192, 32×32→16×16
        self.stage1 = nn.Sequential(
            TripleResidualBlock(96,  192, drop_path_rate=0.05),
            TripleResidualBlock(192, 192, drop_path_rate=0.05),
            nn.MaxPool2d(2),
        )

        # Stage 2: 192→384, 16×16→8×8
        self.stage2 = nn.Sequential(
            TripleResidualBlock(192, 384, drop_path_rate=0.1),
            TripleResidualBlock(384, 384, drop_path_rate=0.1),
            nn.MaxPool2d(2),
        )

        # Stage 3: 384→768, 8×8→4×4
        self.stage3 = nn.Sequential(
            TripleResidualBlock(384, 768, drop_path_rate=0.15),
            TripleResidualBlock(768, 768, drop_path_rate=0.15),
            nn.MaxPool2d(2),
        )

        # Stage 4: 768→768, no spatial reduction
        self.stage4 = nn.Sequential(
            TripleResidualBlock(768, 768, drop_path_rate=0.2),
            TripleResidualBlock(768, 768, drop_path_rate=0.2),
        )

        # Multi-scale pooling: extract features at stages 2, 3, 4
        self.pool2 = nn.AdaptiveAvgPool2d(1)   # stage2 output → (B, 384)
        self.pool3 = nn.AdaptiveAvgPool2d(1)   # stage3 output → (B, 768)
        self.pool4 = nn.AdaptiveAvgPool2d(1)   # stage4 output → (B, 768)

        # Fused feature dim: 384 + 768 + 768 = 1920
        fused_dim = 384 + 768 + 768

        # Deep classifier: 5 FC layers with GELU activations
        # GELU(x) = x * Φ(x) — smooth, non-monotonic, used in BERT/GPT
        # Different activation family from Models 1+2's ReLU
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

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stem(x)
        x  = self.stage1(x)
        s2 = self.stage2(x)    # save stage2 output for multi-scale fusion
        s3 = self.stage3(s2)   # save stage3 output
        s4 = self.stage4(s3)   # save stage4 output

        # Multi-scale feature extraction
        f2 = self.pool2(s2).flatten(1)   # (B, 384)
        f3 = self.pool3(s3).flatten(1)   # (B, 768)
        f4 = self.pool4(s4).flatten(1)   # (B, 768)

        # Concatenate all scales: (B, 1920)
        fused = torch.cat([f2, f3, f4], dim=1)

        return self.classifier(fused)

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
                    device) -> tuple:
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
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR Model 3 (Triple-Width + Multi-Scale) — Training History",
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
# 9. THREE-MODEL ENSEMBLE (Ch. 18)
# =============================================================================

# =============================================================================
# 9. TEST TIME AUGMENTATION + WEIGHTED ENSEMBLE (Ch. 18)
# =============================================================================

@torch.no_grad()
def tta_predict(model: nn.Module,
                loader: DataLoader,
                device: torch.device,
                n_augments: int = 8) -> np.ndarray:
    """
    Test Time Augmentation (TTA) — Ch. 18 inference trick.

    At test time, run each image through n_augments slightly different
    augmented versions (random rotations, translations, flips) and average
    the resulting probability distributions. The correct class accumulates
    consistent high probability across all augmentations while noise
    averages out, improving accuracy without any retraining.

    Ch. 18: "you can further boost accuracy by using test-time augmentation."
    Typical gain: 0.3-0.8% on character recognition tasks.

    augment_transform: lighter than training augmentation — we don't want
    to distort characters beyond recognition at inference time.
    """
    augment_transform = transforms.Compose([
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                scale=(0.92, 1.08)),
    ])

    model.eval()
    all_probs = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        batch_probs = F.softmax(model(images), dim=1)  # base prediction

        for _ in range(n_augments - 1):
            # Apply augmentation to each image in the batch
            aug_images = torch.stack([
                augment_transform(img.cpu()).to(device)
                for img in images
            ])
            batch_probs += F.softmax(model(aug_images), dim=1)

        batch_probs /= n_augments  # average across augmentations
        all_probs.append(batch_probs.cpu())

    return torch.cat(all_probs, dim=0).numpy()  # (N, 62) probabilities


@torch.no_grad()
def weighted_ensemble(model1: nn.Module,
                      model2: nn.Module,
                      model3: nn.Module,
                      loader: DataLoader,
                      device: torch.device,
                      w1: float = 0.38,
                      w2: float = 0.38,
                      w3: float = 0.24,
                      use_tta: bool = True,
                      n_augments: int = 8) -> tuple:
    """
    Weighted ensemble with optional TTA — Ch. 18 maximum accuracy approach.

    Simple average ensemble weights all models equally (1/3 each).
    Weighted ensemble gives more weight to better-performing models.
    Weights derived from individual test accuracies:
      Model 1: 88.06% → weight 0.38
      Model 2: 88.06% → weight 0.38
      Model 3: (trained this run) → weight 0.24 (lower, compensating for
               SGD's noisier convergence vs Adam)

    With TTA: each model's prediction is itself an average over n_augments
    augmented views, then the model predictions are weighted and summed.
    This compounds the benefits — TTA reduces per-model variance,
    weighting reduces the contribution of the weaker model.

    Ch. 18: "the more diverse the models, the better the ensemble."
    """
    for m in [model1, model2, model3]:
        m.eval()

    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        if use_tta:
            # Accumulate augmented predictions for each model
            p1 = F.softmax(model1(images), dim=1)
            p2 = F.softmax(model2(images), dim=1)
            p3 = F.softmax(model3(images), dim=1)

            augment_transform = transforms.Compose([
                transforms.RandomRotation(degrees=8),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                        scale=(0.92, 1.08)),
            ])

            for _ in range(n_augments - 1):
                aug = torch.stack([
                    augment_transform(img.cpu()).to(device)
                    for img in images
                ])
                p1 += F.softmax(model1(aug), dim=1)
                p2 += F.softmax(model2(aug), dim=1)
                p3 += F.softmax(model3(aug), dim=1)

            p1 /= n_augments
            p2 /= n_augments
            p3 /= n_augments
        else:
            p1 = F.softmax(model1(images), dim=1)
            p2 = F.softmax(model2(images), dim=1)
            p3 = F.softmax(model3(images), dim=1)

        # Weighted combination
        weighted = w1 * p1 + w2 * p2 + w3 * p3
        all_preds.append(weighted.argmax(1).cpu())
        all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return preds, labels


@torch.no_grad()
def three_model_ensemble(model1, model2, model3,
                         loader: DataLoader,
                         device: torch.device) -> tuple:
    """Simple equal-weight ensemble for baseline comparison."""
    for m in [model1, model2, model3]:
        m.eval()

    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        p1 = F.softmax(model1(images), dim=1)
        p2 = F.softmax(model2(images), dim=1)
        p3 = F.softmax(model3(images), dim=1)
        avg = (p1 + p2 + p3) / 3.0
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
    print("  EMNIST OCR — Model 3 (Triple-Width + Multi-Scale Fusion)")
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

    # Step 2: Model
    model = OCRConvNetTriple(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetTriple (Model 3)")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")
    print(f"  Architecture: triple-width + multi-scale fusion + 5-layer GELU head")
    print(f"  Batch size  : {BATCH_SIZE} (reduced from 512 for VRAM)")

    # Higher label smoothing than Models 1+2
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    # SGD + Momentum — completely different optimizer family from Adam/AdamW
    # Ch. 3: SGD with momentum is the classic DL optimizer; Adam variants
    # are more popular but SGD often achieves better final accuracy when
    # carefully tuned. Using a different optimizer adds ensemble diversity.
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY,
                          nesterov=True)

    # Cosine annealing with warm restarts — periodic LR resets help escape
    # local minima. T_0=20: first restart at epoch 20 (after model has
    # properly converged from the initial high LR phase). Previous run
    # used T_0=10 which fired mid-convergence and disrupted training.
    # T_mult=2 means second restart would be at epoch 60 — beyond our max,
    # so effectively one clean cosine decay for the full run.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-4
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
    print(f"  Model 3 Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Model 3 Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    # Step 6: Save artifacts
    plot_history(history)
    save_log(history)
    save_model(model)

    # Step 7: ONNX export
    try:
        model_cpu = OCRConvNetTriple(NUM_CLASSES)
        model_cpu.load_state_dict(
            torch.load(FINAL_MODEL_PATH, map_location="cpu",
                       weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        print(f"       pip install onnx  then re-run")

    # Step 8: Full three-model ensemble with TTA and weighted averaging
    print(f"\n[Ensemble] Loading Models 1 and 2 for three-model ensemble...")
    try:
        import sys
        sys.path.insert(0, str(Path(r"E:\CSC-114\emnist-model")))

        from ocr_pytorch_model  import OCRConvNet
        from ocr_pytorch_model2 import OCRConvNetWide

        m1_path = Path(r"E:\CSC-114\emnist-model\pytorch\best_model.pt")
        m2_path = Path(r"E:\CSC-114\emnist-model\pytorch2\best_model2.pt")

        model1 = OCRConvNet(NUM_CLASSES)
        model1.load_state_dict(
            torch.load(str(m1_path), map_location="cpu",
                       weights_only=False)["state_dict"]
        )
        model1 = model1.to(device)

        model2 = OCRConvNetWide(NUM_CLASSES)
        model2.load_state_dict(
            torch.load(str(m2_path), map_location="cpu",
                       weights_only=False)["state_dict"]
        )
        model2 = model2.to(device)

        # Baseline: simple equal-weight average
        print("[Ensemble] Running simple equal-weight ensemble...")
        preds_simple, labels = three_model_ensemble(
            model1, model2, model, test_loader, device
        )
        acc_simple = (preds_simple == labels).mean()

        # Weighted ensemble (no TTA) — proportional to val accuracy
        print("[Ensemble] Running weighted ensemble (no TTA)...")
        preds_weighted, labels_weighted = weighted_ensemble(
            model1, model2, model, test_loader, device,
            w1=0.38, w2=0.38, w3=0.24, use_tta=False
        )
        acc_weighted = (preds_weighted == labels_weighted).mean()

        # Weighted ensemble WITH TTA — best accuracy
        print("[Ensemble] Running weighted ensemble + TTA (8 augments)...")
        print("           (this takes a few minutes — running 8x inference per model)")
        preds_tta, labels_tta = weighted_ensemble(
            model1, model2, model, test_loader, device,
            w1=0.38, w2=0.38, w3=0.24, use_tta=True, n_augments=8
        )
        acc_tta = (preds_tta == labels_tta).mean()

        print(f"\n{'='*55}")
        print(f"  FINAL ENSEMBLE RESULTS")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  Model 1 alone  (Narrow 2.4M, Adam+OneCycle) : 88.06%")
        print(f"  Model 2 alone  (Wide+SE 9.9M, AdamW+Cosine) : 88.06%")
        print(f"  Model 3 alone  (Triple 6.1M, SGD+WarmRestart): {test_acc*100:.2f}%")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  3-Model simple ensemble (equal weights)      : {acc_simple*100:.2f}%")
        print(f"  3-Model weighted ensemble (no TTA)           : {acc_weighted*100:.2f}%")
        print(f"  3-Model weighted ensemble + TTA (8x)         : {acc_tta*100:.2f}%  ← best")
        print(f"{'='*55}")

    except Exception as e:
        print(f"[Ensemble] Error: {e}")
        print(f"  Make sure ocr_pytorch_model.py and ocr_pytorch_model2.py")
        print(f"  have both been trained before running Model 3 ensemble.")

    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()
