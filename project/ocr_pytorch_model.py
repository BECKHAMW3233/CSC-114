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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 62
IMG_HEIGHT       = 32
IMG_WIDTH        = 32

BATCH_SIZE       = 512     # RTX 4080 16 GB + AMP float16
EPOCHS           = 50      # EarlyStopping handles actual stopping point
LEARNING_RATE    = 1e-3    # Ch. 3 Adam default
WEIGHT_DECAY     = 1e-4    # Ch. 5 L2 regularization via optimizer
VALIDATION_SPLIT = 0.15
PATIENCE         = 7       # EarlyStopping patience epochs
NUM_WORKERS      = 8       # 7900X 24 threads — 8 workers saturates GPU pipeline

USE_AMP          = True    # Ch. 18 mixed-precision — float16 compute, float32 weights

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
    """
    Ch. 3: tensors must be explicitly moved to a device in PyTorch.
    Unlike Keras which handles device placement automatically, PyTorch requires
    model.to(device) and tensor.to(device) calls throughout the code.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1024**3
        print(f"[Device] {props.name}  |  {vram:.1f} GB VRAM  |  "
              f"CUDA {torch.version.cuda}  |  AMP: {USE_AMP}")
        # cuDNN autotuner: benchmarks conv algorithms for fixed input sizes
        # and caches the fastest one — equivalent effect to XLA compilation in TF
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU — {torch.get_num_threads()} threads available")
        print("         No GPU found. Run 03_verify_gpu.py to diagnose.")
    return device


# =============================================================================
# 3. DATA TRANSFORMS AND PIPELINE
# =============================================================================

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Ch. 8 augmentation strategy translated to torchvision transforms.
    Ch. 5: augmentation is a form of regularization — it synthetically
    expands the training set by creating plausible variants of each image,
    which prevents the model from memorizing exact training samples.

    augment=True  → training pipeline (augmentation active)
    augment=False → val/test pipeline (clean images only)
    """
    aug_transforms = [
        # Ch. 8 equivalents:
        transforms.RandomRotation(degrees=8),          # RandomRotation(0.08)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                      # RandomTranslation
            scale=(0.9, 1.1),                          # RandomZoom
            shear=5,                                    # slight shear for cursive
        ),
        transforms.ColorJitter(contrast=0.2),          # ink density variation
    ] if augment else []

    base_transforms = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),                         # uint8 [0,255] → float32 [0,1]
        # Normalize to [-1, 1] — improves gradient flow vs raw [0,1]
        # mean=0.5, std=0.5 centers the distribution around 0
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]

    return transforms.Compose(aug_transforms + base_transforms)


def load_emnist(data_dir: Path):
    """
    Ch. 6 workflow step: prepare data.
    Downloads EMNIST byclass via torchvision (~540 MB first run).
    Splits training set into train + val using fixed seed for reproducibility.

    EMNIST byclass: 697,932 train + 116,323 test, 62 classes.
    Reference: Cohen et al. 2017.

    torchvision note: EMNIST images are stored transposed in the raw binary
    format. torchvision automatically corrects this for byclass split.
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

    # Val subset uses non-augmented transforms
    train_ds = Subset(train_full, train_indices.indices)
    val_base = EMNIST(root=str(data_dir), split="byclass", train=True,
                      download=False, transform=get_transforms(augment=False))
    val_ds   = Subset(val_base, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def make_dataloader(dataset, shuffle: bool = False) -> DataLoader:
    """
    PyTorch DataLoader — equivalent to tf.data pipeline with cache/prefetch.
    Ch. 18 performance tip: pin_memory=True enables faster host→GPU transfers
    by keeping data in pinned (page-locked) CPU RAM.
    persistent_workers avoids Python process fork overhead between epochs on Windows.
    """
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
# 4. MODEL ARCHITECTURE (Ch. 3 nn.Module + Ch. 8 ConvNet + Ch. 9 residual/BN)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Ch. 9 depthwise separable convolution.
    Standard Conv2D applies a filter jointly across all input channels.
    Depthwise separable splits this into:
      1. Depthwise conv: one filter per input channel (spatial features)
      2. Pointwise conv: 1x1 conv to mix channels (cross-channel features)
    Result: same representational power at ~8-9x fewer parameters.
    Used in Xception architecture (Ch. 8) and MobileNet.
    groups=in_channels in Conv2d implements the depthwise step.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn         = nn.BatchNorm2d(out_ch)   # Ch. 9 BatchNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pointwise(self.depthwise(x))), inplace=True)


class ResidualBlock(nn.Module):
    """
    Ch. 9 residual block with BatchNormalization.

    Architecture: Conv → BN → ReLU → Conv → BN → add_skip → ReLU

    Ch. 9 explains residual connections solve vanishing gradients:
    "the skip connection gives gradients a direct path backward through the
    entire network depth, bypassing the Conv layers entirely if needed."

    Ch. 9 BatchNorm placement: BN after Conv, before activation.
    bias=False when followed by BN — BN has its own learnable bias (beta).

    1x1 projection shortcut when in_channels != out_channels, so the skip
    connection tensor shapes match for the Add operation.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Ch. 9: projection shortcut when dimensions differ
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ch. 3 Listing 3.27: implement forward() computation
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))          # BN before adding skip
        return F.relu(x + residual, inplace=True)


class OCRConvNet(nn.Module):
    """
    OCR ConvNet combining Ch. 8 architecture with Ch. 9 improvements.

    Input:  (batch, 1, 32, 32)   grayscale character image
    Output: (batch, 62)          logits for 62 character classes

    Architecture:
        Stem:    DepthwiseSeparableConv(1→32)   — Ch. 9 efficient feature extraction
        Stage 1: ResidualBlock(32→64)  + MaxPool + SpatialDropout  — Ch. 8+9
        Stage 2: ResidualBlock(64→128) + MaxPool + SpatialDropout  — Ch. 8+9
        Stage 3: ResidualBlock(128→256)+ MaxPool                   — Ch. 8+9
        Stage 4: ResidualBlock(256→256)                            — deeper = better
        Pool:    AdaptiveAvgPool2d(1)   — Ch. 8 GlobalAveragePooling equivalent
        Head:    Linear(256→256) → BN → ReLU → Dropout → Linear(256→62)

    Parameters: ~2.4M   |   Saved size: ~9 MB
    Better than Keras version: extra ResidualBlock at stage 4 adds depth without
    requiring a larger input resolution, improving character discrimination.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Ch. 9: depthwise separable stem — efficient low-level feature extraction
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(1, 32),
        )

        # Ch. 8 filter progression 32→64→128→256 with Ch. 9 residual blocks
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),           # 32×32 → 16×16
            nn.Dropout2d(0.1),         # Ch. 5: SpatialDropout drops entire channels
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),           # 16×16 → 8×8
            nn.Dropout2d(0.1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),           # 8×8 → 4×4
        )
        # Ch. 9: additional depth without downsampling — learns more abstract features
        self.stage4 = ResidualBlock(256, 256)

        # Ch. 8: GlobalAveragePooling — averages spatial dims to (batch, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head — Ch. 8 Listing 8.26 pattern
        # Ch. 5: Dropout(0.5) is standard for the penultimate Dense layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),       # Ch. 9: BN stabilizes deep FC layers too
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),  # raw logits — CrossEntropyLoss applies softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ch. 3 Listing 3.27: forward() defines the computation.
        Called via model(x) → __call__() → forward().
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)    # (batch, 256, 1, 1)
        x = x.flatten(1)           # (batch, 256)
        x = self.classifier(x)     # (batch, 62)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities for inference — not used during training."""
        return F.softmax(self.forward(x), dim=1)


# =============================================================================
# 5. EARLY STOPPING + CHECKPOINT
# =============================================================================

class EarlyStopping:
    """
    Manual early stopping — equivalent to keras.callbacks.EarlyStopping
    from Ch. 7 Listing 7.19. PyTorch has no built-in callback system,
    so we implement the same logic explicitly.

    Monitors val_loss. If no improvement for `patience` epochs, sets
    self.stop = True. Saves best weights to disk on every improvement
    (equivalent to ModelCheckpoint(save_best_only=True)).
    """
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
            # Save state_dict only — smaller file, portable across machines
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
# 6. TRAINING LOOP (Ch. 3 explicit training step)
# =============================================================================

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scaler:    torch.cuda.amp.GradScaler,
                    scheduler,
                    device:    torch.device) -> tuple:
    """
    Ch. 3 training step (Listing 3.27 expanded to full epoch loop):
      1. Forward pass      — logits = model(images)
      2. Compute loss      — loss = criterion(logits, labels)
      3. Backward pass     — loss.backward()  [populates .grad on Parameters]
      4. Clip gradients    — prevents exploding gradients in deeper networks
      5. Update weights    — optimizer.step()
      6. Reset gradients   — optimizer.zero_grad()  [MUST happen before next forward]

    Ch. 18 mixed precision via torch.autocast + GradScaler:
      - autocast: runs eligible ops in float16 (conv, matmul), keeps others in float32
      - GradScaler: multiplies loss by a scale factor before backward() to prevent
        float16 gradient underflow, then unscales before optimizer.step()
      This is the PyTorch equivalent of keras.optimizers.LossScaleOptimizer (Ch. 18).
    """
    model.train()   # activates Dropout and BatchNorm training-mode behavior
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # Non-blocking transfers use pinned memory for faster host→GPU copies
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ch. 3: zero_grad() before forward — prevents gradient accumulation
        optimizer.zero_grad()

        # Ch. 18: autocast runs forward pass in float16 where safe
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=USE_AMP and device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        # Ch. 3: loss.backward() computes gradients via backpropagation
        # GradScaler scales loss upward to prevent float16 gradient underflow
        scaler.scale(loss).backward()

        # Gradient clipping — Ch. 9 best practice for deep residual networks
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Ch. 3: optimizer.step() applies gradients to update weights
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR steps per batch (not per epoch)
        if scheduler is not None:
            scheduler.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> tuple:
    """
    Ch. 3: torch.no_grad() skips building the computation graph entirely —
    no backward pass needed at evaluation time, saving memory and compute.
    model.eval() disables Dropout and puts BatchNorm into inference mode
    (uses running mean/variance instead of batch statistics).
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

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
# 7. LEARNING RATE SCHEDULING
# =============================================================================

def build_scheduler(optimizer: optim.Optimizer,
                    train_loader: DataLoader) -> optim.lr_scheduler.OneCycleLR:
    """
    OneCycleLR — cosine annealing with linear warmup.
    Outperforms Ch. 7's ReduceLROnPlateau for ConvNets because it is
    proactive rather than reactive: it doesn't wait for the loss to stall.

    Phase 1 (first 30%): linear warmup from lr/10 to max_lr
    Phase 2 (last 70%): cosine decay from max_lr down to lr/1000

    Warm-up phase prevents large initial updates from destabilizing BatchNorm
    statistics early in training — especially important with large batch sizes.
    Called per-batch (inside train_one_epoch), not per-epoch.
    """
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
# 8. HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Ch. 18 equivalent)
# =============================================================================

def run_hyperparameter_search(train_ds, val_ds,
                              device: torch.device,
                              n_trials: int = 20):
    """
    Ch. 18 hyperparameter optimization — implemented via Optuna, the PyTorch
    ecosystem equivalent of KerasTuner BayesianOptimization.
    Both use Bayesian optimization (TPE sampler in Optuna) to intelligently
    explore the hyperparameter space rather than random search.

    Searches over: filter counts, dense units, dropout rates, learning rate,
    weight decay — the same search space as the Keras KerasTuner version.

    Install: pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[HPO] optuna not installed. pip install optuna")
        return None

    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds)

    def objective(trial):
        # Ch. 18: replace fixed values with trial.suggest_* ranges
        filters1 = trial.suggest_categorical("filters1", [32, 64, 96])
        filters2 = trial.suggest_categorical("filters2", [64, 128, 192])
        filters3 = trial.suggest_categorical("filters3", [128, 256, 384])
        dropout  = trial.suggest_float("dropout", 0.3, 0.6, step=0.1)
        lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # Build a trial-specific model variant
        class TrialModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem   = nn.Sequential(DepthwiseSeparableConv(1, 32))
                self.stage1 = nn.Sequential(ResidualBlock(32, filters1), nn.MaxPool2d(2))
                self.stage2 = nn.Sequential(ResidualBlock(filters1, filters2), nn.MaxPool2d(2))
                self.stage3 = nn.Sequential(ResidualBlock(filters2, filters3), nn.MaxPool2d(2))
                self.pool   = nn.AdaptiveAvgPool2d(1)
                self.head   = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(filters3, 256), nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 0.6),
                    nn.Linear(256, NUM_CLASSES),
                )
            def forward(self, x):
                x = self.stem(x); x = self.stage1(x)
                x = self.stage2(x); x = self.stage3(x)
                x = self.pool(x).flatten(1)
                return self.head(x)

        m         = TrialModel().to(device)
        opt       = optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")
        best_val  = float("inf")
        patience  = 0

        for epoch in range(15):    # short runs per trial to save time
            train_one_epoch(m, train_loader, criterion, opt, scaler, None, device)
            val_loss, _ = evaluate(m, val_loader, criterion, device)
            if val_loss < best_val:
                best_val = val_loss; patience = 0
            else:
                patience += 1
                if patience >= 3: break   # aggressive early stop during search

        return best_val

    print(f"\n[HPO] Running Optuna BayesianOptimization — {n_trials} trials")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[HPO] Best hyperparameters found:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return best


# =============================================================================
# 9. MODEL ENSEMBLING (Ch. 18)
# =============================================================================

def ensemble_predict(models: list,
                     loader: DataLoader,
                     device: torch.device) -> np.ndarray:
    """
    Ch. 18 model ensembling — averages softmax predictions from multiple
    independently trained models. Works because each model makes different
    errors due to different random initializations and training dynamics;
    averaging cancels individual biases.

    Ch. 18: "diversity is strength — ensemble models that are as different
    as possible while being as good as possible."

    Best ensembles for this OCR task: combine models trained from different
    random seeds, OR combine the PyTorch OCRConvNet with the Keras Xception
    model (entirely different architectures = maximum diversity).
    """
    all_preds = []
    for i, model in enumerate(models):
        model.eval()
        batch_preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                probs  = F.softmax(model(images), dim=1)
                batch_preds.append(probs.cpu().numpy())
        all_preds.append(np.concatenate(batch_preds, axis=0))
        print(f"[Ensemble] Model {i+1}/{len(models)} predictions collected.")

    # Ch. 18: simple equal-weight average
    return np.mean(all_preds, axis=0)


def ensemble_accuracy(models: list,
                      loader: DataLoader,
                      device: torch.device) -> float:
    """Evaluates ensemble accuracy on a labeled dataset."""
    preds  = ensemble_predict(models, loader, device)
    labels = np.concatenate([y.numpy() for _, y in loader])
    acc    = (preds.argmax(1) == labels).mean()
    print(f"[Ensemble] Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    return acc


# =============================================================================
# 10. LOGGING AND PLOTTING
# =============================================================================

def plot_history(history: dict, path: str = PLOT_PATH):
    """Training curves — mirrors Keras plot_history() output."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR (PyTorch) — Training History",
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
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Saved to {path}")


def save_log(history: dict, path: str = LOG_PATH):
    """Per-epoch CSV log — equivalent to Keras CSVLogger callback (Ch. 7)."""
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
# 11. SAVE / LOAD / EXPORT
# =============================================================================

def save_model(model: nn.Module, path: str = FINAL_MODEL_PATH):
    """
    torch.save() — Ch. 3 pattern. Saves state_dict (weights only) plus
    metadata. Requires this file's architecture definition to reload.
    """
    torch.save({
        "state_dict":  model.state_dict(),
        "num_classes": NUM_CLASSES,
        "img_height":  IMG_HEIGHT,
        "img_width":   IMG_WIDTH,
        "label_map":   LABEL_MAP,
    }, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Save] {path}  ({size_mb:.1f} MB)")


def load_saved_model(path: str = FINAL_MODEL_PATH,
                     device: torch.device = None) -> nn.Module:
    """Reload weights from checkpoint. Works on CPU-only machines."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt  = torch.load(path, map_location=device)
    model = OCRConvNet(num_classes=ckpt.get("num_classes", NUM_CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[Load] Model loaded from {path}")
    return model


def export_onnx(model: nn.Module, path: str = ONNX_PATH):
    """
    ONNX export — framework-agnostic portable format.
    On the school computer: pip install onnxruntime (~10 MB, no CUDA needed)
    then load with onnxruntime.InferenceSession for CPU inference.
    Dynamic axes allow any batch size at inference time.
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
    """
    Ch. 18 int8 quantization for faster CPU inference.
    Post-training dynamic quantization: converts Linear layer weights from
    float32 to int8. No calibration data needed — scaling computed per-batch.
    Result: ~2-3x faster CPU inference, ~4x smaller weight storage.
    Ideal for running on the school computer without a GPU.
    """
    model.eval().cpu()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},   # quantize Linear layers only
        dtype=torch.qint8,
    )
    torch.save(quantized, path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Quantize] int8 model saved to {path}  ({size_mb:.1f} MB)")
    print(f"           Load with: model = torch.load('{path}')")
    return quantized


# =============================================================================
# 12. INFERENCE
# =============================================================================

def predict_image(model:      nn.Module,
                  image_path: str,
                  device:     torch.device,
                  top_k:      int = 5) -> list:
    """
    Single character image → top-k (character, confidence) predictions.
    Accepts any image format, any size. Handles both GPU and CPU inference.
    Uses the non-augmented transform pipeline for clean inference.
    """
    from PIL import Image
    transform = get_transforms(augment=False)
    img = Image.open(image_path).convert("L")
    arr = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(arr), dim=1)[0].cpu().numpy()
    top_i = np.argsort(probs)[::-1][:top_k]
    return [(LABEL_MAP[i], float(probs[i])) for i in top_i]


def predict_string(model:       nn.Module,
                   image_paths: list,
                   device:      torch.device) -> str:
    """Predict a sequence of character crop images and return as a string."""
    return "".join(predict_image(model, p, device, top_k=1)[0][0]
                   for p in image_paths)


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Pure PyTorch")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {BASE_DIR}")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    # ── Ch. 6 Universal ML Workflow ──────────────────────────────────────────

    # Step 1: Prepare data
    train_ds, val_ds, test_ds = load_emnist(DATA_DIR)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    # Step 2: Build model
    model = OCRConvNet(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNet")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    # Ch. 18: CrossEntropyLoss with label_smoothing
    # Label smoothing prevents overconfident predictions by targeting
    # 0.9 instead of 1.0 — a Ch. 18 best practice for classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Ch. 5: weight_decay adds L2 penalty to all weights via optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    scheduler = build_scheduler(optimizer, train_loader)

    # Ch. 18 GradScaler — PyTorch's LossScaleOptimizer equivalent
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP and device.type == "cuda"
    )

    early_stop = EarlyStopping(patience=PATIENCE, path=CHECKPOINT_PATH)

    # Step 3: Train
    print(f"\n[Train] Starting — max epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    history = {k: [] for k in ["train_loss", "train_acc",
                                "val_loss",   "val_acc", "lr"]}

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
                     ("val_loss", val_loss),   ("val_acc", val_acc),
                     ("lr", current_lr)]:
            history[k].append(v)

        early_stop(val_loss, model)
        if early_stop.stop:
            break

    # Step 4: Reload best weights
    print(f"\n[Train] Loading best checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Step 5: Evaluate on test set — Ch. 6: only touch test set once, at end
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*40}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    # Step 6: Save artifacts
    plot_history(history)
    save_log(history)
    save_model(model)

    # Step 7: Export for deployment
    model_cpu = OCRConvNet(NUM_CLASSES)
    model_cpu.load_state_dict(
        torch.load(FINAL_MODEL_PATH, map_location="cpu")["state_dict"]
    )
    export_onnx(model_cpu)             # for school computer (onnxruntime)
    export_quantized(model_cpu)        # Ch. 18 int8 for faster CPU inference

    # Step 8: Inference demo
    demo = BASE_DIR / "sample_char.png"
    if demo.exists():
        print(f"\n[Inference] Predicting {demo}")
        results = predict_image(model, str(demo), device, top_k=5)
        print("  Top 5 predictions:")
        for char, conf in results:
            bar = "\u2588" * int(conf * 40)
            print(f"    '{char}'  {conf:.4f}  {bar}")
    else:
        print(f"\n[Inference] Drop a character image at {demo} to test.")

    print(f"\n[Done] All files saved to {BASE_DIR}")
    print(f"\n[School demo] Copy these files to your USB/GitHub:")
    print(f"  {BASE_DIR / 'ocr_model_quantized.pt'}  (fastest CPU inference)")
    print(f"  {BASE_DIR / 'ocr_model.onnx'}           (no PyTorch needed, just onnxruntime)")


if __name__ == "__main__":
    main()
