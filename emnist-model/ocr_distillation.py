"""
ocr_distillation.py
===================
Knowledge Distillation for EMNIST OCR 3-Model Ensemble.

Each model is retrained using a combined loss:
  - Cross-entropy against ground truth labels
  - KL-divergence against soft labels from the other two models

Output directories (original models untouched):
  pytorch_distill1/  — Model 1 distilled from Models 2+3
  pytorch_distill2/  — Model 2 distilled from Models 1+3
  pytorch_distill3/  — Model 3 distilled from Models 1+2

Phase 1: Generate soft labels from .pt checkpoints with temperature scaling
Phase 2: Distillation training with combined loss
Phase 3: ONNX validation to verify improvement on real input

Usage:
    python ocr_distillation.py --phase 1          # generate soft labels
    python ocr_distillation.py --phase 2 --model 1  # distill Model 1
    python ocr_distillation.py --phase 2 --model 2  # distill Model 2
    python ocr_distillation.py --phase 2 --model 3  # distill Model 3
    python ocr_distillation.py --phase 3          # validate with ONNX
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import EMNIST

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_CLASSES      = 62
IMG_SIZE         = 64
BATCH_SIZE       = 256
EPOCHS           = 50
PATIENCE         = 15
NUM_WORKERS      = 8
USE_AMP          = True
TEMPERATURE      = 4.0    # soft label temperature — higher = softer distributions
ALPHA            = 0.7    # weight for soft label loss (1-ALPHA for hard label loss)

BASE_DIR         = Path(r"E:\CSC-114\emnist-model")
DATA_DIR         = BASE_DIR / "datasets" / "pytorch"

# Source model checkpoints
M1_PT   = BASE_DIR / "pytorch"  / "best_model.pt"
M2_PT   = BASE_DIR / "pytorch2" / "best_model2.pt"
M3_PT   = BASE_DIR / "pytorch3" / "best_model3.pt"

# Source ONNX models for validation
M1_ONNX = BASE_DIR / "pytorch"  / "ocr_model.onnx"
M2_ONNX = BASE_DIR / "pytorch2" / "ocr_model2.onnx"
M3_ONNX = BASE_DIR / "pytorch3" / "ocr_model3.onnx"

# Soft label storage
SOFT_DIR = BASE_DIR / "soft_labels"

# Distilled model output directories
DISTILL_DIRS = {
    1: BASE_DIR / "pytorch_distill1",
    2: BASE_DIR / "pytorch_distill2",
    3: BASE_DIR / "pytorch_distill3",
}

LABEL_MAP = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz")
)


# =============================================================================
# DEVICE
# =============================================================================

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        print(f"[Device] {props.name}  |  {props.total_memory/1024**3:.1f} GB VRAM")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transform(augment=False):
    aug = [
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=3),
        transforms.ColorJitter(contrast=0.2, brightness=0.1),
    ] if augment else []
    return transforms.Compose(aug + [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


# =============================================================================
# DATASET
# =============================================================================

def load_dataset():
    """Load full training dataset — same sources as original training."""
    print("[Dataset] Loading EMNIST byclass...")
    train_full = EMNIST(root=str(DATA_DIR), split="byclass", train=True,
                        download=True, transform=get_transform(augment=True))
    test_ds    = EMNIST(root=str(DATA_DIR), split="byclass", train=False,
                        download=True, transform=get_transform(augment=False))

    total       = len(train_full)
    val_count   = int(total * 0.15)
    train_count = total - val_count
    generator   = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(total), [train_count, val_count], generator=generator)

    train_ds = Subset(train_full, train_idx.indices)
    val_base = EMNIST(root=str(DATA_DIR), split="byclass", train=True,
                      download=False, transform=get_transform(augment=False))
    val_ds   = Subset(val_base, val_idx.indices)

    try:
        from supplementary_data import load_supplementary, get_combined_weights
        supp_ds = load_supplementary(
            transform=get_transform(augment=True),
            use_balanced=True, use_digits=True, use_mnist=True,
            use_usps=True, use_svhn=True, use_kaggle=True,
            use_chars_hnd=True, use_chars_img=True, train=True,
        )
        if supp_ds:
            train_ds = ConcatDataset([train_ds, supp_ds])
            print(f"[Dataset] Combined: {len(train_ds):,} samples")
    except ImportError:
        print("[Dataset] supplementary_data.py not found — EMNIST byclass only")

    print(f"[Dataset] Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


# =============================================================================
# MODEL IMPORTS
# =============================================================================

def load_model_classes():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from ocr_pytorch_model  import OCRConvNet
    from ocr_pytorch_model2 import OCRConvNetWide
    from ocr_pytorch_model3 import OCRConvNetTriple
    return OCRConvNet, OCRConvNetWide, OCRConvNetTriple


def load_teacher(model_idx, device):
    """Load a trained model checkpoint as a teacher."""
    OCRConvNet, OCRConvNetWide, OCRConvNetTriple = load_model_classes()
    paths   = {1: M1_PT, 2: M2_PT, 3: M3_PT}
    classes = {1: OCRConvNet, 2: OCRConvNetWide, 3: OCRConvNetTriple}

    model = classes[model_idx](NUM_CLASSES)
    ckpt  = torch.load(str(paths[model_idx]), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  [Teacher] Model {model_idx} loaded from {paths[model_idx]}")
    return model


# =============================================================================
# PHASE 1 — SOFT LABEL GENERATION
# =============================================================================

def generate_soft_labels(device):
    """
    Run all three trained models against the training dataset.
    Save probability distributions (softened by temperature T) to disk.
    Soft labels encode what each model learned — used as teaching signal.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 — Soft Label Generation")
    print(f"  Temperature: {TEMPERATURE}")
    print("=" * 60)

    SOFT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, _, _ = load_dataset()

    # Use no-augmentation loader for soft label generation
    # (we want clean predictions, not augmented ones)
    val_base = EMNIST(root=str(DATA_DIR), split="byclass", train=True,
                      download=False, transform=get_transform(augment=False))
    full_loader = DataLoader(
        val_base, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    for model_idx in [1, 2, 3]:
        out_path = SOFT_DIR / f"soft_labels_m{model_idx}.npy"
        if out_path.exists():
            print(f"  [Skip] Soft labels for Model {model_idx} already exist")
            continue

        print(f"\n  Generating soft labels from Model {model_idx}...")
        teacher = load_teacher(model_idx, device)
        all_probs = []

        with torch.no_grad():
            for images, _ in full_loader:
                images = images.to(device, non_blocking=True)
                logits = teacher(images)
                # Temperature scaling — higher T = softer distributions
                soft   = F.softmax(logits / TEMPERATURE, dim=1)
                all_probs.append(soft.cpu().numpy())

        soft_labels = np.concatenate(all_probs, axis=0)
        np.save(str(out_path), soft_labels)
        print(f"  [Saved] {out_path}  shape: {soft_labels.shape}")

    print("\n[Phase 1] Soft label generation complete.")


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
      alpha    * KL-divergence against soft labels from teacher models
      (1-alpha) * CrossEntropy against hard ground truth labels

    Higher alpha = learn more from teachers.
    Lower alpha  = stay closer to ground truth.
    ALPHA=0.7 balances both signals.
    """
    def __init__(self, alpha=ALPHA, temperature=TEMPERATURE, label_smoothing=0.05):
        super().__init__()
        self.alpha           = alpha
        self.temperature     = temperature
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, soft_targets):
        # Hard label loss
        hard_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # Soft label loss — KL divergence scaled by T^2
        soft_log  = F.log_softmax(logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_log, soft_targets, reduction="batchmean") * (self.temperature ** 2)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience, path):
        self.patience  = patience
        self.path      = path
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss, model):
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


# =============================================================================
# PHASE 2 — DISTILLATION TRAINING
# =============================================================================

def distill_model(student_idx, device):
    """
    Retrain one model using soft labels from the other two as teachers.
    Student model idx: 1, 2, or 3
    Teachers: the other two models
    """
    teacher_idxs = [i for i in [1, 2, 3] if i != student_idx]

    print("\n" + "=" * 60)
    print(f"  PHASE 2 — Distilling Model {student_idx}")
    print(f"  Teachers: Models {teacher_idxs}")
    print(f"  Alpha: {ALPHA}  Temperature: {TEMPERATURE}")
    print("=" * 60)

    out_dir = DISTILL_DIRS[student_idx]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load soft labels from both teachers and average them
    print("\n[Soft Labels] Loading teacher soft labels...")
    soft_arrays = []
    for tidx in teacher_idxs:
        path = SOFT_DIR / f"soft_labels_m{tidx}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Soft labels for Model {tidx} not found. Run Phase 1 first.")
        arr = np.load(str(path))
        soft_arrays.append(arr)
        print(f"  Model {tidx}: {arr.shape}")

    # Average soft labels from both teachers
    avg_soft = np.mean(soft_arrays, axis=0)
    avg_soft_tensor = torch.tensor(avg_soft, dtype=torch.float32)
    print(f"  Averaged soft labels: {avg_soft_tensor.shape}")

    # Load dataset
    train_ds, val_ds, test_ds = load_dataset()

    # Build student model
    OCRConvNet, OCRConvNetWide, OCRConvNetTriple = load_model_classes()
    classes = {1: OCRConvNet, 2: OCRConvNetWide, 3: OCRConvNetTriple}
    student = classes[student_idx](NUM_CLASSES).to(device)

    # Load student's own pretrained weights as starting point
    src_paths = {1: M1_PT, 2: M2_PT, 3: M3_PT}
    ckpt = torch.load(str(src_paths[student_idx]), map_location=device, weights_only=False)
    student.load_state_dict(ckpt["state_dict"])
    print(f"[Student] Loaded pretrained weights from {src_paths[student_idx]}")

    # Batch size adjustments per model
    batch_sizes = {1: 256, 2: 256, 3: 128}
    batch_size  = batch_sizes[student_idx]

    # DataLoaders
    from supplementary_data import _extract_targets
    targets        = _extract_targets(train_ds)
    class_counts   = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Optimizer and scheduler — lower LR since starting from pretrained weights
    lr_map = {1: 3e-5, 2: 1e-4, 3: 1e-3}
    optimizer  = optim.AdamW(student.parameters(), lr=lr_map[student_idx], weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion  = DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)
    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(PATIENCE, str(out_dir / f"best_distill{student_idx}.pt"))

    print(f"\n[Train] Starting distillation — max epochs: {EPOCHS} | batch: {batch_size}")

    # Note: soft labels are indexed by position in the EMNIST byclass train set
    # For the distillation loader we need to track original indices
    # Use the EMNIST byclass base dataset directly with index tracking

    emnist_base = EMNIST(root=str(DATA_DIR), split="byclass", train=True,
                         download=False, transform=get_transform(augment=True))
    emnist_loader = DataLoader(emnist_base, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        student.train()
        total_loss = total_correct = total_samples = 0

        for batch_idx, (images, labels) in enumerate(emnist_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get soft labels for this batch
            # Using batch_idx * batch_size to approximate position
            start = batch_idx * batch_size
            end   = min(start + images.size(0), len(avg_soft_tensor))
            soft  = avg_soft_tensor[start:end].to(device, non_blocking=True)

            if soft.size(0) != images.size(0):
                # Batch size mismatch at end of dataset — use hard labels only
                soft = F.one_hot(labels, NUM_CLASSES).float()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                 enabled=USE_AMP and device.type == "cuda"):
                logits = student(images)
                loss   = criterion(logits, labels, soft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        scheduler.step()
        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # Validation
        student.eval()
        val_loss = val_correct = val_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = student(images)
                loss   = F.cross_entropy(logits, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += images.size(0)

        val_loss /= val_samples
        val_acc   = val_correct / val_samples
        elapsed   = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  |  "
              f"lr: {current_lr:.2e}  [{elapsed:.0f}s]")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[k].append(v)

        early_stop(val_loss, student)
        if early_stop.stop:
            break

    # Load best checkpoint
    ckpt = torch.load(str(out_dir / f"best_distill{student_idx}.pt"),
                      map_location=device, weights_only=False)
    student.load_state_dict(ckpt["state_dict"])

    # Test evaluation with per-class accuracy
    print(f"\n[Eval] Test accuracy for distilled Model {student_idx}...")
    student.eval()
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)
    test_correct = test_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = student(images).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_samples += images.size(0)
            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    test_acc = test_correct / test_samples
    class_acc = class_correct / class_total.clamp(min=1)
    worst = class_acc.argsort()[:15]

    print(f"\n  [Per-Class] 15 worst-performing classes:")
    for idx in worst:
        print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
              f"{class_acc[idx]*100:.1f}%  ({int(class_total[idx])} samples)")

    print(f"\n{'='*50}")
    print(f"  Distilled Model {student_idx} Test accuracy: {test_acc*100:.2f}%")
    print(f"{'='*50}")

    # Save final model and ONNX
    final_path = str(out_dir / f"final_distill{student_idx}.pt")
    torch.save({"state_dict": student.state_dict()}, final_path)
    print(f"[Save] {final_path}")

    onnx_path = str(out_dir / f"ocr_model{student_idx}_distill.onnx")
    student.eval()
    student_cpu = classes[student_idx](NUM_CLASSES)
    student_cpu.load_state_dict(student.state_dict())
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(student_cpu, dummy, onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    size_mb = Path(onnx_path).stat().st_size / 1024**2
    print(f"[ONNX] {onnx_path}  ({size_mb:.1f} MB)")

    return test_acc


# =============================================================================
# PHASE 3 — ONNX VALIDATION
# =============================================================================

def validate_onnx(device):
    """
    Compare original ONNX models vs distilled ONNX models on test set.
    Shows per-model accuracy before and after distillation.
    """
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("  PHASE 3 — ONNX Validation")
    print("=" * 60)

    _, _, test_ds = load_dataset()
    test_loader   = DataLoader(test_ds, batch_size=512, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=False)

    def run_onnx(onnx_path, loader):
        sess = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        inp  = sess.get_inputs()[0].name
        correct = total = 0
        for images, labels in loader:
            arr   = images.numpy()
            logits = sess.run(None, {inp: arr})[0]
            preds  = logits.argmax(1)
            correct += (preds == labels.numpy()).sum()
            total   += len(labels)
        return correct / total

    results = {}
    for idx in [1, 2, 3]:
        orig_path    = [M1_ONNX, M2_ONNX, M3_ONNX][idx-1]
        distill_path = DISTILL_DIRS[idx] / f"ocr_model{idx}_distill.onnx"

        orig_acc = run_onnx(orig_path, test_loader) if orig_path.exists() else None
        dist_acc = run_onnx(distill_path, test_loader) if distill_path.exists() else None

        results[idx] = (orig_acc, dist_acc)
        print(f"\n  Model {idx}:")
        if orig_acc is not None:
            print(f"    Original   : {orig_acc*100:.2f}%")
        if dist_acc is not None:
            print(f"    Distilled  : {dist_acc*100:.2f}%")
            if orig_acc is not None:
                delta = (dist_acc - orig_acc) * 100
                print(f"    Delta      : {delta:+.2f}%")

    print(f"\n{'='*60}")
    print("  Validation complete.")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Knowledge Distillation")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="1=generate soft labels, 2=distill training, 3=ONNX validation")
    parser.add_argument("--model", type=int, choices=[1, 2, 3],
                        help="Model index to distill (required for phase 2)")
    args = parser.parse_args()

    device = setup_device()

    if args.phase == 1:
        generate_soft_labels(device)

    elif args.phase == 2:
        if args.model is None:
            parser.error("--model required for phase 2")
        distill_model(args.model, device)

    elif args.phase == 3:
        validate_onnx(device)