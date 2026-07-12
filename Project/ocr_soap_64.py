"""
ocr_soap_64.py
==============
MNIST OCR — OCRConvNetTriple, SOAP optimizer, 64×64
Pure PyTorch — Triple-width channels + Multi-Scale feature pyramid fusion.

Architecture — OCRConvNetTriple:
  Channel progression: 96→192→384→768
  Feature pyramid: concatenates pooled outputs from stages 2+3+4 (fused_dim=1344)
  Classifier head: 1344→1024→512→256→128→10 (5 layers, GELU)
  SE reduction: 16, drop_path=0.05, dropout=0.35 (first classifier layer)
  label_smoothing=0.05
  ~4.6M parameters

Install: pip install pytorch_optimizer psutil

OPTIMIZER — SOAP (Shampoo + Adam, Kronecker-factored second-order):
  lr=1e-3, betas=(0.95, 0.95), weight_decay=5e-4
  precondition_frequency=100 — Kronecker factor update every 100 steps.
    Was 10; raised to 100. At freq=10 with ~143K steps/epoch the CPU-side
    eigendecomposition for Kronecker factoring ran continuously, producing
    ~50% CUDA / ~50% CPU split and zero epochs completed in 90 minutes.
    At 100 the preconditioning overhead drops 10x while curvature information
    is still updated ~1,435 times per epoch.
  500-step linear warmup before CosineAnnealingLR
  PATIENCE=20
  Standard first-order backward — no create_graph needed.
  AMP disabled — Kronecker eigendecomposition requires float32.
  256×256 viability TBD — standard backward gives it the best chance after Lion.

Augmentation: rotation ±5°, affine, perspective, blur+noise

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB  — batch TBD (auto-detected, candidates 512→256→128→64)

Output: E:\\CSC-114\\project\\pytorch_soap_64\\
"""

import os
import sys
import json
import csv
import time
import datetime
import subprocess
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort

try:
    from pytorch_optimizer import SOAP
except ImportError:
    print("ERROR: pytorch_optimizer not installed. Run: pip install pytorch_optimizer")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from supplementary_data import load_supplementary, get_combined_weights
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — digit supplementary data unavailable")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 64
NUM_CLASSES   = 10
PATIENCE      = 20
LR            = 1e-3
BETAS         = (0.95, 0.95)
WEIGHT_DECAY  = 5e-4
PRECOND_FREQ  = 100   # was 10 — raised to eliminate CPU-bound Kronecker overhead
WARMUP_STEPS  = 500
LABEL_SMOOTH  = 0.05
DROP_PATH     = 0.05
DROPOUT_HEAD  = 0.35
WALL_CLOCK_LIMIT_S = 36000  # 10 hours — the actual stopping condition (patience or this, whichever first)
SCHEDULE_EPOCH_ESTIMATE = 200  # used only to shape the cosine LR curve; NOT a hard epoch cap.
                                # At ~174s/epoch (64x64 SOAP observed pace), 10h wall clock fits
                                # ~206 epochs, so 200 keeps the cosine decay from bottoming out
                                # early if the run goes long. Training itself is bounded by
                                # PATIENCE and WALL_CLOCK_LIMIT_S only — see the main loop below.
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_soap_64")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tee logging ───────────────────────────────────────────────────────────────
class _Tee:
    """
    Mirror stdout AND stderr to a timestamped .log file in OUTPUT_DIR.
    Capturing stderr is required to catch PyTorch UserWarnings that do not
    appear on stdout.
    Log location: <OUTPUT_DIR>/soap_64_<YYYYMMDD_HHMMSS>.log
    """
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"soap_64_{ts}.log")
        self._log    = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout   = self
        sys.stderr   = self
        print(f"  Log: {log_path}")

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._log.close()


# ── GPU monitoring ────────────────────────────────────────────────────────────
def get_gpu_stats():
    vram_alloc = torch.cuda.max_memory_allocated() / 1024**3
    vram_res   = torch.cuda.max_memory_reserved()  / 1024**3
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts     = result.stdout.strip().split(",")
        cuda_util = int(parts[0].strip())
        temp      = int(parts[1].strip())
    except Exception:
        cuda_util, temp = -1, -1
    cpu_pct  = -1
    ram_used = -1
    if HAS_PSUTIL:
        try:
            cpu_pct  = psutil.cpu_percent(interval=None)
            ram_used = round(psutil.virtual_memory().used / 1024**3, 2)
        except Exception:
            pass
    return vram_alloc, vram_res, cuda_util, temp, cpu_pct, ram_used


# ── Architecture ──────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device) < keep
        return x * mask.float() / keep


class TripleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_path=0.0):
        super().__init__()
        mid = out_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se       = SEBlock(out_ch, reduction=16)
        self.drop     = DropPath(drop_path)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.drop(self.se(self.conv(x))) + self.shortcut(x))


class OCRConvNetTriple(nn.Module):
    def __init__(self, num_classes=62, drop_path=0.05, dropout=0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.stage1 = self._make_stage(96,  96,  2, drop_path)
        self.stage2 = self._make_stage(96,  192, 2, drop_path)
        self.stage3 = self._make_stage(192, 384, 2, drop_path)
        self.stage4 = self._make_stage(384, 768, 2, drop_path)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        fused_dim  = 192 + 384 + 768  # 1344

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def _make_stage(self, in_ch, out_ch, num_blocks, drop_path):
        layers = [TripleBlock(in_ch, out_ch, stride=2, drop_path=drop_path)]
        for _ in range(num_blocks - 1):
            layers.append(TripleBlock(out_ch, out_ch, stride=1, drop_path=drop_path))
        return nn.Sequential(*layers)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        f2 = self.pool2(s2).flatten(1)
        f3 = self.pool3(s3).flatten(1)
        f4 = self.pool4(s4).flatten(1)
        return self.head(torch.cat([f2, f3, f4], dim=1))


# ── Warmup + Cosine scheduler ─────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            )
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count":   self.step_count,
            "base_lrs":     self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "total_steps":  self.total_steps,
            "eta_min":      self.eta_min,
        }

    def load_state_dict(self, state):
        self.step_count   = state["step_count"]
        self.base_lrs     = state["base_lrs"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps  = state["total_steps"]
        self.eta_min      = state["eta_min"]


# ── Data loading ──────────────────────────────────────────────────────────────
def build_transform(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def get_loaders(img_size, batch_size):
    from torchvision.datasets import MNIST
    from torch.utils.data import ConcatDataset, random_split, Subset

    DATA_DIR = r"E:\CSC-114\emnist-model\datasets\pytorch"

    train_transform = build_transform(img_size, train=True)
    test_transform  = build_transform(img_size, train=False)

    train_full = MNIST(root=DATA_DIR, train=True,
                        download=True, transform=train_transform)
    test_ds    = MNIST(root=DATA_DIR, train=False,
                        download=True, transform=test_transform)

    total       = len(train_full)
    val_count   = int(total * 0.15)
    train_count = total - val_count
    generator   = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(total), [train_count, val_count],
                                       generator=generator)

    train_ds = Subset(train_full, train_idx.indices)
    val_base = MNIST(root=DATA_DIR, train=True,
                      download=False, transform=test_transform)
    val_ds   = Subset(val_base, val_idx.indices)

    if HAS_SUPPLEMENTARY:
        supp_ds = load_supplementary(
            transform=train_transform,
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
            combined       = ConcatDataset([train_ds, supp_ds])
            sample_weights = get_combined_weights(train_ds, supp_ds)
            print(f"[Dataset] Combined: {len(combined):,} samples")
        else:
            combined = train_ds
            targets        = torch.tensor(
                [int(train_full.targets[i]) for i in train_idx.indices], dtype=torch.long
            )
            class_counts   = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
            class_weights  = 1.0 / class_counts
            sample_weights = class_weights[targets]
    else:
        combined = train_ds
        targets        = torch.tensor(
            [int(train_full.targets[i]) for i in train_idx.indices], dtype=torch.long
        )
        class_counts   = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
        class_weights  = 1.0 / class_counts
        sample_weights = class_weights[targets]

    sampler  = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dl = DataLoader(combined, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True, persistent_workers=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True, persistent_workers=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True, persistent_workers=False)
    return train_dl, val_dl, test_dl


# ── Batch size probe ──────────────────────────────────────────────────────────
def try_batch_size(img_size):
    """Standard backward probe — SOAP uses first-order gradients only."""
    for bs in [512, 256, 128, 64]:
        try:
            torch.cuda.empty_cache()
            # Pre-flight: skip if free VRAM < 1.5GB to avoid shared memory spillover
            _free_vram = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_reserved()) / 1024**3
            if _free_vram < 1.5:
                print(f'  Batch size {bs} — insufficient free VRAM ({_free_vram:.1f}GB free), skipping')
                continue
            model = OCRConvNetTriple(num_classes=NUM_CLASSES, drop_path=DROP_PATH, dropout=DROPOUT_HEAD).to(DEVICE)
            dummy = torch.zeros(bs, 1, img_size, img_size, device=DEVICE)
            out   = model(dummy)
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
                if torch.cuda.memory_allocated() / 1024**3 > 15.0:
                    raise RuntimeError('out of memory')
            del model, dummy, out, loss
            torch.cuda.empty_cache()
            # Check for silent shared VRAM spillover (Windows)
            _alloc_mb = torch.cuda.memory_allocated() / 1024**2
            _total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            if _alloc_mb > _total_mb * 0.95:
                del model, dummy, out, loss
                torch.cuda.empty_cache()
                print(f'  Batch size {bs} — spilling to shared VRAM, stepping down')
                continue
            print(f"  Batch size {bs} — OK")
            return bs
        except RuntimeError as e:
            _emsg = str(e).lower()
            if "out of memory" in _emsg or "find was unable" in _emsg or "engine" in _emsg:
                torch.cuda.empty_cache()
                print(f"  Batch size {bs} — failed ({type(e).__name__}), stepping down")
            else:
                raise
    raise RuntimeError("OOM at all batch sizes — cannot continue")


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


def export_onnx(model, img_size, path):
    model.eval()
    dummy = torch.zeros(1, 1, img_size, img_size, device=DEVICE)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"  ONNX exported and validated: {path}")


def plot_curves(log_path, out_path):
    epochs, tr_acc, va_acc = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_acc.append(float(row["train_acc"]))
            va_acc.append(float(row["val_acc"]))
    fig, ax = plt.subplots()
    ax.plot(epochs, tr_acc, label="Train")
    ax.plot(epochs, va_acc, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("OCRConvNetTriple — SOAP 64x64")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = _Tee(OUTPUT_DIR)

    print("=" * 60)
    print("  OCRConvNetTriple — SOAP — 64x64")
    print(f"  Device : {DEVICE}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    batch_size = try_batch_size(IMG_SIZE)
    train_dl, val_dl, test_dl = get_loaders(IMG_SIZE, batch_size)

    total_steps = SCHEDULE_EPOCH_ESTIMATE * len(train_dl)

    model     = OCRConvNetTriple(num_classes=NUM_CLASSES, drop_path=DROP_PATH, dropout=DROPOUT_HEAD).to(DEVICE)
    optimizer = SOAP(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        precondition_frequency=PRECOND_FREQ,
    )
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {param_count:,}")
    print(f"  Batch size : {batch_size}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup     : {WARMUP_STEPS} steps")
    print(f"  Precond Hz : every {PRECOND_FREQ} steps")
    print()

    log_path    = os.path.join(OUTPUT_DIR, "soap_64_training_log.csv")
    resume_path = os.path.join(OUTPUT_DIR, "soap_64_resume.json")
    best_path   = os.path.join(OUTPUT_DIR, "soap_64_best.pt")
    final_path  = os.path.join(OUTPUT_DIR, "soap_64_final.pt")
    onnx_path   = os.path.join(OUTPUT_DIR, "soap_64.onnx")
    curve_path  = os.path.join(OUTPUT_DIR, "soap_64_curves.png")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "lr", "epoch_time_s",
            "vram_peak_alloc_gb", "vram_peak_reserved_gb", "cuda_util_pct", "gpu_temp_c",
            "cpu_pct", "ram_used_gb",
        ])

    best_acc    = 0.0
    patience_ct = 0

    # ── Resume from previous session ───────────────────────────────────────
    start_epoch = 1
    _rpath_pt = resume_path.replace(".json", ".pt")
    if os.path.exists(_rpath_pt) and os.path.exists(best_path):
        try:
            _rs = torch.load(_rpath_pt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(
                torch.load(best_path, map_location=DEVICE, weights_only=True)
            )
            optimizer.load_state_dict(_rs["optimizer_state"])
            scheduler.load_state_dict(_rs["scheduler_state"])
            best_acc    = _rs["best_acc"]
            patience_ct = _rs["patience_counter"]
            start_epoch = _rs["epoch"] + 1
            print(f"[Resume] Loaded from epoch {_rs['epoch']} "
                  f"(best_acc={_rs['best_acc']:.4f}, patience={_rs['patience_counter']}/{PATIENCE})")
            print(f"[Resume] Continuing from epoch {start_epoch}")
        except Exception as _e:
            print(f"[Resume] Could not load state: {_e} — starting fresh")
            start_epoch = 1
    else:
        print("[Resume] No prior checkpoint — starting fresh")

    wall_start = time.time()
    epoch = start_epoch
    while True:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, scheduler, criterion)
        va_loss, va_acc = eval_epoch(model, val_dl, criterion)
        elapsed = time.time() - t0

        vram_a, vram_r, cuda_u, temp, cpu_pct, ram_used = get_gpu_stats()
        lr_now = scheduler.get_lr()[0]

        print(
            f"  Ep {epoch:>3} | "
            f"tr {tr_acc:.4f} | val {va_acc:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.0f}s | "
            f"VRAM {vram_a:.1f}/{vram_r:.1f}GB | "
            f"CUDA {cuda_u}% | {temp}°C | "
            f"CPU {cpu_pct}% | RAM {ram_used}GB"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                f"{va_loss:.6f}", f"{va_acc:.6f}",
                f"{lr_now:.8f}", f"{elapsed:.1f}",
                f"{vram_a:.3f}", f"{vram_r:.3f}", cuda_u, temp,
                cpu_pct, ram_used,
            ])

        if va_acc > best_acc:
            best_acc    = va_acc
            patience_ct = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ↑ New best: {best_acc:.4f}")
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stop at epoch {epoch} — patience {PATIENCE} exhausted")
                break

        # Save resume state
        try:
            _rpath_pt = resume_path.replace(".json", ".pt")
            torch.save({
                "epoch":            epoch,
                "patience_counter": patience_ct,
                "best_acc":         float(best_acc),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
            }, _rpath_pt)
        except Exception as _re:
            print(f"  [Resume] Warning: could not save state: {_re}")

        wall_elapsed = time.time() - wall_start
        if wall_elapsed >= WALL_CLOCK_LIMIT_S:
            print(f"  [Wall clock] 10h limit reached after epoch {epoch} "
                  f"({wall_elapsed/3600:.2f}h) — stopping cleanly.")
            break

        epoch += 1

    torch.save(model.state_dict(), final_path)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    torch.cuda.empty_cache()
    te_loss, te_acc = eval_epoch(model, test_dl, criterion)
    print(f"\n  Test accuracy (best checkpoint): {te_acc:.4f}")

    # Clear resume state — training complete
    if os.path.exists(resume_path.replace(".json", ".pt")):
        os.remove(resume_path.replace(".json", ".pt"))
        print("  [Resume] State cleared — training complete.")

    export_onnx(model, IMG_SIZE, onnx_path)
    plot_curves(log_path, curve_path)

    print()
    print(f"  Best val accuracy  : {best_acc:.4f}")
    print(f"  Test accuracy      : {te_acc:.4f}")
    print(f"  Log                : {log_path}")
    print(f"  ONNX               : {onnx_path}")
    print("=" * 60)

    tee.close()
