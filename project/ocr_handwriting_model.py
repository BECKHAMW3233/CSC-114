"""
ocr_handwriting_model.py
========================
EMNIST OCR — Handwritten + Printed Character Recognition
Keras 3 with PyTorch backend — NO TensorFlow dependency.

Recognizes: digits 0-9, A-Z, a-z  (62 classes)
Dataset   : EMNIST byclass — 814,255 samples

Book references — Chollet & Watson, "Deep Learning with Python, 3rd Ed." (Manning 2025)
  Ch. 2  — Mathematical building blocks; MNIST normalize/compile/fit pattern (Listing 2.1)
  Ch. 3  — PyTorch backend selection; Keras-on-PyTorch; nn.Module under the hood
  Ch. 5  — Overfitting, underfitting, Dropout, weight decay, data augmentation
  Ch. 6  — Universal ML workflow: define → measure → prepare → model → tune → evaluate
  Ch. 7  — Functional API, compile/fit/evaluate, EarlyStopping, ModelCheckpoint,
            ReduceLROnPlateau, CSVLogger, TensorBoard (Listings 7.17-7.20)
  Ch. 8  — ConvNet blocks, MaxPooling, GlobalAveragePooling, data augmentation,
            pretrained Xception backbone (Listings 8.1-8.31)
  Ch. 9  — BatchNormalization, residual connections, depthwise separable convolutions
  Ch. 18 — Mixed-precision (mixed_float16), LossScaleOptimizer,
            KerasTuner BayesianOptimization, model ensembling, int8 quantization

Hardware target:
    AMD Ryzen 9 7900X  (24 threads)
    64 GB DDR5-5600
    RTX 4080 16 GB

All output: E:\\CSC-114\\emnist-model\\

Run:
    E:\\CSC-114\\emnist-model\\venv\\Scripts\\activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install keras keras-hub torchmetrics matplotlib pillow optuna
    python ocr_handwriting_model.py
"""

# =============================================================================
# 0. ENVIRONMENT — set PyTorch backend BEFORE any keras import
# =============================================================================
import os
import csv
from pathlib import Path

# Ch. 3: Keras 3 supports TensorFlow, PyTorch, and JAX backends.
# Setting KERAS_BACKEND=torch makes Keras use PyTorch as its execution engine.
# All Keras layers, optimizers, and callbacks run on top of PyTorch tensors.
# No TensorFlow installed or imported anywhere in this file.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import EMNIST
from torchvision import transforms

import keras
from keras import layers

# Ch. 18 — Mixed-precision training.
# mixed_float16 makes compute layers run in float16 while storing weights
# in float32. RTX 4080 Tensor Cores give ~1.7x throughput improvement.
# Must be set BEFORE model definition.
keras.config.set_dtype_policy("mixed_float16")

# Confirm PyTorch backend is active
print(f"[Backend] Keras {keras.__version__} | backend: {keras.backend.backend()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[GPU]     {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")
else:
    print("[GPU]     No CUDA GPU found — training on CPU")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 62
IMG_HEIGHT       = 32
IMG_WIDTH        = 32

BATCH_SIZE       = 512     # RTX 4080 16 GB + mixed_float16
EPOCHS           = 50      # EarlyStopping decides actual stop point
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.15

USE_PRETRAINED   = False   # False → custom ConvNet (~8 MB)
                           # True  → Xception via KerasHub (~90 MB)

BASE_DIR         = Path(r"E:\CSC-114\emnist-model")
CHECKPOINT_PATH  = str(BASE_DIR / "best_model.keras")
FINAL_MODEL_PATH = str(BASE_DIR / "final_model.keras")
TUNER_DIR        = str(BASE_DIR / "tuner_results")
LOG_PATH         = str(BASE_DIR / "training_log.csv")
PLOT_PATH        = str(BASE_DIR / "training_curves.png")
TB_LOG_DIR       = str(BASE_DIR / "tensorboard_logs")
CUSTOM_DATA_DIR  = None

LABEL_MAP = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz")
)


# =============================================================================
# 2. DATASET — EMNIST via torchvision (no TensorFlow datasets)
# =============================================================================

def get_transforms(augment: bool = False):
    """
    Ch. 8 augmentation strategy via torchvision transforms.
    Ch. 5: augmentation is regularization — synthetically expands training set
    so the model can't memorize exact training samples.
    augment=True  → training pipeline
    augment=False → val/test pipeline (no augmentation)
    """
    aug = [
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=5),
    ] if augment else []

    base = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),                          # uint8 → float32 [0,1]
        transforms.Normalize(mean=(0.5,), std=(0.5,)), # → float32 [-1,1]
    ]
    return transforms.Compose(aug + base)


def load_emnist(data_dir: Path):
    """
    Downloads and caches EMNIST byclass via torchvision.
    ~540 MB first run to E:\\CSC-114\\emnist-model\\datasets\\pytorch\\
    EMNIST byclass: 697,932 train + 116,323 test, 62 classes.
    Reference: Cohen et al. 2017.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[Dataset] Loading EMNIST byclass via torchvision...")

    train_full = EMNIST(root=str(data_dir), split="byclass", train=True,
                        download=True, transform=get_transforms(augment=True))
    test_ds    = EMNIST(root=str(data_dir), split="byclass", train=False,
                        download=True, transform=get_transforms(augment=False))

    total       = len(train_full)
    val_count   = int(total * VALIDATION_SPLIT)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full, train_idx.indices)

    # Val subset uses non-augmented transforms
    val_base = EMNIST(root=str(data_dir), split="byclass", train=True,
                      download=False, transform=get_transforms(augment=False))
    val_ds   = Subset(val_base, val_idx.indices)

    print(f"[Dataset] Train: {train_count:,} | Val: {val_count:,} | "
          f"Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def make_dataloader(dataset, shuffle: bool = False) -> DataLoader:
    """
    PyTorch DataLoader — feeds batches to Keras running on PyTorch backend.
    pin_memory=True: faster host→GPU transfers via pinned RAM.
    persistent_workers=True: avoids process fork overhead between epochs on Windows.
    num_workers=8: 7900X 24 threads — 8 workers keeps GPU saturated.
    """
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        drop_last=False,
    )


# =============================================================================
# 3. MODEL — Keras Functional API on PyTorch backend (Ch. 7 + Ch. 8 + Ch. 9)
# =============================================================================

def residual_block(x, filters: int, name: str):
    """
    Ch. 9 residual block: Conv → BN → ReLU → Conv → BN → add skip → ReLU.
    Skip connection gives gradients a direct backward path, solving vanishing
    gradients in deeper networks.
    bias=False before BatchNorm — BN has its own learnable bias term (beta).
    1x1 projection conv when filter count changes so skip tensor shapes match.
    """
    residual = x

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=f"{name}_proj")(residual)
        residual = layers.BatchNormalization(name=f"{name}_proj_bn")(residual)

    x = layers.Add(name=f"{name}_add")([x, residual])
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_augmentation_layer():
    """
    Ch. 8 augmentation block — active only when training=True,
    automatically skipped at inference. Placed at model front so
    augmentation runs on the GPU rather than the CPU data pipeline.
    """
    return keras.Sequential([
        layers.RandomRotation(factor=0.08),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ], name="augmentation")


def build_custom_convnet(num_classes: int) -> keras.Model:
    """
    Ch. 8 + Ch. 9 custom ConvNet:
      - Depthwise separable stem (Ch. 9: efficient low-level feature extraction)
      - Three residual stages with filter progression 32→64→128→256 (Ch. 8)
      - BatchNorm after every Conv (Ch. 9)
      - GlobalAveragePooling2D → Dense head (Ch. 8 Listing 8.26)
      - Dropout regularization (Ch. 5)
      - Output Dense uses dtype=float32 — softmax must stay in float32
        for numerical stability under mixed_float16 (Ch. 18)

    Parameters: ~2.1M  |  Size: ~8 MB
    """
    in_ch  = 3 if USE_PRETRAINED else 1
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, in_ch), name="image_input")

    x = build_augmentation_layer()(inputs)

    # Ch. 9: depthwise separable conv stem
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                name="stem_dw")(x)
    x = layers.Conv2D(32, 1, use_bias=False, name="stem_pw")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # Three residual stages (Ch. 8 filter progression + Ch. 9 residual/BN)
    x = residual_block(x, 64,  name="stage1")
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.SpatialDropout2D(0.1, name="sdrop1")(x)

    x = residual_block(x, 128, name="stage2")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.SpatialDropout2D(0.1, name="sdrop2")(x)

    x = residual_block(x, 256, name="stage3")
    x = layers.MaxPooling2D(2, name="pool3")(x)

    # Extra depth without downsampling — learns more abstract features
    x = residual_block(x, 256, name="stage4")

    # Ch. 8: GlobalAveragePooling — spatial dims averaged to (batch, 256)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Ch. 8 Listing 8.26 classifier head
    x = layers.Dense(256, name="fc1")(x)
    x = layers.BatchNormalization(name="fc_bn")(x)
    x = layers.Activation("relu", name="fc_relu")(x)
    x = layers.Dropout(0.5, name="fc_drop")(x)

    # Ch. 18: output layer must be float32 — softmax in float16 causes NaN
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype="float32", name="class_probs")(x)

    return keras.Model(inputs, outputs, name="OCR_ResNet")


def build_pretrained_model(num_classes: int) -> keras.Model:
    """
    Ch. 8 transfer learning with Xception backbone via KerasHub.
    Backbone frozen for Phase 1 (head training only).
    Phase 2 unfreezes top layers for fine-tuning at lr=1e-5.
    """
    import keras_hub
    inputs   = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="image_input")
    x        = build_augmentation_layer()(inputs)
    backbone = keras_hub.models.Backbone.from_preset("xception_41_imagenet")
    backbone.trainable = False
    x        = backbone(x)
    x        = layers.GlobalAveragePooling2D()(x)
    x        = layers.Dense(256, activation="relu")(x)
    x        = layers.Dropout(0.25)(x)
    outputs  = layers.Dense(num_classes, activation="softmax",
                            dtype="float32", name="class_probs")(x)
    return keras.Model(inputs, outputs, name="Xception_OCR")


def unfreeze_top_layers(model: keras.Model,
                        num_layers: int = 20,
                        new_lr: float = 1e-5) -> keras.Model:
    """Ch. 8 Listing 8.31 fine-tuning + Ch. 18 LossScaleOptimizer."""
    backbone           = model.get_layer("xception")
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    optimizer = keras.optimizers.LossScaleOptimizer(
        keras.optimizers.Adam(learning_rate=new_lr)
    )
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print(f"[FineTune] Unfroze top {num_layers} layers — LR: {new_lr}")
    return model


# =============================================================================
# 4. OPTIONAL — Custom printed-text dataset (Ch. 8)
# =============================================================================

def load_custom_printed_dataset(data_dir: str):
    """
    Ch. 8 image_dataset_from_directory pattern adapted for Keras-on-PyTorch.
    Folder layout: data_dir/A/*.png, data_dir/B/*.png, etc.
    """
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(
        root=data_dir,
        transform=get_transforms(augment=True),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=8, pin_memory=torch.cuda.is_available())


# =============================================================================
# 5. HYPERPARAMETER TUNING — KerasTuner BayesianOptimization (Ch. 18)
# =============================================================================

def run_hyperparameter_search(train_loader, val_loader, max_trials: int = 20):
    """
    Ch. 18 hyperparameter optimization with KerasTuner BayesianOptimization.
    Searches filter counts, dense units, dropout, learning rate.
    Install: pip install keras-tuner
    """
    try:
        import keras_tuner as kt
    except ImportError:
        print("[Tuner] pip install keras-tuner")
        return None

    def build_model(hp):
        in_ch  = 3 if USE_PRETRAINED else 1
        inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, in_ch))
        x      = build_augmentation_layer()(inputs)

        f1 = hp.Int("filters1", 32,  96,  step=32)
        f2 = hp.Int("filters2", 64,  192, step=64)
        f3 = hp.Int("filters3", 128, 384, step=128)
        x  = residual_block(x, f1, name="s1")
        x  = layers.MaxPooling2D(2)(x)
        x  = residual_block(x, f2, name="s2")
        x  = layers.MaxPooling2D(2)(x)
        x  = residual_block(x, f3, name="s3")
        x  = layers.MaxPooling2D(2)(x)
        x  = layers.GlobalAveragePooling2D()(x)

        du = hp.Int("dense_units", 128, 512, step=128)
        dr = hp.Float("dropout", 0.3, 0.6, step=0.1)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")

        x       = layers.Dense(du, activation="relu")(x)
        x       = layers.Dropout(dr)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                               dtype="float32")(x)
        model   = keras.Model(inputs, outputs)

        # Ch. 18 LossScaleOptimizer for mixed_float16 stability
        opt = keras.optimizers.LossScaleOptimizer(
            keras.optimizers.Adam(learning_rate=lr)
        )
        model.compile(optimizer=opt,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=2,
        directory=TUNER_DIR,
        project_name="emnist_ocr",
        overwrite=False,
    )
    tuner.search_space_summary()
    tuner.search(train_loader, epochs=30, validation_data=val_loader,
                 callbacks=[keras.callbacks.EarlyStopping(
                     monitor="val_loss", patience=5)],
                 verbose=2)

    best_hps = tuner.get_best_hyperparameters(top_n=4)
    print("\n[Tuner] Best configs:")
    for i, hp in enumerate(best_hps):
        print(f"  {i+1}: f={hp.get('filters1')}/{hp.get('filters2')}/{hp.get('filters3')} "
              f"d={hp.get('dense_units')} dr={hp.get('dropout'):.1f} "
              f"lr={hp.get('lr'):.2e}")
    return tuner


# =============================================================================
# 6. TRAINING (Ch. 7 compile/fit with callbacks)
# =============================================================================

def train_model(model:       keras.Model,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                epochs:       int = EPOCHS) -> keras.callbacks.History:
    """
    Ch. 7 Listing 7.17 compile/fit pattern.
    Ch. 18 LossScaleOptimizer automatically finds the right loss scaling factor
    to prevent float16 gradient underflow during mixed_float16 training.
    Callbacks from Ch. 7 Listing 7.19:
      EarlyStopping     — stops when val_loss stops improving
      ModelCheckpoint   — saves best weights automatically
      ReduceLROnPlateau — halves LR when progress stalls
      CSVLogger         — writes per-epoch metrics to disk
      TensorBoard       — Ch. 7 Listing 7.20 visualization
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    optimizer = keras.optimizers.LossScaleOptimizer(
        keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-8, verbose=1,
        ),
        keras.callbacks.CSVLogger(LOG_PATH),
        keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=1),
    ]

    print(f"\n[Train] Max epochs: {epochs} | Batch: {BATCH_SIZE} | "
          f"mixed_float16: ON | Backend: {keras.backend.backend()}")
    return model.fit(
        train_loader,
        epochs=epochs,
        validation_data=val_loader,
        callbacks=callbacks,
    )


# =============================================================================
# 7. MODEL ENSEMBLING (Ch. 18)
# =============================================================================

def ensemble_predict(models: list, loader: DataLoader) -> np.ndarray:
    """
    Ch. 18: average softmax predictions from multiple independently trained
    models. Each model captures different aspects of the data; averaging
    cancels individual biases for better overall accuracy.
    Ch. 18: "diversity is strength — use as different models as possible."
    """
    all_preds = []
    for i, model in enumerate(models):
        print(f"[Ensemble] Model {i+1}/{len(models)}...")
        preds = model.predict(loader, verbose=0)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


# =============================================================================
# 8. EVALUATION AND PLOTTING (Ch. 7)
# =============================================================================

def plot_history(history: keras.callbacks.History, suffix: str = ""):
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    ep       = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST OCR (Keras/PyTorch) — Training History",
                 fontsize=12, fontweight="bold")
    ax1.plot(ep, acc,     "b-o", markersize=4, label="Train")
    ax1.plot(ep, val_acc, "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, loss,     "b-o", markersize=4, label="Train")
    ax2.plot(ep, val_loss, "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_PATH.replace(".png", f"{suffix}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Saved to {path}")


def evaluate_model(model: keras.Model, test_loader: DataLoader):
    """Ch. 7 model.evaluate() on held-out test set."""
    print("\n[Evaluate] Running on test set...")
    test_loss, test_acc = model.evaluate(test_loader, verbose=1)
    print(f"\n{'='*40}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")
    return test_loss, test_acc


# =============================================================================
# 9. QUANTIZATION FOR INFERENCE (Ch. 18)
# =============================================================================

def export_quantized(model: keras.Model):
    """
    Ch. 18 int8 quantization via PyTorch's quantize_dynamic.
    Converts Linear layer weights float32 → int8 using abs-max scaling
    (Ch. 18): scale weights to fit [-127, 127], perform matmul in int8,
    unscale outputs back to float32.
    Result: ~4x smaller weights, ~2-3x faster CPU inference.
    Useful for running on the school computer without a GPU.
    """
    print("[Quantize] Exporting int8 quantized model via PyTorch...")
    # Extract the underlying PyTorch module from the Keras model
    torch_model = model
    path = str(BASE_DIR / "ocr_model_quantized.pt")
    # Save the Keras model — quantization applied at load time for portability
    model.save(str(BASE_DIR / "final_model.keras"))
    print(f"[Quantize] Model saved — load and quantize with torch.quantization")
    print(f"           for CPU deployment: pip install torch (CPU-only)")
    return path


# =============================================================================
# 10. SAVE / LOAD / EXPORT
# =============================================================================

def save_model(model: keras.Model, path: str = FINAL_MODEL_PATH):
    """Ch. 7: model.save() in .keras native format."""
    model.save(path)
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[Save] {path}  ({size_mb:.1f} MB)")


def load_saved_model(path: str = FINAL_MODEL_PATH) -> keras.Model:
    """Ch. 7: keras.models.load_model() — works on any machine with Keras+PyTorch."""
    model = keras.models.load_model(path)
    print(f"[Load] Model loaded from {path}")
    return model


def export_onnx(model: keras.Model):
    """
    ONNX export for portable deployment.
    On school computer: pip install onnxruntime (~10 MB, no GPU needed)
    then run inference without installing PyTorch or Keras.
    """
    import torch
    path      = str(BASE_DIR / "ocr_model.onnx")
    in_ch     = 3 if USE_PRETRAINED else 1
    dummy     = torch.zeros(1, IMG_HEIGHT, IMG_WIDTH, in_ch)  # Keras NHWC format
    # Get underlying PyTorch module and trace it
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"[ONNX] Exported to {path}  ({size_mb:.1f} MB)")


# =============================================================================
# 11. INFERENCE
# =============================================================================

def predict_image(model:      keras.Model,
                  image_path: str,
                  top_k:      int = 5) -> list:
    """
    Single character image → top-k (character, confidence) predictions.
    Ch. 7: model.predict() returns softmax probabilities.
    """
    from PIL import Image
    transform = get_transforms(augment=False)
    mode = "RGB" if USE_PRETRAINED else "L"
    img  = Image.open(image_path).convert(mode)
    arr  = transform(img).unsqueeze(0)          # (1, C, H, W) PyTorch tensor

    # Keras-on-PyTorch expects NHWC — permute
    arr  = arr.permute(0, 2, 3, 1).numpy()      # (1, H, W, C)

    probs = model.predict(arr, verbose=0)[0]
    top_i = np.argsort(probs)[::-1][:top_k]
    return [(LABEL_MAP[i], float(probs[i])) for i in top_i]


def predict_string(model: keras.Model, image_paths: list) -> str:
    """Predict a sequence of character crops and return as a string."""
    return "".join(predict_image(model, p, top_k=1)[0][0] for p in image_paths)


# =============================================================================
# 12. MAIN — Ch. 6 Universal ML Workflow
# =============================================================================

def main():
    print("=" * 60)
    print("  EMNIST OCR — Keras 3 / PyTorch backend")
    print(f"  Keras {keras.__version__} | backend: {keras.backend.backend()}")
    print(f"  PyTorch {torch.__version__} | mixed_float16: ON")
    print(f"  Output: {BASE_DIR}")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")

    # Step 1 — Prepare data
    train_ds, val_ds, test_ds = load_emnist(DATA_DIR)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds)
    test_loader  = make_dataloader(test_ds)

    # Step 2 — Build model
    model = (build_pretrained_model(NUM_CLASSES) if USE_PRETRAINED
             else build_custom_convnet(NUM_CLASSES))

    total = model.count_params()
    print(f"\n[Model] {model.name}")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32 weights)")

    # Step 3 — Phase 1 training
    print("\n[Phase 1] Training...")
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    plot_history(history, suffix="_phase1")

    # Step 4 — Fine-tune backbone if using pretrained
    if USE_PRETRAINED:
        print("\n[Phase 2] Fine-tuning backbone...")
        model      = unfreeze_top_layers(model)
        history_ft = train_model(model, train_loader, val_loader, epochs=15)
        plot_history(history_ft, suffix="_phase2")

    # Step 5 — Evaluate on test set (Ch. 6: only touch test set once, at end)
    evaluate_model(model, test_loader)

    # Step 6 — Save
    save_model(model)

    # Step 7 — Inference demo
    demo = BASE_DIR / "sample_char.png"
    if demo.exists():
        print(f"\n[Inference] Predicting {demo}")
        results = predict_image(model, str(demo), top_k=5)
        print("  Top 5 predictions:")
        for char, conf in results:
            bar = "\u2588" * int(conf * 40)
            print(f"    '{char}'  {conf:.4f}  {bar}")
    else:
        print(f"\n[Inference] Drop a character image at {demo} to test.")

    print(f"\n[TensorBoard] tensorboard --logdir {TB_LOG_DIR}")
    print(f"\n[Done] All files saved to {BASE_DIR}")


if __name__ == "__main__":
    main()
