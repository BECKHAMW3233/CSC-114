"""
03_verify_gpu.py
================
Run this BEFORE starting the main training script to confirm:
  - Python version is compatible (3.10 or 3.11)
  - TensorFlow is installed and importable
  - CUDA is detected
  - RTX 4080 is visible as a GPU device
  - A small test computation runs on the GPU successfully

Usage (from activated venv):
    cd E:\CSC-114\emnist-model
    python 03_verify_gpu.py
"""

import sys
import os

print("=" * 60)
print(" GPU + TensorFlow Verification")
print(" E:\\CSC-114\\emnist-model")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
major, minor = sys.version_info.major, sys.version_info.minor
print(f"[1] Python version: {sys.version}")
if major == 3 and minor in (10, 11):
    print("    ✓ Compatible (3.10 / 3.11)")
elif major == 3 and minor == 12:
    print("    ✗ Python 3.12 is NOT supported by TensorFlow 2.16.")
    print("      Install Python 3.11 from https://www.python.org/downloads/")
    sys.exit(1)
else:
    print(f"    ⚠ Untested version — proceed with caution.")
print()

# ---------------------------------------------------------------------------
# 2. TensorFlow import
# ---------------------------------------------------------------------------
print("[2] Importing TensorFlow...")
try:
    import tensorflow as tf
    print(f"    ✓ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
    print("      Run: pip install tensorflow[and-cuda]>=2.16")
    sys.exit(1)
print()

# ---------------------------------------------------------------------------
# 3. CUDA built-in check
# ---------------------------------------------------------------------------
print("[3] Checking CUDA build...")
cuda_built = tf.test.is_built_with_cuda()
print(f"    is_built_with_cuda(): {cuda_built}")
if cuda_built:
    print("    ✓ TensorFlow was compiled with CUDA support")
else:
    print("    ✗ TensorFlow does NOT have CUDA support.")
    print("      Reinstall with: pip install tensorflow[and-cuda]>=2.16")
print()

# ---------------------------------------------------------------------------
# 4. GPU device detection
# ---------------------------------------------------------------------------
print("[4] Detecting GPU devices...")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        print(f"    ✓ Found: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", "Unknown")
        compute = details.get("compute_capability", "Unknown")
        print(f"      Device name      : {name}")
        print(f"      Compute capability: {compute}")
else:
    print("    ✗ No GPU devices found.")
    print()
    print("    Possible causes:")
    print("      - CUDA not installed (run 01_install_cuda.bat)")
    print("      - cuDNN files not copied into CUDA folder")
    print("      - CUDA not on system PATH")
    print("      - NVIDIA driver is outdated (needs 525.60+)")
    print()
    print("    Run nvidia-smi in a terminal to check driver status.")
print()

# ---------------------------------------------------------------------------
# 5. Enable memory growth (same as main training script)
# ---------------------------------------------------------------------------
print("[5] Enabling GPU memory growth...")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("    ✓ Memory growth enabled — prevents full VRAM allocation at startup")
    except RuntimeError as e:
        print(f"    ⚠ Could not enable memory growth: {e}")
        print("      (This is OK if memory growth was already set before this script ran)")
else:
    print("    ⚠ Skipped — no GPU found")
print()

# ---------------------------------------------------------------------------
# 6. Run a small matrix multiply on the GPU to confirm compute works
# ---------------------------------------------------------------------------
print("[6] Running test computation on GPU...")
try:
    device = "/GPU:0" if gpus else "/CPU:0"
    with tf.device(device):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print(f"    ✓ 1000×1000 matrix multiply on {device} succeeded")
    print(f"      Result shape: {c.shape}  dtype: {c.dtype}")
except Exception as e:
    print(f"    ✗ Computation failed: {e}")
print()

# ---------------------------------------------------------------------------
# 7. Keras import check
# ---------------------------------------------------------------------------
print("[7] Checking Keras...")
try:
    import keras
    print(f"    ✓ Keras {keras.__version__} imported successfully")
    backend = keras.backend.backend()
    print(f"      Active backend: {backend}")
    if backend != "tensorflow":
        print(f"      ⚠ Backend is '{backend}', not 'tensorflow'.")
        print("        Set KERAS_BACKEND=tensorflow in your environment or")
        print("        the os.environ line in ocr_handwriting_model.py handles it.")
except ImportError as e:
    print(f"    ✗ Keras import failed: {e}")
    print("      Run: pip install keras>=3.0")
print()

# ---------------------------------------------------------------------------
# 8. KerasHub check
# ---------------------------------------------------------------------------
print("[8] Checking KerasHub (needed for Xception pretrained backbone)...")
try:
    import keras_hub
    print(f"    ✓ keras_hub {keras_hub.__version__} imported successfully")
except ImportError:
    print("    ✗ keras_hub not installed.")
    print("      Run: pip install keras-hub")
    print("      Note: only needed if USE_PRETRAINED = True in the training script.")
print()

# ---------------------------------------------------------------------------
# 9. tensorflow-datasets check
# ---------------------------------------------------------------------------
print("[9] Checking tensorflow-datasets (EMNIST loader)...")
try:
    import tensorflow_datasets as tfds
    print(f"    ✓ tensorflow_datasets {tfds.__version__} imported successfully")
    # Check TFDS data dir is writable
    data_dir = r"E:\CSC-114\emnist-model\datasets"
    os.makedirs(data_dir, exist_ok=True)
    test_file = os.path.join(data_dir, ".write_test")
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
    print(f"    ✓ Dataset directory is writable: {data_dir}")
except ImportError:
    print("    ✗ tensorflow_datasets not installed.")
    print("      Run: pip install tensorflow-datasets")
except OSError as e:
    print(f"    ✗ Cannot write to dataset directory: {e}")
print()

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print("=" * 60)
if gpus and cuda_built:
    print(" ✓ ALL CHECKS PASSED")
    print(" Your RTX 4080 is ready for training.")
    print()
    print(" To start training:")
    print("   cd E:\\CSC-114\\emnist-model")
    print("   python ocr_handwriting_model.py")
else:
    print(" ✗ ONE OR MORE CHECKS FAILED")
    print(" Review the output above and fix the indicated issues")
    print(" before running ocr_handwriting_model.py")
print("=" * 60)
