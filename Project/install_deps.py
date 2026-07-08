import subprocess
import sys

PYTHON = sys.executable

# Core inference packages
inference_packages = [
    "onnxruntime",
    "numpy",
    "opencv-python",
    "pillow",
    "requests",
]

# Training packages
training_packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "matplotlib",
    "pandas",
    "kaggle",
    "scipy",
    "certifi",
]

# Advanced optimizer packages
optimizer_packages = [
    "lion-pytorch",
    "schedulefree",
    "pytorch_optimizer",   # SOAP, AdaHessian, and 100+ modern optimizers
    "psutil",              # Hardware monitoring — CPU%, system RAM per epoch
]

# Distillation packages
distillation_packages = [
    "onnx",
]

print("=" * 60)
print("  EMNIST OCR — Dependency Installer")
print(f"  Python: {PYTHON}")
print("=" * 60)

def install_group(name, packages):
    print(f"\n[{name}]")
    for pkg in packages:
        print(f"  Installing {pkg}...")
        result = subprocess.run(
            [PYTHON, "-m", "pip", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().splitlines()
                     if "Successfully" in l or "already satisfied" in l]
            for line in lines:
                print(f"    {line}")
        else:
            print(f"    [FAILED] {pkg}")
            print(f"    {result.stderr.strip()}")

install_group("Inference", inference_packages)
install_group("Training", training_packages)
install_group("Optimizers", optimizer_packages)
install_group("Distillation / Export", distillation_packages)

print("\n" + "=" * 60)
print("  Verifying installs...")
print("=" * 60)

checks = [
    ("onnxruntime",   "onnxruntime"),
    ("onnx",          "onnx"),
    ("numpy",         "numpy"),
    ("opencv",        "cv2"),
    ("Pillow",        "PIL"),
    ("requests",      "requests"),
    ("torch",         "torch"),
    ("torchvision",   "torchvision"),
    ("torchmetrics",  "torchmetrics"),
    ("matplotlib",    "matplotlib"),
    ("pandas",        "pandas"),
    ("scipy",         "scipy"),
    ("certifi",       "certifi"),
    ("kaggle",        "kaggle"),
    ("lion-pytorch",      "lion_pytorch"),
    ("schedulefree",      "schedulefree"),
    ("pytorch_optimizer", "pytorch_optimizer"),
    ("psutil",            "psutil"),
]

all_ok = True
for name, module in checks:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        print(f"  [OK]   {name:<20} {version}")
    except ImportError:
        print(f"  [FAIL] {name}")
        all_ok = False

print()
if all_ok:
    print("  All packages installed.")
    print()
    print("  Ready for:")
    print("    python ocr_pipeline.py              — inference (12-model ensemble)")
    print("    python ocr_pytorch_model.py         — train Model 1 (Lion, auto batch)")
    print("    python ocr_pytorch_model2.py        — train Model 2 (SF-AdamW, auto batch)")
    print("    python ocr_pytorch_model3.py        — train Model 3 (SGD, auto batch)")
    print("    python ocr_adahessian_64.py         — experimental: AdaHessian 64x64")
    print("    python ocr_adahessian_128.py        — experimental: AdaHessian 128x128")
    print("    python ocr_soap_64.py               — experimental: SOAP 64x64")
    print("    python ocr_soap_128.py              — experimental: SOAP 128x128")
    print("    python ocr_distillation.py          — knowledge distillation")
    print("    python download_datasets.py         — download all datasets")
    print("    python verify_datasets.py           — verify dataset integrity")
else:
    print("  Some packages failed. Check errors above.")
print("=" * 60)