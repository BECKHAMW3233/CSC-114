import subprocess
import sys

PYTHON = sys.executable

packages = [
    "onnxruntime",
    "numpy",
    "opencv-python",
    "pillow",
    "requests",
]

print("=" * 60)
print("  EMNIST OCR — Dependency Installer")
print(f"  Python: {PYTHON}")
print("=" * 60)

for pkg in packages:
    print(f"\n[Installing] {pkg}...")
    result = subprocess.run(
        [PYTHON, "-m", "pip", "install", pkg],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = [l for l in result.stdout.strip().splitlines()
                 if "Successfully" in l or "already satisfied" in l]
        for line in lines:
            print(f"  {line}")
    else:
        print(f"  [FAILED] {pkg}")
        print(f"  {result.stderr.strip()}")

print("\n" + "=" * 60)
print("  Verifying installs...")
print("=" * 60)

checks = [
    ("onnxruntime", "onnxruntime"),
    ("numpy",       "numpy"),
    ("opencv",      "cv2"),
    ("Pillow",      "PIL"),
    ("requests",    "requests"),
]

all_ok = True
for name, module in checks:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        print(f"  [OK] {name:<20} {version}")
    except ImportError:
        print(f"  [FAIL] {name}")
        all_ok = False

print()
if all_ok:
    print("  All packages installed. Ready to run ocr_pipeline.py")
else:
    print("  Some packages failed. Check errors above.")
print("=" * 60)