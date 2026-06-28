
# ---------------------------------------------------------------------------
# 10. Supplementary dataset check
# ---------------------------------------------------------------------------
print("[10] Checking supplementary datasets...")
from pathlib import Path
import numpy as np

kaggle_imgs = Path(r"E:\CSC-114\emnist-model\datasets\kaggle\az_images.npy")
kaggle_lbls = Path(r"E:\CSC-114\emnist-model\datasets\kaggle\az_labels.npy")

if kaggle_imgs.exists() and kaggle_lbls.exists():
    imgs = np.load(str(kaggle_imgs))
    lbls = np.load(str(kaggle_lbls))
    print(f"    [OK] Kaggle A-Z: {len(imgs):,} images  shape={imgs.shape}")
else:
    print("    [INFO] Kaggle A-Z not downloaded yet")
    print("           Run: python download_datasets.py")

try:
    from torchvision.datasets import EMNIST
    from torchvision import transforms
    balanced = EMNIST(
        root=r"E:\CSC-114\emnist-model\datasets\pytorch",
        split="balanced", train=True, download=False,
        transform=transforms.ToTensor()
    )
    print(f"    [OK] EMNIST Balanced: {len(balanced):,} training samples")
except Exception as e:
    print(f"    [INFO] EMNIST Balanced: {e}")

print()
