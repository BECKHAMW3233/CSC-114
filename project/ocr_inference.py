"""
ocr_inference.py
================
EMNIST OCR — Inference Script
Runs a handwritten character image through all three trained models
and returns the predicted character with confidence scores.

Usage:
    python ocr_inference.py <image_path>
    python ocr_inference.py C:\\path\\to\\image.png

    Or import and use predict() directly in your own code.

Supported image formats: PNG, JPG, JPEG, BMP, TIFF
Image can be any size — automatically resized to 32x32 grayscale.

Requirements:
    - All three .pt checkpoint files must exist in their respective folders
    - venv must be activated (or full python path used)

Output:
    Top-5 predictions with confidence percentages from each model
    and the ensemble (averaged softmax) result.
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# =============================================================================
# PATHS — update these if your folder structure is different
# =============================================================================

BASE          = Path(r"E:\CSC-114\emnist-model")
MODEL1_PATH   = BASE / "pytorch"  / "best_model.pt"
MODEL2_PATH   = BASE / "pytorch2" / "best_model2.pt"
MODEL3_PATH   = BASE / "pytorch3" / "best_model3.pt"
MODEL1_SRC    = BASE / "ocr_pytorch_model.py"
MODEL2_SRC    = BASE / "ocr_pytorch_model2.py"
MODEL3_SRC    = BASE / "ocr_pytorch_model3.py"

# =============================================================================
# LABEL MAP — 62 classes: 0-9, A-Z, a-z
# =============================================================================

LABEL_MAP = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz")
)
NUM_CLASSES = len(LABEL_MAP)  # 62

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])


def load_image(image_path: str) -> torch.Tensor:
    """
    Load an image file and preprocess it for the OCR models.

    Handles:
    - Color images (converted to grayscale)
    - Any size (resized to 32x32)
    - Dark-on-light or light-on-dark (auto-inverted if needed)

    Returns tensor of shape (1, 1, 32, 32) ready for model input.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("L")  # force grayscale

    # Auto-invert: EMNIST expects white character on black background
    # If image is mostly dark (background is dark), it's already correct
    # If image is mostly light (white paper with dark ink), invert it
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    tensor = TRANSFORM(img)           # (1, 32, 32)
    return tensor.unsqueeze(0)        # (1, 1, 32, 32)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(device: torch.device):
    """
    Load all three trained model architectures and their checkpoints.
    Imports the model classes directly from the training scripts.
    """
    import importlib.util

    def import_class(src_path, class_name):
        spec   = importlib.util.spec_from_file_location("mod", str(src_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    print("[Loading] Model 1 (Narrow ResNet)...")
    OCRConvNet     = import_class(MODEL1_SRC, "OCRConvNet")
    model1         = OCRConvNet(NUM_CLASSES)
    ckpt1          = torch.load(str(MODEL1_PATH), map_location=device,
                                weights_only=False)
    model1.load_state_dict(ckpt1["state_dict"])
    model1         = model1.to(device).eval()

    print("[Loading] Model 2 (Wide + SE)...")
    OCRConvNetWide = import_class(MODEL2_SRC, "OCRConvNetWide")
    model2         = OCRConvNetWide(NUM_CLASSES)
    ckpt2          = torch.load(str(MODEL2_PATH), map_location=device,
                                weights_only=False)
    model2.load_state_dict(ckpt2["state_dict"])
    model2         = model2.to(device).eval()

    print("[Loading] Model 3 (Triple + Multi-Scale)...")
    OCRConvNetTriple = import_class(MODEL3_SRC, "OCRConvNetTriple")
    model3           = OCRConvNetTriple(NUM_CLASSES)
    ckpt3            = torch.load(str(MODEL3_PATH), map_location=device,
                                  weights_only=False)
    model3.load_state_dict(ckpt3["state_dict"])
    model3           = model3.to(device).eval()

    return model1, model2, model3


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict(image_path: str,
            model1=None, model2=None, model3=None,
            device=None,
            top_k: int = 5) -> dict:
    """
    Run a single image through all three models and return predictions.

    Args:
        image_path: path to the image file
        model1, model2, model3: pre-loaded models (loaded if None)
        device: torch device (auto-detected if None)
        top_k: number of top predictions to return

    Returns dict with:
        - ensemble: top-k predictions from weighted ensemble
        - model1/2/3: top-k predictions from each individual model
        - predicted_char: the top ensemble prediction
        - confidence: confidence percentage of top prediction
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if any(m is None for m in [model1, model2, model3]):
        model1, model2, model3 = load_models(device)

    # Load and preprocess image
    tensor = load_image(image_path).to(device)

    # Get softmax probabilities from each model
    p1 = F.softmax(model1(tensor), dim=1)   # (1, 62)
    p2 = F.softmax(model2(tensor), dim=1)
    p3 = F.softmax(model3(tensor), dim=1)

    # Weighted ensemble — weights from test accuracy
    ensemble = 0.38 * p1 + 0.38 * p2 + 0.24 * p3

    def top_k_results(probs, k):
        values, indices = probs[0].topk(k)
        return [
            {"char": LABEL_MAP[idx.item()],
             "confidence": f"{val.item()*100:.1f}%",
             "class_id": idx.item()}
            for val, idx in zip(values, indices)
        ]

    results = {
        "predicted_char": LABEL_MAP[ensemble[0].argmax().item()],
        "confidence":     f"{ensemble[0].max().item()*100:.1f}%",
        "ensemble":       top_k_results(ensemble, top_k),
        "model1":         top_k_results(p1, top_k),
        "model2":         top_k_results(p2, top_k),
        "model3":         top_k_results(p3, top_k),
    }
    return results


# =============================================================================
# CLI
# =============================================================================

def print_results(results: dict, image_path: str):
    print(f"\n{'='*50}")
    print(f"  Image: {Path(image_path).name}")
    print(f"{'='*50}")
    print(f"  PREDICTION : '{results['predicted_char']}'")
    print(f"  CONFIDENCE : {results['confidence']}")
    print(f"{'='*50}")

    print(f"\n  Ensemble (weighted M1×0.38 + M2×0.38 + M3×0.24):")
    for i, r in enumerate(results["ensemble"], 1):
        print(f"    {i}. '{r['char']}'  {r['confidence']}")

    print(f"\n  Model 1 (Narrow ResNet 88.06%):")
    for i, r in enumerate(results["model1"], 1):
        print(f"    {i}. '{r['char']}'  {r['confidence']}")

    print(f"\n  Model 2 (Wide+SE 88.06%):")
    for i, r in enumerate(results["model2"], 1):
        print(f"    {i}. '{r['char']}'  {r['confidence']}")

    print(f"\n  Model 3 (Triple+MultiScale 87.74%):")
    for i, r in enumerate(results["model3"], 1):
        print(f"    {i}. '{r['char']}'  {r['confidence']}")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="EMNIST OCR — predict handwritten character from image"
    )
    parser.add_argument("image", help="Path to image file (PNG, JPG, BMP)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top predictions to show (default: 5)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (default: auto-detect GPU)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Verify checkpoints exist
    for path, name in [(MODEL1_PATH, "Model 1"), (MODEL2_PATH, "Model 2"),
                       (MODEL3_PATH, "Model 3")]:
        if not path.exists():
            print(f"[Error] {name} checkpoint not found: {path}")
            print(f"        Run the training script first.")
            sys.exit(1)

    model1, model2, model3 = load_models(device)
    results = predict(args.image, model1, model2, model3, device, args.top)
    print_results(results, args.image)


if __name__ == "__main__":
    main()
