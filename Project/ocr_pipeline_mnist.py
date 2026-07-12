"""
ocr_pipeline_mnist.py
=====================
MNIST OCR Pipeline — Single model, ensemble, or any combination of ONNX models.
No post-processing. No remapping. Raw inference only.

Usage:
  Single model:
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx image.jpg

  Multiple models (ensemble vote):
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx adamw_64/v1_adamw_64_64.onnx image.jpg

  Scan a directory for all .onnx files (recursive):
    python ocr_pipeline_mnist.py --model-dir E:\\CSC-114\\project digit.jpg

  Glob patterns for images:
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx test*.jpg

  Multiple images:
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx a.jpg b.jpg c.jpg

  --models and --model-dir are mutually exclusive — use one or the other.

Requires: pip install onnxruntime-gpu opencv-python numpy
"""

import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import sys
import glob
import os
import argparse
import datetime
import re


# ── Logging ──────────────────────────────────────────────────────────────────

class _Tee:
    """Mirrors stdout to a timestamped log file beside this script."""
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "pipeline_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"ocr_mnist_{ts}.log")
        self._log = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        print(f"  Log: {log_path}")

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()


# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = [str(i) for i in range(10)]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# If the best confidence across ALL models for a character is below this threshold,
# the character is flagged as a likely non-digit rather than silently output as a
# low-confidence digit guess. Tune this value based on real-world testing.
NON_DIGIT_CONF_FLOOR = 0.40


# ── Image utilities ───────────────────────────────────────────────────────────

def get_model_input_size(session) -> int:
    return int(session.get_inputs()[0].shape[2])


def short_model_name(path: str) -> str:
    """Extract a readable short name from the ONNX filename.

    v1_lion_64_64.onnx        -> lion_64
    v1_adamw_128_128.onnx     -> adamw_128
    v1_sgd_64_64.onnx         -> sgd_64
    adahessian_64.onnx        -> adahessian_64
    soap_128.onnx             -> soap_128
    anything_else.onnx        -> filename stem (no extension)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # v1_{optimizer}_{res}_{res} pattern
    m = re.match(r"v1_([a-z0-9]+)_(\d+)_\d+$", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # {optimizer}_{res} pattern (adahessian_64, soap_128, etc.)
    m = re.match(r"([a-z0-9]+)_(\d+)$", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return stem


def normalize_char(char_gray, img_size: int):
    _, binary = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]
    size = max(w, h)
    pad = int(size * 0.2)
    canvas = np.zeros((size + pad*2, size + pad*2), dtype=np.uint8)
    x_off = pad + (size - w) // 2
    y_off = pad + (size - h) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = cropped
    kernel_size = max(2, size // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    canvas = cv2.dilate(canvas, kernel, iterations=2)
    return cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)


def predict_char_topn(session, char_gray, img_size: int, n: int = 3):
    normalized = normalize_char(char_gray, img_size)
    if normalized is None:
        return [("?", 0.0)] * n
    arr = normalized.astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, img_size, img_size)
    logits = session.run(["logits"], {"image": arr})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_indices = np.argsort(probs)[::-1][:n]
    return [(LABELS[idx], float(probs[idx])) for idx in top_indices]


# ── Box detection ─────────────────────────────────────────────────────────────

def merge_nearby_boxes(boxes, gap_x=15, gap_y=35):
    if not boxes:
        return boxes
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            ax, ay, aw, ah = boxes[i]
            acy = ay + ah // 2
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx, by, bw, bh = boxes[j]
                bcy = by + bh // 2
                x_close = ax < bx + bw + gap_x and bx < ax + aw + gap_x
                y_close = abs(acy - bcy) < gap_y
                if x_close and y_close:
                    nx = min(ax, bx)
                    ny = min(ay, by)
                    nw = max(ax + aw, bx + bw) - nx
                    nh = max(ay + ah, by + bh) - ny
                    boxes[i] = (nx, ny, nw, nh)
                    ax, ay, aw, ah = nx, ny, nw, nh
                    acy = ay + ah // 2
                    used[j] = True
                    merged = True
            new_boxes.append(boxes[i])
            used[i] = True
        boxes = new_boxes
    return boxes


def get_boxes(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread returned None: {image_path}")
    h_img, w_img = img.shape[:2]
    scale = min(1000 / w_img, 1000 / h_img, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if (h_img*0.03 < h < h_img*0.35 and
            w_img*0.01 < w < w_img*0.25 and
            0.1 < aspect < 10.0):
            boxes.append((x, y, w, h))

    if not boxes:
        return gray, []

    boxes = merge_nearby_boxes(boxes, gap_x=15, gap_y=35)
    boxes.sort(key=lambda b: b[1] + b[3] // 2)
    lines = []
    current_line = [boxes[0]]
    line_thresh = max(b[3] for b in boxes) * 0.5
    for box in boxes[1:]:
        cy_new = box[1] + box[3] // 2
        cy_ref = int(sum(b[1] + b[3] // 2 for b in current_line) / len(current_line))
        if abs(cy_new - cy_ref) < line_thresh:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda b: b[0]))
    return gray, lines


# ── Voting ────────────────────────────────────────────────────────────────────

def vote_topn(all_top3, conf_threshold=0.20):
    """Majority vote across models. Returns (label, agreement, top1_list)."""
    top1_labels = [t[0][0] for t in all_top3 if t[0][1] >= conf_threshold]
    if not top1_labels:
        top1_labels = [t[0][0] for t in all_top3]

    count = Counter(top1_labels)
    if not count:
        return "?", "SPLIT", top1_labels

    top_label, top_count = count.most_common(1)[0]
    n = len(all_top3)

    if top_count == n:
        return top_label, "ALL", top1_labels
    elif top_count > 1:
        return top_label, "MAJORITY", top1_labels

    # Weighted confidence tiebreak
    scores = {}
    for top3 in all_top3:
        for rank, (lbl, conf) in enumerate(top3):
            weight = conf * (1.0 / (rank + 1))
            scores[lbl] = scores.get(lbl, 0.0) + weight

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.3:
        return max(scores, key=scores.get), "WEIGHTED", top1_labels

    # Single-model dominant confidence rescue
    all_top1_confs = [(t[0][0], t[0][1]) for t in all_top3]
    best_conf_label, best_conf = max(all_top1_confs, key=lambda x: x[1])
    if best_conf > 0.22:
        other_confs = [c for l, c in all_top1_confs if l != best_conf_label]
        if not other_confs or best_conf > max(other_confs) * 1.2:
            return best_conf_label, "WEIGHTED", top1_labels

    return "?", "SPLIT", top1_labels


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_image_paths(args):
    if not args:
        return []
    paths = []
    for arg in args:
        expanded = glob.glob(arg)
        if expanded:
            paths.extend(sorted(expanded))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"  [warn] Not found, skipping: {arg}")
    filtered = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in SUPPORTED_EXTS:
            filtered.append(p)
        else:
            print(f"  [warn] Unsupported extension '{ext}', skipping: {p}")
    return filtered


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(image_path, sessions, img_sizes, model_names, ground_truth=None):
    n_models = len(sessions)
    ensemble_label = f"{n_models}-Model Ensemble" if n_models > 1 else "Single Model"

    print(f"\n{'='*60}")
    print(f"  MNIST OCR Pipeline — {ensemble_label}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    try:
        gray, lines = get_boxes(image_path)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    total_chars = sum(len(l) for l in lines)
    print(f"  Detected: {total_chars} characters across {len(lines)} line(s)\n")

    if not lines:
        print("  No characters detected.\n")
        return

    # Run inference
    all_model_lines = [[] for _ in range(n_models)]
    ensemble_lines  = []

    for line in lines:
        per_model_line = [[] for _ in range(n_models)]
        ensemble_line  = []
        prev_x_end = None

        for (x, y, w, h) in line:
            space = prev_x_end is not None and (x - prev_x_end) > w * 1.2

            all_top3 = []
            for i, session in enumerate(sessions):
                if session is None:
                    all_top3.append([("?", 0.0), ("?", 0.0), ("?", 0.0)])
                else:
                    top3 = predict_char_topn(session, gray[y:y+h, x:x+w], img_sizes[i], n=3)
                    all_top3.append(top3)

            raw_top1s = [t[0][0] if t[0][1] >= 0.20 else "?" for t in all_top3]

            # Non-digit detection — if best top-1 confidence across all models
            # is below the floor, flag as likely non-digit regardless of vote
            best_conf_any = max(top3[0][1] for top3 in all_top3)
            non_digit = best_conf_any < NON_DIGIT_CONF_FLOOR

            if non_digit:
                winner    = "NON-DIGIT?"
                agreement = "NON-DIGIT"
            elif n_models == 1:
                winner    = all_top3[0][0][0]
                agreement = "SINGLE"
            else:
                winner, agreement, _ = vote_topn(all_top3)

            for i in range(n_models):
                if space:
                    per_model_line[i].append(" ")
                per_model_line[i].append(raw_top1s[i])

            if space:
                ensemble_line.append((" ", "ALL", [" "] * n_models, []))
            ensemble_line.append((winner, agreement, raw_top1s, all_top3))
            prev_x_end = x + w

        for i in range(n_models):
            all_model_lines[i].append("".join(per_model_line[i]))
        ensemble_lines.append(ensemble_line)

    # Individual model output (only shown if >1 model)
    if n_models > 1:
        print(f"  {'─'*50}")
        print(f"  INDIVIDUAL MODEL PREDICTIONS")
        print(f"  {'─'*50}")
        for i, name in enumerate(model_names):
            print(f"  {name}  [{img_sizes[i]}x{img_sizes[i]}]:")
            for ln, text in enumerate(all_model_lines[i], 1):
                print(f"    Line {ln}: {text}")
        print()

    # Result
    print(f"  {'─'*50}")
    if n_models > 1:
        print(f"  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  ?=split)")
    else:
        print(f"  RESULT")
    print(f"  {'─'*50}")

    agree_count = 0
    total_count = 0
    for ln, line in enumerate(ensemble_lines, 1):
        text = ""
        for entry in line:
            label, agreement = entry[0], entry[1]
            if label == " ":
                text += " "
                continue
            total_count += 1
            if agreement == "NON-DIGIT":
                text += "[NON-DIGIT?]"
            elif agreement in ("ALL", "SINGLE"):
                text += label
                agree_count += 1
            elif agreement in ("MAJORITY", "WEIGHTED"):
                text += f"[{label}]"
            else:
                text += "?"
        print(f"  Line {ln}: {text}")

    if n_models > 1:
        pct = 100 * agree_count / total_count if total_count else 0
        print(f"\n  Consensus: {agree_count}/{total_count} chars ({pct:.1f}% full agreement)")

    # Character detail — full top-3 per model per character
    print(f"\n  {'─'*50}")
    print(f"  CHARACTER DETAIL — full top-3 per model")
    print(f"  {'─'*50}")

    for ln, (line_boxes, ens_line) in enumerate(zip(lines, ensemble_lines), 1):
        print(f"\n  Line {ln}:")
        char_idx = 0
        for box_idx, (x, y, w, h) in enumerate(line_boxes):
            while char_idx < len(ens_line) and ens_line[char_idx][0] == " ":
                char_idx += 1
            if char_idx >= len(ens_line):
                break
            entry = ens_line[char_idx]
            final_label, agreement, raw_top1s, all_top3 = entry

            flag = ("✓ all"      if agreement == "ALL"       else
                    "single"     if agreement == "SINGLE"     else
                    "~ maj"      if agreement == "MAJORITY"   else
                    "~ wgt"      if agreement == "WEIGHTED"   else
                    "! NON-DIG"  if agreement == "NON-DIGIT"  else
                    "✗ split")

            aspect = w / h if h > 0 else 0
            nd_warn = "  *** LIKELY NON-DIGIT — check image ***" if agreement == "NON-DIGIT" else ""
            print(f"\n    Char {box_idx+1:>2}  [w:{w} h:{h} asp:{aspect:.2f}]  vote={flag}  final={final_label}{nd_warn}")
            for mi, (mname, top3) in enumerate(zip(model_names, all_top3)):
                top1_lbl, top1_conf = top3[0]
                top2_lbl, top2_conf = top3[1] if len(top3) > 1 else ("?", 0.0)
                top3_lbl, top3_conf = top3[2] if len(top3) > 2 else ("?", 0.0)
                marker = " <-- SPLIT" if agreement == "SPLIT" and top1_lbl != final_label else ""
                print(
                    f"      [{mi+1:>2}] {mname:<20s}  "
                    f"#1:{top1_lbl}({top1_conf:5.1%})  "
                    f"#2:{top2_lbl}({top2_conf:5.1%})  "
                    f"#3:{top3_lbl}({top3_conf:5.1%})"
                    f"{marker}"
                )

            char_idx += 1

    # Per-model summary — individual read per model across all lines
    print(f"\n  {'─'*50}")
    print(f"  PER-MODEL SUMMARY")
    print(f"  {'─'*50}")
    for i, name in enumerate(model_names):
        read = "".join(
            " "           if entry[0] == " "          else
            "[ND?]"       if entry[1] == "NON-DIGIT"  else
            entry[2][i]
            for line in ensemble_lines
            for entry in line
        )
        print(f"  [{i+1:>2}] {name:<20s}  read: {read}")

    if ground_truth is not None:
        gt = ground_truth.strip()
        print(f"\n  {'─'*50}")
        print(f"  PER-MODEL ACCURACY  (ground truth: {gt})")
        print(f"  {'─'*50}")
        for i, name in enumerate(model_names):
            pred_chars = [
                entry[2][i]
                for line in ensemble_lines
                for entry in line
                if entry[0] != " " and entry[1] != "NON-DIGIT"
            ]
            nd_count = sum(
                1 for line in ensemble_lines
                for entry in line
                if entry[1] == "NON-DIGIT"
            )
            n_gt  = len(gt)
            n_cmp = min(len(pred_chars), n_gt)
            correct = sum(1 for p, g in zip(pred_chars, gt) if p == g)
            pct = 100 * correct / n_gt if n_gt else 0.0
            nd_note = f"  [{nd_count} non-digit skipped]" if nd_count else ""
            print(f"  [{i+1:>2}] {name:<20s}  {correct:>3}/{n_gt}  ({pct:5.1f}%)"
                  f"  pred: {''.join(pred_chars)}{nd_note}")

    # Accuracy scoring
    if ground_truth is not None:
        gt = ground_truth.strip()
        final_chars = [
            entry[0]
            for line in ensemble_lines
            for entry in line
            if entry[0] != " " and entry[1] != "NON-DIGIT"
        ]
        nd_total = sum(
            1 for line in ensemble_lines
            for entry in line
            if entry[1] == "NON-DIGIT"
        )
        n_pred = len(final_chars)
        n_gt   = len(gt)
        correct = sum(1 for p, g in zip(final_chars, gt) if p == g)
        pct_acc = 100 * correct / n_gt if n_gt else 0.0

        print(f"  {'─'*50}")
        print(f"  ACCURACY  (ground truth: {gt})")
        print(f"  {'─'*50}")
        print(f"  Predicted chars : {n_pred}")
        print(f"  Ground truth    : {n_gt}")
        print(f"  Correct         : {correct} / {n_gt}  ({pct_acc:.1f}%)")
        if nd_total:
            print(f"  [warn] {nd_total} character(s) flagged as non-digit — excluded from accuracy count")
        if n_pred != n_gt:
            print(f"  [warn] Predicted {n_pred} digit chars but ground truth has {n_gt} — count mismatch")

    print(f"\n{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tee = _Tee()

    parser = argparse.ArgumentParser(
        description="MNIST OCR Pipeline — single model, ensemble, or any combination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Single model:
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx digit.jpg

  Two-model ensemble:
    python ocr_pipeline_mnist.py --models lion_64/v1_lion_64_64.onnx adamw_64/v1_adamw_64_64.onnx digit.jpg

  Glob model paths:
    python ocr_pipeline_mnist.py --models E:\\CSC-114\\project\\lion_64\\*.onnx digit.jpg

  Scan a directory for all .onnx files (recursive):
    python ocr_pipeline_mnist.py --model-dir E:\\CSC-114\\project digit.jpg

  Scan directory, multiple test images:
    python ocr_pipeline_mnist.py --model-dir E:\\CSC-114\\project test*.jpg

  Scan directory, single image:
    python ocr_pipeline_mnist.py --model-dir E:\\CSC-114\\project\\lion_64 digit.jpg
        """
    )
    model_src = parser.add_mutually_exclusive_group(required=True)
    model_src.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more ONNX model paths (space-separated, supports globs)"
    )
    model_src.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Directory to scan recursively for all .onnx files"
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image file(s) or glob patterns"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="STRING",
        help=(
            "Known correct answer for accuracy scoring. "
            "Digits only, no spaces (e.g. --ground-truth 5038). "
            "Compared against final read character by character."
        )
    )
    args = parser.parse_args()

    # Resolve model paths
    # Directory names to never descend into when scanning --model-dir.
    # Prevents sweeping up venv/site-packages test fixtures (onnx ships
    # hundreds of tiny unit-test .onnx files, many literally named
    # "model.onnx") and any unrelated model-zoo files that happen to live
    # under the project root.
    EXCLUDED_DIRS = {"venv", ".venv", "env", "site-packages", "__pycache__",
                     ".git", "node_modules"}
    model_paths = []
    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            print(f"  ERROR: --model-dir not found: {args.model_dir}")
            sys.exit(1)
        skipped_dirs = []
        for root, dirs, files in os.walk(args.model_dir):
            before = set(dirs)
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            skipped = before - set(dirs)
            if skipped:
                skipped_dirs.extend(os.path.join(root, d) for d in skipped)
            dirs.sort()
            for fname in sorted(files):
                if fname.lower().endswith(".onnx"):
                    model_paths.append(os.path.join(root, fname))
        if skipped_dirs:
            print(f"  Skipped {len(skipped_dirs)} excluded folder(s) "
                  f"(venv/site-packages/etc.) during scan:")
            for d in skipped_dirs:
                print(f"    - {d}")
        if not model_paths:
            print(f"  ERROR: No .onnx files found in: {args.model_dir}")
            sys.exit(1)
        print(f"  Scanned {args.model_dir} — found {len(model_paths)} .onnx file(s)")
    else:
        for m in args.models:
            expanded = glob.glob(m)
            if expanded:
                model_paths.extend(sorted(expanded))
            elif os.path.isfile(m):
                model_paths.append(m)
            else:
                print(f"  [warn] Model not found, skipping: {m}")

    if not model_paths:
        print("  ERROR: No valid model paths provided.")
        sys.exit(1)

    image_paths = resolve_image_paths(args.images)
    if not image_paths:
        parser.print_help()
        sys.exit(1)

    # Load models
    sessions    = []
    img_sizes   = []
    model_names = []
    print(f"\nLoading {len(model_paths)} model(s)...")
    for path in model_paths:
        try:
            s = ort.InferenceSession(
                path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            sessions.append(s)
            img_sizes.append(get_model_input_size(s))
            model_names.append(short_model_name(path))
            print(f"  Loaded: {short_model_name(path):20s}  [{img_sizes[-1]}x{img_sizes[-1]}]  ({os.path.basename(path)})")
        except Exception as e:
            print(f"  Failed: {os.path.basename(path)}: {e}")

    if not sessions:
        print("  ERROR: No models loaded successfully.")
        sys.exit(1)

    # Print model roster once — not repeated per image
    print(f"\n  {'─'*50}")
    print(f"  MODELS ({len(sessions)} loaded)")
    print(f"  {'─'*50}")
    for name, size in zip(model_names, img_sizes):
        print(f"  {name:20s}  [{size}x{size}]")
    print()

    print(f"Processing {len(image_paths)} image(s) with {len(sessions)} model(s)...\n")

    for image_path in image_paths:
        run_pipeline(image_path, sessions, img_sizes, model_names, ground_truth=args.ground_truth)

    tee.close()
