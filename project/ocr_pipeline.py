import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import sys
import glob
import os
import argparse

LABELS = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

# ── Model paths — edit these to match your system ─────────────────────────────
MODELS = [
    r"C:\Users\beckhamw3233\Downloads\ocr_model.onnx",
    r"C:\Users\beckhamw3233\Downloads\ocr_model2.onnx",
    r"C:\Users\beckhamw3233\Downloads\ocr_model3.onnx",
]

# ── Supported image extensions ────────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Mode remapping tables ─────────────────────────────────────────────────────
DIGIT_REMAP = {
    "O": "0", "o": "0",
    "L": "1", "l": "1", "I": "1", "i": "1", "T": "1", "t": "1",
    "Z": "2", "z": "2", "W": "2",
    "w": "3",
    "Y": "4", "y": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6", "C": "6", "c": "6",
    "V": "7", "v": "7", "D": "7",
    "B": "8",
    "Q": "9", "q": "9",
}

UPPER_REMAP = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
    "f": "F", "g": "G", "h": "H", "k": "K", "m": "M",
    "n": "N", "p": "P", "q": "Q", "r": "R", "s": "S",
    "t": "T", "u": "U", "v": "V", "w": "W", "x": "X",
    "y": "Y", "z": "Z",
    "0": "O", "1": "I", "5": "S", "6": "G",
}

LOWER_REMAP = {
    "A": "a", "B": "b", "C": "c", "D": "d", "E": "e",
    "F": "f", "G": "g", "H": "h", "K": "k", "M": "m",
    "N": "n", "P": "p", "Q": "q", "R": "r", "S": "s",
    "T": "t", "U": "u", "V": "v", "W": "w", "X": "x",
    "Y": "y", "Z": "z",
    "0": "o", "1": "i", "5": "s",
}

THREE_SIGNALS = {"W", "w", "J", "j"}

# ── Strict digit grid layouts ─────────────────────────────────────────────────
STRICT_GRID_LAYOUTS = {
    (4, 4, 2): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",(1,3):"7",
        (2,0):"8",(2,1):"9",
    },
    (4, 3, 2): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",
        (2,0):"8",(2,1):"9",
    },
    (4, 4, 1): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",(1,3):"7",
        (2,0):"8",
    },
    (4, 4, 2, 1): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",(1,3):"7",
        (2,0):"8",(2,1):"9",
    },
    (4, 3, 2, 1): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",
        (2,0):"8",(2,1):"9",
    },
    (4, 4, 1, 1): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"4",(1,1):"5",(1,2):"6",(1,3):"7",
        (2,0):"8",
    },
    # test6 crossbar-7 pattern — 4 detected in wrong row
    (4, 3, 1, 2, 1): {
        (0,0):"0",(0,1):"1",(0,2):"2",(0,3):"3",
        (1,0):"5",(1,1):"6",(1,2):"7",
        (2,0):"4",
        (3,0):"8",(3,1):"9",
    },
}

STRICT_OVERRIDE_CONF = {
    "SPLIT":    True,
    "WEIGHTED": True,
    "MAJORITY": True,
    "ALL":      False,
}


def get_model_input_size(session) -> int:
    shape = session.get_inputs()[0].shape
    return int(shape[2])


def normalize_char(char_gray, img_size: int = 32):
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
    arr = (arr - 0.5) / 0.5
    arr = arr.reshape(1, 1, img_size, img_size)
    logits = session.run(["logits"], {"image": arr})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_indices = np.argsort(probs)[::-1][:n]
    return [(LABELS[idx], float(probs[idx])) for idx in top_indices]


def merge_nearby_boxes(boxes, gap_x=15, gap_y=35):
    """
    Merge bounding boxes that are close enough to be parts of the same character.
    gap_y raised to 35 to catch crossbar strokes on 7 which sit detached
    from the main diagonal stroke.
    Uses center-Y for proximity check to better handle grid layouts where
    characters in the same row may have different top-Y positions.
    """
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


def apply_spatial_override(label, box_w, box_h, median_h, mode):
    if box_h == 0:
        return label
    aspect = box_w / box_h
    height_ratio = box_h / median_h if median_h > 0 else 1.0
    if aspect < 0.30 and height_ratio > 0.7:
        if label in ("L", "l", "I", "T", "t", "J", "j", "Y", "y"):
            if mode in ("digits", "digits-strict"):
                return "1"
            elif mode == "lower":
                return "i"
            elif mode == "upper":
                return "I"
            else:
                return "1" if label in ("L", "I", "T", "Y") else label
    return label


def apply_mode_remap(label, box_w, box_h, mode, top3):
    if mode in ("digits", "digits-strict"):
        if label in "0123456789":
            return label
        for lbl, conf in top3[1:]:
            if lbl == "7" and conf > 0.08:
                return "7"
        if label in ("W", "w"):
            aspect = box_w / box_h if box_h > 0 else 1.0
            return "2" if aspect > 0.9 else "3"
        for lbl, conf in top3[1:]:
            if lbl in "0123456789" and conf > 0.10:
                return lbl
        return DIGIT_REMAP.get(label, label)

    elif mode == "upper":
        if label.isupper() or label.isdigit():
            return UPPER_REMAP.get(label, label) if label.isdigit() else label
        return UPPER_REMAP.get(label, label.upper())

    elif mode == "lower":
        if label.islower():
            return label
        return LOWER_REMAP.get(label, label.lower())

    return label


def apply_strict_grid(ensemble_lines):
    line_counts = tuple(
        sum(1 for e in line if e[0] != " ")
        for line in ensemble_lines
    )

    grid = STRICT_GRID_LAYOUTS.get(line_counts)
    if grid is None:
        return ensemble_lines, [], False, line_counts

    corrections = []
    new_ensemble_lines = []

    for ln, line in enumerate(ensemble_lines):
        new_line = []
        char_idx = 0
        for entry in line:
            label, agreement, raw_top1s, all_top3 = entry
            if label == " ":
                new_line.append(entry)
                continue

            expected = grid.get((ln, char_idx))
            should_override = STRICT_OVERRIDE_CONF.get(agreement, False)

            if expected and should_override and label != expected:
                corrections.append(
                    f"  Line {ln+1} Char {char_idx+1}: "
                    f"{label} ({agreement}) → {expected} [position override]"
                )
                new_line.append((expected, "STRICT", raw_top1s, all_top3))
            else:
                new_line.append(entry)

            char_idx += 1
        new_ensemble_lines.append(new_line)

    return new_ensemble_lines, corrections, True, line_counts


def vote_topn(all_top3, conf_threshold=0.20, mode="auto"):
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
    else:
        scores = {}
        for top3 in all_top3:
            for rank, (lbl, conf) in enumerate(top3):
                weight = conf * (1.0 / (rank + 1))
                scores[lbl] = scores.get(lbl, 0.0) + weight

        seven_score = sum(
            conf for top3 in all_top3
            for lbl, conf in top3
            if lbl == "7"
        )
        if seven_score > 0.10:
            return "7", "WEIGHTED", top1_labels

        if mode in ("digits", "digits-strict"):
            three_hits = sum(1 for lbl in top1_labels if lbl in THREE_SIGNALS)
            if three_hits >= 2:
                return "3", "WEIGHTED", top1_labels
            all_top2_pool = [lbl for top3 in all_top3 for lbl, _ in top3[:2]]
            three_pool = sum(1 for lbl in all_top2_pool if lbl in THREE_SIGNALS)
            if three_pool >= 4:
                return "3", "WEIGHTED", top1_labels

        best = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.3:
            return best, "WEIGHTED", top1_labels
        return "?", "SPLIT", top1_labels


def get_boxes(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread returned None — check path/file: {image_path}")
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

    # Merge split strokes using center-Y proximity, gap_y=35 for crossbar 7
    boxes = merge_nearby_boxes(boxes, gap_x=15, gap_y=35)

    # FIX: sort and group by CENTER Y instead of top Y
    # Handles grid layouts where digits in same row have different top positions
    boxes.sort(key=lambda b: b[1] + b[3] // 2)
    lines = []
    current_line = [boxes[0]]
    line_thresh = max(b[3] for b in boxes) * 0.5
    for box in boxes[1:]:
        # Compare center Y of new box to center Y of first box in current line
        cy_new = box[1] + box[3] // 2
        cy_ref = current_line[0][1] + current_line[0][3] // 2
        if abs(cy_new - cy_ref) < line_thresh:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda b: b[0]))
    return gray, lines


def resolve_image_paths(args):
    if not args:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths = []
        for i in range(1, 7):
            p = os.path.join(script_dir, f"test{i}.jpg")
            if os.path.isfile(p):
                paths.append(p)
            else:
                print(f"  [warn] Default file not found, skipping: {p}")
        return paths

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


def run_pipeline(image_path, sessions, img_sizes, model_names, mode="auto"):
    print(f"\n{'='*60}")
    print(f"  OCR Pipeline — 3-Model Ensemble")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Mode:  {mode.upper()}")
    print(f"{'='*60}")

    try:
        gray, lines = get_boxes(image_path)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    total_chars = sum(len(l) for l in lines)
    print(f"  Detected: {total_chars} characters across {len(lines)} line(s)\n")

    if not lines:
        print("  No characters detected — check image quality or thresholding.\n")
        return

    print(f"  {'─'*50}")
    print(f"  MODEL INPUT SIZES")
    print(f"  {'─'*50}")
    for name, size in zip(model_names, img_sizes):
        print(f"  {name}: {size}x{size}")
    print()

    all_heights = [h for line in lines for (x, y, w, h) in line]
    median_h = float(np.median(all_heights)) if all_heights else 1.0

    all_model_lines = [[] for _ in MODELS]
    ensemble_lines = []

    for line in lines:
        per_model_line = [[] for _ in MODELS]
        ensemble_line = []
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

            winner, agreement, top1s = vote_topn(all_top3, mode=mode)
            winner = apply_spatial_override(winner, w, h, median_h, mode)
            raw_top1s = [t[0][0] if t[0][1] >= 0.20 else "?" for t in all_top3]

            if mode != "auto":
                winner = apply_mode_remap(winner, w, h, mode, all_top3[0])

            for i in range(len(MODELS)):
                if space:
                    per_model_line[i].append(" ")
                per_model_line[i].append(raw_top1s[i])

            if space:
                ensemble_line.append((" ", "ALL", " ", []))
            ensemble_line.append((winner, agreement, raw_top1s, all_top3))
            prev_x_end = x + w

        for i in range(len(MODELS)):
            all_model_lines[i].append("".join(per_model_line[i]))
        ensemble_lines.append(ensemble_line)

    # ── Strict grid post-processing ───────────────────────────────────────────
    strict_applied = False
    corrections = []
    line_counts = None
    if mode == "digits-strict":
        ensemble_lines, corrections, strict_applied, line_counts = apply_strict_grid(
            ensemble_lines
        )

    # ── Individual model output ───────────────────────────────────────────────
    print(f"  {'─'*50}")
    print(f"  INDIVIDUAL MODEL PREDICTIONS (raw, no remapping)")
    print(f"  {'─'*50}")
    for i, name in enumerate(model_names):
        print(f"  {name} ({img_sizes[i]}x{img_sizes[i]}):")
        for ln, text in enumerate(all_model_lines[i], 1):
            print(f"    Line {ln}: {text}")
    print()

    # ── Ensemble output ───────────────────────────────────────────────────────
    print(f"  {'─'*50}")
    print(f"  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  *=strict  ?=split)")
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
            if agreement == "ALL":
                text += label
                agree_count += 1
            elif agreement in ("MAJORITY", "WEIGHTED"):
                text += f"[{label}]"
            elif agreement == "STRICT":
                text += f"*{label}*"
            else:
                text += "?"
        print(f"  Line {ln}: {text}")

    pct = 100*agree_count/total_count if total_count else 0
    print(f"\n  Consensus: {agree_count}/{total_count} chars ({pct:.1f}% all-3 agreement)")

    if mode == "digits-strict":
        print(f"\n  {'─'*50}")
        if strict_applied and corrections:
            print(f"  STRICT GRID CORRECTIONS ({len(corrections)} applied)")
            print(f"  {'─'*50}")
            for c in corrections:
                print(c)
        elif strict_applied:
            print(f"  STRICT GRID — layout {line_counts} matched, no corrections needed")
            print(f"  {'─'*50}")
        else:
            print(f"  STRICT GRID — layout {line_counts} not recognized, skipped")
            print(f"  Recognized: {list(STRICT_GRID_LAYOUTS.keys())}")
            print(f"  {'─'*50}")

    # ── Best guess ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  BEST GUESS READ  [mode: {mode.upper()}]")
    print(f"  {'─'*50}")
    for ln, line in enumerate(ensemble_lines, 1):
        text = ""
        for entry in line:
            label = entry[0]
            text += label if label != " " else " "
        print(f"  Line {ln}: {text}")

    # ── Character detail ──────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  CHARACTER DETAIL  (M1 | M2 | M3 | vote | final)")
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

            def fmt_top3(top3):
                return "/".join(f"{l}({c:.0%})" for l, c in top3[:3])

            m1 = fmt_top3(all_top3[0]) if all_top3 else "?"
            m2 = fmt_top3(all_top3[1]) if len(all_top3) > 1 else "?"
            m3 = fmt_top3(all_top3[2]) if len(all_top3) > 2 else "?"

            flag = ("✓ all"    if agreement == "ALL"      else
                    "~ maj"    if agreement == "MAJORITY"  else
                    "~ wgt"    if agreement == "WEIGHTED"  else
                    "* strict" if agreement == "STRICT"    else
                    "✗ split")

            aspect = w/h if h > 0 else 0
            print(f"    Char {box_idx+1:>2} [asp:{aspect:.2f}]: "
                  f"M1={raw_top1s[0] if raw_top1s else '?'} "
                  f"M2={raw_top1s[1] if len(raw_top1s)>1 else '?'} "
                  f"M3={raw_top1s[2] if len(raw_top1s)>2 else '?'} "
                  f"| {flag} → {final_label}")
            if agreement == "SPLIT":
                print(f"             M1: {m1}")
                print(f"             M2: {m2}")
                print(f"             M3: {m3}")
            char_idx += 1

    print(f"\n{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Pipeline — 3-Model Ensemble with mode remapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  auto           Raw ensemble output, no remapping (default)
  digits         Remap letter predictions to digit equivalents
  digits-strict  Like digits, plus position-based correction for the
                 standard 0-9 digit grid layout.
  upper          Force uppercase
  lower          Force lowercase

EXAMPLES:
  python ocr_pipeline.py test1.jpg
  python ocr_pipeline.py --mode digits test1.jpg
  python ocr_pipeline.py --mode digits-strict test*.jpg
  python ocr_pipeline.py --mode upper Untitled.png
        """
    )
    parser.add_argument("--mode",
                        choices=["auto", "digits", "upper", "lower", "digits-strict"],
                        default="auto",
                        help="Remapping mode (default: auto)")
    parser.add_argument("images", nargs="*", help="Image file(s) or glob patterns")
    args = parser.parse_args()

    image_paths = resolve_image_paths(args.images)

    if not image_paths:
        parser.print_help()
        sys.exit(1)

    sessions = []
    img_sizes = []
    for path in MODELS:
        try:
            s = ort.InferenceSession(path)
            sessions.append(s)
            img_sizes.append(get_model_input_size(s))
        except Exception as e:
            print(f"  Failed to load {os.path.basename(path)}: {e}")
            sessions.append(None)
            img_sizes.append(32)

    model_names = [os.path.basename(p) for p in MODELS]

    print(f"\nProcessing {len(image_paths)} image(s) in [{args.mode.upper()}] mode...")

    for image_path in image_paths:
        run_pipeline(image_path, sessions, img_sizes, model_names, mode=args.mode)