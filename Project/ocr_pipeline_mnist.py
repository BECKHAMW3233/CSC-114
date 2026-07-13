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

Output markers:
  ??            A character position was detected but the ensemble could not
                agree on a digit (models split with no majority/weighted
                winner). Distinct from [NON-DIGIT?], which means every model's
                top confidence was too low to trust any answer at all.
  [NON-DIGIT?]  Best confidence across all models fell below NON_DIGIT_CONF_FLOOR
                — likely not a digit at all, not just a hard-to-read one.

Note: get_boxes() has no concept of a grid — it clusters detected contours into
"lines" purely by proximity and vertical position, not fixed rows/columns. Real
handwritten digits are rarely uniform: they vary in size, spacing, and vertical
alignment even within the same line, and get_boxes() has no ground truth to
compare against. A digit that fails the width filter (e.g. a stray connecting
stroke fuses onto it, making its box wider than a normal character) is now
recovered by a rescue pass if doing so fills an unusually large gap in its
line — see get_boxes()'s own docstring for exactly what is and isn't covered.
This does not catch every case: a digit rejected for height or aspect ratio,
or one whose vertical center lands outside its line's clustering window, can
still be silently absent from the output with no ?? or other marker, because
no character-slot was ever created for it. ?? only covers characters that WERE
detected and handed to the models but couldn't be agreed on afterward.

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
    """Detect character bounding boxes and group them into lines.

    Returns (gray_image, lines), where lines is a list of rows and each row
    is a list of (x, y, w, h) boxes sorted left-to-right.

    This is proximity-based clustering, not a fixed grid: each contour that
    survives the height/width/aspect filter below becomes a candidate box,
    and boxes are grouped into a "line" purely by how close their vertical
    centers are to each other (line_thresh, computed from that line's own
    box heights). There is no assumption that digits are evenly spaced,
    uniformly sized, or aligned to rows/columns — real handwriting isn't.

    Rescue pass: a contour that fails ONLY the width ceiling (correct height
    and aspect ratio, just too wide — the usual cause is a stray connecting
    stroke or underline fusing onto an otherwise normal digit) is not
    discarded outright. Each line is checked for an unusually large gap
    relative to its own typical spacing — between two existing characters,
    or after the last one, or before the first one (a trailing/leading
    dropped digit, like "1 8 4" missing a final "5", produces no gap
    between existing boxes at all, so the last/first-box edges have to be
    checked too, not just the gaps strictly between pairs). If an oversized
    contour's position and height fit inside that gap, it's added back as a
    real character instead of silently dropped. This does not touch
    contours rejected for height, aspect ratio, or being more than 2x over
    the width ceiling (those stay rejected — they're far more likely to be
    scan noise or a genuine full-width artifact than a real digit).

    On the underlying cause (a stray stroke fusing onto a digit's contour,
    inflating its bounding box): this is a well-documented failure mode in
    OCR preprocessing literature, usually called "underline/rule-line
    removal." The standard published technique detects long thin strokes
    with a wide horizontal morphological opening (cv2.getStructuringElement
    with a kernel like (25, 1) applied via cv2.MORPH_OPEN) and erases them
    before contour extraction, rather than rescuing an oversized box after
    the fact. That approach was tested against this project's own images
    and NOT adopted here: the stray stroke on a real "5" in this dataset was
    only ~45px long (~8% of the working image width) — a kernel wide enough
    to reliably reject full-width scan artifacts elsewhere in the same image
    (which run ~560px, near the full width) was too wide to catch that short
    a stroke, and a kernel narrow enough to catch it started eroding real
    digit strokes elsewhere. The width-ceiling rescue approach implemented
    below sidesteps that tuning problem entirely by working in box-space
    after contour detection rather than pixel-space before it, at the cost
    of only fixing width-based rejections specifically (see the remaining
    failure modes below for what it still misses).

    Remaining known failure modes (still neither raise an error nor flag
    anything — the character in question is just absent from the returned
    lines):
      - A digit rejected for HEIGHT or ASPECT RATIO (not width) is never
        reconsidered by the rescue pass above.
      - A line with only one detected character has no internal gap to
        measure, so the rescue pass has nothing to compare against and is
        skipped for that line entirely.
      - A digit whose vertical center falls outside the current line's
        line_thresh window (e.g. it's written higher/lower than the rest of
        that row) can still be split into its own line or merged into the
        wrong one.
    Callers should still not assume the number of boxes returned equals the
    number of digits actually on the page — the rescue pass narrows one
    specific gap, it doesn't close all of them.
    """
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
    wide_rejects = []  # contours rejected ONLY for exceeding the width ceiling —
                        # these are the ones worth a second look, since a digit
                        # with a stray connecting stroke (see rescue pass below)
                        # is the most common real-world cause, not noise.
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        h_ok = h_img*0.03 < h < h_img*0.35
        aspect_ok = 0.1 < aspect < 10.0
        w_ok = w_img*0.01 < w < w_img*0.25
        if h_ok and w_ok and aspect_ok:
            boxes.append((x, y, w, h))
        elif h_ok and aspect_ok and w >= w_img*0.25:
            # Failed on width alone — height and aspect both look like a real
            # character. Cap how far over the ceiling we'll ever consider
            # (2x) so an actual full-width scan artifact (w ~= w_img) is
            # never rescued, only a moderately oversized single digit.
            if w < w_img*0.50:
                wide_rejects.append((x, y, w, h))

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

    # Rescue pass — a wide-reject contour whose position and height fit
    # plausibly into a line (right after the last box, right before the
    # first box, or in an unusually large gap between two boxes — relative
    # to that line's OWN normal spacing) is a strong sign a character was
    # dropped by the width filter above, not that the writer left a
    # deliberate blank space (deliberate spaces are handled separately
    # downstream, at print time, by comparing each gap against that
    # character's own width — this is a coarser, earlier check that runs
    # before any character has been classified).
    if len(lines) > 0 and wide_rejects:
        for line in lines:
            avg_h = sum(b[3] for b in line) / len(line)
            cy_ref = sum(b[1] + b[3] // 2 for b in line) / len(line)
            # Typical spacing for this line: median gap between existing
            # boxes if there are at least two, otherwise fall back to a
            # spacing estimate based on character height (roughly how wide
            # a MNIST-style digit tends to be relative to its own height).
            if len(line) >= 2:
                gaps = [line[i+1][0] - (line[i][0] + line[i][2]) for i in range(len(line) - 1)]
                median_gap = sorted(gaps)[len(gaps) // 2]
            else:
                median_gap = avg_h * 0.3
            gap_floor = max(median_gap * 3, avg_h * 0.8)

            def _try_rescue(zone_left, zone_right):
                for wr in wide_rejects:
                    wx, wy, ww, wh = wr
                    wcy = wy + wh // 2
                    if (zone_left - 10 <= wx and wx + ww <= zone_right + 10
                            and abs(wcy - cy_ref) < line_thresh
                            and abs(wh - avg_h) < avg_h):
                        return wr
                return None

            # Gaps strictly between two existing boxes.
            if len(line) >= 2:
                gaps = [line[i+1][0] - (line[i][0] + line[i][2]) for i in range(len(line) - 1)]
                for i, gap in enumerate(gaps):
                    if gap > gap_floor:
                        found = _try_rescue(line[i][0] + line[i][2], line[i+1][0])
                        if found:
                            line.append(found)
                            wide_rejects.remove(found)

            # After the last box in the line — the case a purely
            # between-boxes check can never catch (e.g. a dropped trailing
            # character, as when a line ends "1 8 4" but should be "1 8 4 5").
            last = max(line, key=lambda b: b[0])
            found = _try_rescue(last[0] + last[2], last[0] + last[2] + gap_floor + avg_h)
            if found:
                line.append(found)
                wide_rejects.remove(found)

            # Before the first box in the line, symmetric with the above.
            first = min(line, key=lambda b: b[0])
            found = _try_rescue(max(0, first[0] - gap_floor - avg_h), first[0])
            if found:
                line.append(found)
                wide_rejects.remove(found)
        lines = [sorted(line, key=lambda b: b[0]) for line in lines]
    return gray, lines


# ── Voting ────────────────────────────────────────────────────────────────────

def vote_topn(all_top3, conf_threshold=0.20):
    """Majority vote across models for one already-detected character box.

    Returns (label, agreement, top1_list).
    label is a digit '0'-'9', or '??' if the models genuinely split with no
    majority, no weighted winner, and no single dominant-confidence model —
    i.e. this character WAS detected and classified by every model, but the
    ensemble could not settle on one answer. agreement is 'ALL', 'MAJORITY',
    'WEIGHTED', or 'SPLIT' (SPLIT always pairs with the '??' label).
    """
    top1_labels = [t[0][0] for t in all_top3 if t[0][1] >= conf_threshold]
    if not top1_labels:
        top1_labels = [t[0][0] for t in all_top3]

    count = Counter(top1_labels)
    if not count:
        return "??", "SPLIT", top1_labels

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

    return "??", "SPLIT", top1_labels


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
    print(f"  Detected: {total_chars} characters across {len(lines)} line(s)")
    print(f"  (Detection is proximity-based, not a grid — an unusually large,")
    print(f"   small, or disconnected digit may not produce a box at all and")
    print(f"   would be missing from this count without any warning.)\n")

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
        print(f"  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  "
              f"??=models split, no answer  [NON-DIGIT?]=likely not a digit)")
    else:
        print(f"  RESULT")
    print(f"  {'─'*50}")

    agree_count = 0
    total_count = 0
    unknown_count = 0
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
                unknown_count += 1
            elif agreement in ("ALL", "SINGLE"):
                text += label
                agree_count += 1
            elif agreement in ("MAJORITY", "WEIGHTED"):
                text += f"[{label}]"
            else:
                text += "??"
                unknown_count += 1
        print(f"  Line {ln}: {text}")

    if n_models > 1:
        pct = 100 * agree_count / total_count if total_count else 0
        print(f"\n  Consensus: {agree_count}/{total_count} chars ({pct:.1f}% full agreement)")
    if unknown_count:
        print(f"  Unidentified: {unknown_count} character(s) marked ?? or [NON-DIGIT?] above "
              f"— see CHARACTER DETAIL below for per-model votes on each.")

    # Page layout — same characters as ENSEMBLE RESULT above, but positioned
    # left-to-right and top-to-bottom to roughly match their actual placement
    # on the page, instead of every line being flush-left with even spacing.
    img_h, img_w = gray.shape[:2]
    layout_width = 60  # character columns to map the page width onto
    print(f"\n  {'─'*50}")
    print(f"  PAGE LAYOUT  (approximate — horizontal position and line spacing")
    print(f"  scaled from the image; not a precise ruler)")
    print(f"  {'─'*50}")

    prev_line_bottom = None
    for line_boxes, ens_line in zip(lines, ensemble_lines):
        # Blank line(s) when the gap above this line is unusually large —
        # mirrors a paragraph break / skipped row on the physical page.
        line_top = min(b[1] for b in line_boxes)
        if prev_line_bottom is not None:
            gap = line_top - prev_line_bottom
            avg_h = sum(b[3] for b in line_boxes) / len(line_boxes)
            if gap > avg_h * 1.5:
                print()
        prev_line_bottom = max(b[1] + b[3] for b in line_boxes)

        # Build the visible characters for this line (skip the synthetic
        # space entries already inserted for in-line gaps — those are about
        # spacing within a line, not the line's position on the page).
        chars = [(e[0], e[1]) for e in ens_line if e[0] != " "]
        left_x = min(b[0] for b in line_boxes)
        indent = int((left_x / img_w) * layout_width)

        rendered = ""
        for label, agreement in chars:
            if agreement == "NON-DIGIT":
                rendered += "[NON-DIGIT?]"
            elif agreement in ("MAJORITY", "WEIGHTED"):
                rendered += f"[{label}]"
            else:
                rendered += label
        print(f"  {' ' * indent}{rendered}")

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
            print(f"\n    Char {box_idx+1:>2}  [x:{x} y:{y} w:{w} h:{h} asp:{aspect:.2f}]  vote={flag}  final={final_label}{nd_warn}")
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
