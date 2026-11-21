#!/usr/bin/env python3
"""
Extract sheet music from a transcription video into a PDF.

Pipeline:
  1. Sample frames from the video (every --interval seconds).
  2. Detect the notation region by finding staff lines (long thin horizontal lines).
  3. Crop each frame to that region.
  4. Keep only frames where the notation actually changed (new page/system),
     waiting until the image is stable so page-turn transitions are skipped.
  5. Stack the unique crops onto A4 pages and save as PDF.

Usage:
  pip install opencv-python pillow numpy
  python video_to_sheet_pdf.py input.mp4 -o score.pdf
  python video_to_sheet_pdf.py input.mp4 -o score.pdf --roi 0,620,1920,460   # manual crop x,y,w,h
  python video_to_sheet_pdf.py input.mp4 --debug                              # save crops to ./debug_crops
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

# ---------- 1. Frame sampling ----------


def sample_frames(path, interval):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * interval))
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / fps, frame
        idx += 1
    cap.release()


# ---------- 2. Notation region detection ----------


def staff_line_mask(frame):
    """Binary mask of long horizontal dark lines (staff lines)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Dark ink on light background -> white pixels in `binary`
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    w = frame.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 8), 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def detect_notation_box(frame):
    """Return (x, y, w, h) around all staves in the frame, or None."""
    lines = staff_line_mask(frame)
    # Merge the 5 lines of each staff (and nearby staves) into blobs
    h, w = lines.shape
    merged = cv2.dilate(
        lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40)))
    )
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        # A real staff: wide, and contains several parallel lines
        if bw < w * 0.25:
            continue
        rows = np.where(lines[y : y + bh, x : x + bw].sum(axis=1) > 0)[0]
        n_lines = np.count_nonzero(np.diff(rows) > 1) + 1 if rows.size else 0
        if n_lines >= 4:
            boxes.append((x, y, bw, bh))
    if not boxes:
        return None

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    # Pad for notes/ledger lines/lyrics above & below the staff lines
    staff_h = min(b[3] for b in boxes)
    pad_y = int(staff_h * 1.2)
    pad_x = int(w * 0.01)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x) - max(0, x0 - pad_x),
        min(h, y1 + pad_y) - max(0, y0 - pad_y),
    )


def find_global_roi(video, interval, max_samples=60):
    """Union of notation boxes across many frames (layout is usually fixed)."""
    boxes = []
    for i, (_, frame) in enumerate(sample_frames(video, interval * 4)):
        b = detect_notation_box(frame)
        if b:
            boxes.append(b)
        if i >= max_samples:
            break
    if not boxes:
        raise SystemExit("No staff lines detected. Try --roi x,y,w,h manually.")
    arr = np.array(boxes)
    # Use percentiles instead of min/max so a few bad detections don't blow up the box
    x0 = int(np.percentile(arr[:, 0], 10))
    y0 = int(np.percentile(arr[:, 1], 10))
    x1 = int(np.percentile(arr[:, 0] + arr[:, 2], 90))
    y1 = int(np.percentile(arr[:, 1] + arr[:, 3], 90))
    return x0, y0, x1 - x0, y1 - y0


def trim_to_paper(frame, roi):
    """Shrink ROI so it doesn't include dark background outside the white score area."""
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    bright = gray > 180
    rows = np.where(bright.mean(axis=1) > 0.5)[0]
    cols = np.where(bright.mean(axis=0) > 0.5)[0]
    if rows.size == 0 or cols.size == 0:
        return roi
    return (
        int(x + cols[0]),
        int(y + rows[0]),
        int(cols[-1] - cols[0] + 1),
        int(rows[-1] - rows[0] + 1),
    )


def clean_crop(crop):
    """Whiten colored pixels (playhead cursor, note highlights) so the PDF is plain black & white."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    out = crop.copy()
    out[(hsv[..., 1] > 80) & (hsv[..., 2] > 60)] = 255
    return out


# ---------- 3/4. Crop + dedupe ----------


def ink_signature(crop):
    """Small binarized image used to compare crops. Ignores colored playhead/highlights."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Black ink = low value AND low saturation (colored cursors have high saturation)
    ink = ((hsv[..., 2] < 110) & (hsv[..., 1] < 80)).astype(np.uint8) * 255
    # Remove staff lines: they're identical on every page and drown out note changes
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, crop.shape[1] // 10), 1))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz))
    return (
        cv2.resize(
            ink,
            (400, max(20, int(400 * crop.shape[0] / crop.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        > 64
    )


_TOL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def diff_ratio(a, b):
    """Fraction of ink in either image with no nearby match in the other (0 = same, 1 = different).
    Tolerant to 1-2px shifts, compression noise and small occlusions (cursors)."""
    a8, b8 = a.astype(np.uint8), b.astype(np.uint8)
    a_only = np.count_nonzero(a8 & ~cv2.dilate(b8, _TOL).astype(bool))
    b_only = np.count_nonzero(b8 & ~cv2.dilate(a8, _TOL).astype(bool))
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return (a_only + b_only) / total if total else 0.0


def extract_pages(
    video, roi, interval, change_thresh=0.35, stable_thresh=0.15, clean=True
):
    x, y, w, h = roi
    kept, last_sig, prev_sig = [], None, None
    for t, frame in sample_frames(video, interval):
        crop = frame[y : y + h, x : x + w]
        sig = ink_signature(crop)
        if sig.mean() < 0.005:  # blank / intro / transition frame
            prev_sig = sig
            continue
        stable = prev_sig is not None and diff_ratio(sig, prev_sig) < stable_thresh
        changed = last_sig is None or diff_ratio(sig, last_sig) > change_thresh
        if stable and changed:
            kept.append(clean_crop(crop) if clean else crop.copy())
            last_sig = sig
            print(f"  page {len(kept):3d} @ {t:7.1f}s")
        prev_sig = sig
    return kept


# ---------- 5. PDF ----------


def build_pdf(crops, out_path, dpi=150, margin_mm=10, gap_mm=6):
    page_w, page_h = int(8.27 * dpi), int(11.69 * dpi)  # A4
    margin = int(margin_mm / 25.4 * dpi)
    gap = int(gap_mm / 25.4 * dpi)
    usable_w = page_w - 2 * margin

    pages, page, cursor = [], None, 0
    for crop in crops:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        scale = usable_w / img.width
        img = img.resize((usable_w, int(img.height * scale)), Image.LANCZOS)
        if img.height > page_h - 2 * margin:  # tall crop: fit to page height
            s = (page_h - 2 * margin) / img.height
            img = img.resize((int(img.width * s), page_h - 2 * margin), Image.LANCZOS)
        if page is None or cursor + img.height > page_h - margin:
            page = Image.new("RGB", (page_w, page_h), "white")
            pages.append(page)
            cursor = margin
        page.paste(img, (margin + (usable_w - img.width) // 2, cursor))
        cursor += img.height + gap

    pages[0].save(out_path, save_all=True, append_images=pages[1:], resolution=dpi)


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default="score.pdf")
    ap.add_argument(
        "--interval", type=float, default=0.5, help="seconds between sampled frames"
    )
    ap.add_argument("--roi", help="manual crop x,y,w,h (skips auto-detection)")
    ap.add_argument(
        "--change", type=float, default=0.35, help="min ink diff to count as a new page"
    )
    ap.add_argument(
        "--stable",
        type=float,
        default=0.15,
        help="max ink diff between samples to count as stable",
    )
    ap.add_argument(
        "--keep-color",
        action="store_true",
        help="don't remove colored cursors/highlights",
    )
    ap.add_argument(
        "--debug", action="store_true", help="save crops as PNGs in ./debug_crops"
    )
    args = ap.parse_args()

    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
    else:
        print("Detecting notation region...")
        roi = find_global_roi(args.video, args.interval)
        # Trim using a frame from the middle of the video
        frames = [f for _, f in sample_frames(args.video, args.interval * 10)]
        roi = trim_to_paper(frames[len(frames) // 2], roi)
    print(f"ROI (x,y,w,h): {roi}")

    print("Extracting unique pages...")
    crops = extract_pages(
        args.video,
        roi,
        args.interval,
        args.change,
        args.stable,
        clean=not args.keep_color,
    )
    if not crops:
        raise SystemExit(
            "No pages extracted. Try lowering --change or raising --stable."
        )

    if args.debug:
        os.makedirs("debug_crops", exist_ok=True)
        for i, c in enumerate(crops, 1):
            cv2.imwrite(f"debug_crops/{i:03d}.png", c)

    build_pdf(crops, args.output)
    print(f"Saved {len(crops)} systems -> {args.output}")


if __name__ == "__main__":
    main()
