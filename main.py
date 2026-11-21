#!/usr/bin/env python3
"""
Sheet Music Extractor - Gradio GUI

Turns a sheet-music transcription video into a clean PDF, with a visual
review step so you can fix mistakes instead of tuning thresholds blindly.

  pip install gradio opencv-python pillow numpy
  pip install yt-dlp          # optional, only for YouTube URLs
  python sheet_music_app.py
"""

import os
import subprocess
import sys
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# On a public host (Render sets RENDER=true; or set HOSTED=1 yourself) the YouTube
# and local-file-path inputs are disabled: YouTube blocks datacenter IPs, and a
# path box would let visitors read files from the server.
IS_HOSTED = bool(os.environ.get("RENDER") or os.environ.get("HOSTED"))

# =====================================================================
# Core pipeline (same logic as video_to_sheet_pdf.py)
# =====================================================================


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "duration": n / fps if fps else 0}


def sample_frames(path, interval):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * interval))
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / fps, frame
        idx += 1
    cap.release()


def grab_frame(path, t):
    """Jump straight to time t (seconds) instead of decoding everything before it."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def staff_line_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    w = frame.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 8), 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def detect_notation_box(frame):
    lines = staff_line_mask(frame)
    h, w = lines.shape
    merged = cv2.dilate(
        lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40)))
    )
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
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
    staff_h = min(b[3] for b in boxes)
    pad_y = int(staff_h * 1.2)
    pad_x = int(w * 0.01)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x) - max(0, x0 - pad_x),
        min(h, y1 + pad_y) - max(0, y0 - pad_y),
    )


def trim_to_paper(frame, roi):
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


def detect_roi(path, n_samples=40):
    """Auto-detect the notation area by seeking to evenly spaced frames (fast)."""
    info = video_info(path)
    cap = cv2.VideoCapture(path)
    boxes, mid_frame = [], None
    for pos in np.linspace(0.03, 0.97, n_samples) * max(1, info["frames"] - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        if mid_frame is None and pos >= info["frames"] * 0.5:
            mid_frame = frame
        b = detect_notation_box(frame)
        if b:
            boxes.append(b)
    cap.release()
    if not boxes:
        return None
    arr = np.array(boxes)
    x0 = int(np.percentile(arr[:, 0], 10))
    y0 = int(np.percentile(arr[:, 1], 10))
    x1 = int(np.percentile(arr[:, 0] + arr[:, 2], 90))
    y1 = int(np.percentile(arr[:, 1] + arr[:, 3], 90))
    roi = (x0, y0, x1 - x0, y1 - y0)
    if mid_frame is not None:
        roi = trim_to_paper(mid_frame, roi)
    return roi


def clean_crop(crop):
    """Whiten colored pixels (playhead cursor, note highlights)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    out = crop.copy()
    out[(hsv[..., 1] > 80) & (hsv[..., 2] > 60)] = 255
    return out


def trim_blank(crop, pad_ratio=0.015):
    """Cut blank space above/below the content, keeping the full width.
    Width stays fixed so every page is scaled by the same factor in the PDF
    (staff size stays consistent) while the height follows the content."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = np.where((gray < 160).sum(axis=1) >= max(3, int(w * 0.003)))[0]
    if rows.size == 0:
        return crop
    pad = max(6, int(w * pad_ratio))
    return crop[max(0, rows[0] - pad) : min(h, rows[-1] + 1 + pad)]


def finish_crop(crop, clean, trim):
    out = clean_crop(crop) if clean else crop.copy()
    return trim_blank(out) if trim else out


def ink_signature(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ink = ((hsv[..., 2] < 110) & (hsv[..., 1] < 80)).astype(np.uint8) * 255
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
    a8, b8 = a.astype(np.uint8), b.astype(np.uint8)
    a_only = np.count_nonzero(a8 & ~cv2.dilate(b8, _TOL).astype(bool))
    b_only = np.count_nonzero(b8 & ~cv2.dilate(a8, _TOL).astype(bool))
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return (a_only + b_only) / total if total else 0.0


def extract_pages(path, roi, interval, change, stable, clean, trim=True, on_progress=None):
    """Returns a list of {"img": BGR crop, "t": seconds}."""
    x, y, w, h = roi
    duration = video_info(path)["duration"] or 1
    kept, last_sig, prev_sig = [], None, None
    for t, frame in sample_frames(path, interval):
        if on_progress:
            on_progress(min(t / duration, 1.0), len(kept))
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        sig = ink_signature(crop)
        if sig.mean() < 0.005:
            prev_sig = sig
            continue
        is_stable = prev_sig is not None and diff_ratio(sig, prev_sig) < stable
        changed = last_sig is None or diff_ratio(sig, last_sig) > change
        if is_stable and changed:
            kept.append({"img": finish_crop(crop, clean, trim), "t": t})
            last_sig = sig
        prev_sig = sig
    return kept


def build_pdf(crops, out_path, dpi=150, margin_mm=10, gap_mm=6):
    page_w, page_h = int(8.27 * dpi), int(11.69 * dpi)  # A4
    margin = int(margin_mm / 25.4 * dpi)
    gap = int(gap_mm / 25.4 * dpi)
    usable_w = page_w - 2 * margin

    pages, page, cursor = [], None, 0
    for crop in crops:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        scale = usable_w / img.width
        img = img.resize((usable_w, max(1, int(img.height * scale))), Image.LANCZOS)
        if img.height > page_h - 2 * margin:
            s = (page_h - 2 * margin) / img.height
            img = img.resize((max(1, int(img.width * s)), page_h - 2 * margin), Image.LANCZOS)
        if page is None or cursor + img.height > page_h - margin:
            page = Image.new("RGB", (page_w, page_h), "white")
            pages.append(page)
            cursor = margin
        page.paste(img, (margin + (usable_w - img.width) // 2, cursor))
        cursor += img.height + gap

    pages[0].save(out_path, save_all=True, append_images=pages[1:], resolution=dpi)


# =====================================================================
# GUI helpers
# =====================================================================


def download_youtube(url, progress):
    progress(0, desc="Downloading from YouTube...")
    out_dir = tempfile.mkdtemp(prefix="sheet_dl_")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=1080]+ba/b",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
        url,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise gr.Error("yt-dlp is not installed. Run: pip install yt-dlp")
    if res.returncode != 0:
        tail = (res.stderr or res.stdout)[-400:]
        if "No module named yt_dlp" in tail:
            raise gr.Error("yt-dlp is not installed. Run: pip install yt-dlp")
        raise gr.Error(f"Download failed: {tail}")
    files = [f for f in os.listdir(out_dir) if not f.endswith((".part", ".ytdl"))]
    if not files:
        raise gr.Error("Download finished but no file was found.")
    return os.path.join(out_dir, files[0])


def draw_roi_preview(path, t, x, y, w, h):
    if not path:
        return None
    frame = grab_frame(path, t)
    if frame is None:
        return None
    H, W = frame.shape[:2]
    x, y = int(max(0, min(x, W - 1))), int(max(0, min(y, H - 1)))
    w, h = int(max(1, min(w, W - x))), int(max(1, min(h, H - y)))
    thick = max(2, W // 300)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thick)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_gallery(pages, sel=None):
    items = [
        (cv2.cvtColor(p["img"], cv2.COLOR_BGR2RGB), f"#{i + 1}  ({p['t']:.1f}s)")
        for i, p in enumerate(pages)
    ]
    return gr.update(value=items, selected_index=sel)


# =====================================================================
# Event handlers
# =====================================================================


def on_load(upload, source, progress=gr.Progress()):
    path = None
    source = "" if IS_HOSTED else (source or "").strip().strip('"')
    if source:
        if source.lower().startswith(("http://", "https://")):
            path = download_youtube(source, progress)
        elif os.path.isfile(source):
            path = source
        else:
            raise gr.Error(f"File not found: {source}")
    elif upload:
        path = upload if isinstance(upload, str) else upload.name
    if not path:
        raise gr.Error("Upload a video, or paste a file path / YouTube URL.")

    progress(0.5, desc="Detecting notation area...")
    try:
        info = video_info(path)
    except ValueError as e:
        raise gr.Error(str(e))
    W, H = info["w"], info["h"]

    roi = detect_roi(path)
    if roi:
        status = (
            f"Loaded **{os.path.basename(path)}** ({W}x{H}, {info['duration']:.0f}s). "
            "Notation area detected. Check the red box and adjust the sliders if needed."
        )
    else:
        roi = (0, 0, W, H)
        status = (
            f"Loaded **{os.path.basename(path)}** ({W}x{H}, {info['duration']:.0f}s). "
            "No staff lines detected automatically. Please set the red box with the sliders."
        )

    x, y, w, h = roi
    t0 = min(info["duration"] * 0.5, max(0, info["duration"] - 0.1))
    return (
        path,
        status,
        gr.update(maximum=W, value=x),
        gr.update(maximum=H, value=y),
        gr.update(maximum=W, value=w),
        gr.update(maximum=H, value=h),
        gr.update(maximum=max(1, info["duration"]), value=t0),
        draw_roi_preview(path, t0, x, y, w, h),
        [],
        render_gallery([]),
        None,
        os.path.splitext(os.path.basename(path))[0],
    )


def on_preview(path, t, x, y, w, h):
    return draw_roi_preview(path, t, x, y, w, h)


def on_redetect(path, t):
    if not path:
        raise gr.Error("Load a video first.")
    roi = detect_roi(path)
    if not roi:
        gr.Warning("No staff lines detected. Set the box manually.")
        info = video_info(path)
        roi = (0, 0, info["w"], info["h"])
    x, y, w, h = roi
    return x, y, w, h, draw_roi_preview(path, t, x, y, w, h)


def on_extract(path, x, y, w, h, interval, change, stable, clean, trim, progress=gr.Progress()):
    if not path:
        raise gr.Error("Load a video first.")
    roi = (int(x), int(y), int(w), int(h))

    def cb(frac, n):
        progress(frac, desc=f"Scanning video... {n} pages found")

    pages = extract_pages(path, roi, interval, change, stable, clean, trim, cb)
    if not pages:
        gr.Warning(
            "No pages found. Check the red box, lower 'New-page sensitivity', "
            "or raise 'Stability tolerance'."
        )
    else:
        gr.Info(f"Found {len(pages)} pages. Review them below, then export.")
    return pages, render_gallery(pages), None


def on_select(evt: gr.SelectData):
    return evt.index


def on_delete(pages, sel):
    if sel is None or sel >= len(pages):
        gr.Warning("Click a page in the gallery first.")
        return pages, render_gallery(pages), sel
    pages = pages[:sel] + pages[sel + 1 :]
    return pages, render_gallery(pages), None


def on_move(pages, sel, direction):
    if sel is None or sel >= len(pages):
        gr.Warning("Click a page in the gallery first.")
        return pages, render_gallery(pages), sel
    j = sel + direction
    if not 0 <= j < len(pages):
        return pages, render_gallery(pages, sel), sel
    pages = list(pages)
    pages[sel], pages[j] = pages[j], pages[sel]
    return pages, render_gallery(pages, j), j


def on_add_frame(pages, path, t, x, y, w, h, clean, trim):
    if not path:
        raise gr.Error("Load a video first.")
    frame = grab_frame(path, t)
    if frame is None:
        raise gr.Error("Could not read that frame.")
    x, y, w, h = int(x), int(y), int(w), int(h)
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        raise gr.Error("The red box is outside the video frame.")
    new = {"img": finish_crop(crop, clean, trim), "t": float(t)}
    pos = sum(1 for p in pages if p["t"] <= t)  # keep time order
    pages = list(pages[:pos]) + [new] + list(pages[pos:])
    gr.Info(f"Added as page #{pos + 1}.")
    return pages, render_gallery(pages, pos), pos


def safe_filename(name):
    """Strip characters that are invalid in file names and any .pdf suffix."""
    name = (name or "").strip()
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name)
    name = name.strip(" .")
    return name or "score"


def on_export(pages, dpi, gap, filename):
    if not pages:
        raise gr.Error("There are no pages to export.")
    out = os.path.join(tempfile.mkdtemp(prefix="sheet_pdf_"), safe_filename(filename) + ".pdf")
    build_pdf([p["img"] for p in pages], out, dpi=int(dpi), gap_mm=gap)
    return out


# =====================================================================
# UI
# =====================================================================

with gr.Blocks(title="Sheet Music Extractor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Sheet Music Extractor\n"
        "Turn a transcription video into a PDF. **1. Load**, **2. Set the area**, "
        "**3. Extract**, **4. Review**, **5. Export**."
    )

    video_path = gr.State(None)
    pages_state = gr.State([])
    selected = gr.State(None)

    # ---- 1. Load ----
    with gr.Group():
        gr.Markdown("### 1. Load video")
        with gr.Row():
            upload = gr.File(
                label="Upload a video",
                file_types=["video"],
                type="filepath",
            )
            source = gr.Textbox(
                label="...or paste a YouTube URL / local file path",
                placeholder="https://www.youtube.com/watch?v=...   or   C:\\videos\\song.mp4",
                visible=not IS_HOSTED,
            )
        if IS_HOSTED:
            gr.Markdown(
                "YouTube links can't be fetched from this server (YouTube blocks it). "
                "Download the video on your own computer first, then upload it here. "
                "The YouTube link option works when you run the app locally."
            )
        load_btn = gr.Button("Load video", variant="primary")
        status = gr.Markdown()

    # ---- 2. Area ----
    with gr.Group():
        gr.Markdown(
            "### 2. Set the notation area\n"
            "The red box is what gets cropped. Scrub the video and adjust the sliders until "
            "it covers the sheet music."
        )
        preview = gr.Image(label="Preview", interactive=False, show_label=False)
        t_slider = gr.Slider(0, 1, value=0, step=0.5, label="Preview time (seconds)")
        with gr.Row():
            sx = gr.Slider(0, 1920, value=0, step=1, label="X (left)")
            sy = gr.Slider(0, 1080, value=0, step=1, label="Y (top)")
            sw = gr.Slider(1, 1920, value=1920, step=1, label="Width")
            sh = gr.Slider(1, 1080, value=1080, step=1, label="Height")
        redetect_btn = gr.Button("Re-detect automatically", size="sm")

    # ---- 3. Extract ----
    with gr.Group():
        gr.Markdown("### 3. Extract pages")
        with gr.Accordion("Advanced settings", open=False):
            interval = gr.Slider(0.1, 3, value=0.5, step=0.1, label="Seconds between sampled frames")
            change = gr.Slider(
                0.05, 0.9, value=0.35, step=0.05,
                label="New-page sensitivity (lower = keeps more pages)",
            )
            stable = gr.Slider(
                0.02, 0.5, value=0.15, step=0.01,
                label="Stability tolerance (higher = accepts frames mid-transition)",
            )
            clean = gr.Checkbox(value=True, label="Remove colored cursor / highlights")
            trim = gr.Checkbox(
                value=True,
                label="Auto-trim blank space above/below each page",
                info="Lets you draw a generous box that covers the title slide; pages without a title are trimmed to their content.",
            )
        extract_btn = gr.Button("Extract pages", variant="primary")

    # ---- 4. Review ----
    with gr.Group():
        gr.Markdown(
            "### 4. Review\n"
            "Click a page to select it, then delete or reorder it. To add a missed page, scrub the "
            "preview time above to a moment where it is fully visible and click **Add frame**."
        )
        gallery = gr.Gallery(
            label="Detected pages", columns=2, height="auto",
            object_fit="contain", allow_preview=True,
        )
        with gr.Row():
            del_btn = gr.Button("Delete selected")
            up_btn = gr.Button("Move up")
            down_btn = gr.Button("Move down")
            add_btn = gr.Button("Add frame at preview time")

    # ---- 5. Export ----
    with gr.Group():
        gr.Markdown("### 5. Export")
        with gr.Row():
            dpi = gr.Slider(100, 300, value=150, step=25, label="PDF resolution (DPI)")
            gap = gr.Slider(0, 20, value=6, step=1, label="Gap between systems (mm)")
        filename = gr.Textbox(
            label="File name",
            value="score",
            info="Filled in from the video name. The .pdf extension is added automatically.",
        )
        export_btn = gr.Button("Export PDF", variant="primary")
        pdf_out = gr.File(label="Your PDF")

    # ---- Wiring ----
    roi_inputs = [video_path, t_slider, sx, sy, sw, sh]

    load_outputs = [
        video_path, status, sx, sy, sw, sh, t_slider, preview,
        pages_state, gallery, selected, filename,
    ]
    load_btn.click(on_load, [upload, source], load_outputs)

    for comp in (t_slider, sx, sy, sw, sh):
        comp.release(on_preview, roi_inputs, preview, show_progress="hidden")

    redetect_btn.click(on_redetect, [video_path, t_slider], [sx, sy, sw, sh, preview])

    extract_btn.click(
        on_extract,
        [video_path, sx, sy, sw, sh, interval, change, stable, clean, trim],
        [pages_state, gallery, pdf_out],
    )

    gallery.select(on_select, None, selected)
    del_btn.click(on_delete, [pages_state, selected], [pages_state, gallery, selected])
    up_btn.click(lambda p, s: on_move(p, s, -1), [pages_state, selected], [pages_state, gallery, selected])
    down_btn.click(lambda p, s: on_move(p, s, 1), [pages_state, selected], [pages_state, gallery, selected])
    add_btn.click(
        on_add_frame,
        [pages_state, video_path, t_slider, sx, sy, sw, sh, clean, trim],
        [pages_state, gallery, selected],
    )

    export_btn.click(on_export, [pages_state, dpi, gap, filename], pdf_out)


if __name__ == "__main__":
    if IS_HOSTED:
        demo.queue(max_size=5).launch(
            server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))
        )
    else:
        demo.launch(inbrowser=True)
