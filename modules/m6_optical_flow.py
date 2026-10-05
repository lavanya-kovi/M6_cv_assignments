"""
CSc 8830 Computer Vision - Module 6, Part A: optical flow and motion tracking
=============================================================================

README / HOW TO RUN
-------------------
    pip install -r requirements.txt

    # 1. Flow video + motion statistics for a 30 s clip starting at 5 s
    python modules/m6_optical_flow.py flow  my_video.mp4 --start 5 --duration 30 --out results/video1

    # 2. Two-frame tracking validation (frame index inside the clip)
    python modules/m6_optical_flow.py track my_video.mp4 --start 5 --frame 120 --out results/video1

Outputs (in --out):
    flow_video.mp4      original | colour-coded flow | flow arrows, side by side
    flow_stats.csv      per-frame mean speed, moving fraction, direction, divergence
    flow_evidence.png   plots of what the flow tells us (speed, direction, pan, approach)
    track_frames.png    the two frames with tracked points and displacement arrows
    track_table.csv     per point: start (x,y), LK prediction, measured location, error
    track_system.txt    the 2x2 Lucas-Kanade system written out for one point

What each part does
    * Dense optical flow: Farneback polynomial-expansion method (OpenCV).
    * Colour coding: hue = direction, brightness = speed (Middlebury convention).
    * Tracking: Lucas-Kanade implemented here from the derivation (not the
      OpenCV call), using our own bilinear interpolation for sub-pixel sampling.
    * Validation: the LK prediction p + d is compared against the *measured*
      location of the same patch in frame t+1, found independently by
      normalised cross-correlation (template matching) with sub-pixel parabola
      refinement, and against OpenCV's pyramidal LK as a second reference.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video input / output
# ---------------------------------------------------------------------------
def read_clip(path: str, start_s: float = 0.0, duration_s: float = 30.0,
              max_side: int = 480, step: int = 1):
    """Read `duration_s` seconds from `start_s`, resized so the longest side <= max_side."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames, n_max = [], int(round(duration_s * fps))
    while len(frames) * step < n_max:
        ok, f = cap.read()
        if not ok:
            break
        for _ in range(step - 1):
            cap.grab()
        h, w = f.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            f = cv2.resize(f, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps / step


def write_video(frames, path: str, fps: float):
    """H.264 MP4 (plays in browsers) via imageio-ffmpeg; falls back to OpenCV mp4v."""
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                macro_block_size=2, ffmpeg_log_level="error") as w:
            for f in frames:
                w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    except Exception:  # pragma: no cover - fallback path
        h, w_ = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_, h))
        for f in frames:
            vw.write(f)
        vw.release()


# ---------------------------------------------------------------------------
# Dense optical flow and its visualisation
# ---------------------------------------------------------------------------
def dense_flow(prev_bgr, next_bgr):
    """Farneback dense flow: returns H x W x 2 array (u, v) in pixels/frame."""
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=3, poly_n=7, poly_sigma=1.5, flags=0)


def flow_to_color(flow, max_mag: float | None = None):
    """HSV coding: hue = direction angle, value = magnitude (saturating at max_mag)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_mag is None:
        max_mag = max(np.percentile(mag, 99), 1e-3)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)              # OpenCV hue range 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(img, flow, step: int = 16, scale: float = 3.0, min_mag: float = 0.3):
    out = img.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]
            if u * u + v * v < min_mag ** 2:
                continue
            cv2.arrowedLine(out, (x, y), (int(x + scale * u), int(y + scale * v)),
                            (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)
    return out


def color_wheel(size: int = 120):
    """Legend for the colour coding (direction -> hue)."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    f = np.dstack([x, y]).astype(np.float32)
    img = flow_to_color(f, max_mag=1.0)
    img[np.hypot(x, y) > 1] = 255
    return img


def flow_statistics(flow, moving_thresh: float = 0.5):
    """Quantities that can be inferred from one flow field."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    moving = mag > moving_thresh
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    stats = {
        "mean_speed_px": float(mag.mean()),
        "moving_speed_px": float(mag[moving].mean()) if moving.any() else 0.0,
        "moving_fraction": float(moving.mean()),
        # median flow of the whole frame ~ camera pan (background dominates)
        "global_u": float(np.median(u)), "global_v": float(np.median(v)),
        # flow-weighted mean direction of moving pixels (degrees, image coords: 0 = right, 90 = down)
        "direction_deg": float(np.degrees(np.arctan2(v[moving].sum(), u[moving].sum())) % 360)
        if moving.any() else float("nan"),
        # divergence > 0: expansion (approaching / zoom in); < 0: contraction
        "divergence": float((du_dx + dv_dy)[moving].mean()) if moving.any() else 0.0,
        # curl != 0: rotation (image coords, y down: curl > 0 = clockwise on screen)
        "curl": float((dv_dx - du_dy)[moving].mean()) if moving.any() else 0.0,
    }
    return stats


def iter_frames(path: str, start_s: float = 0.0, duration_s: float = 30.0,
                max_side: int = 480, step: int = 1):
    """Stream frames (resized) without holding the clip in memory. Yields (frame, fps)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    n_max, n = int(round(duration_s * fps)), 0
    while n < n_max:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        for _ in range(step - 1):
            cap.grab()
            n += 1
        h, w = f.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            f = cv2.resize(f, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        yield f, fps / step
    cap.release()


def process_clip(path: str, out_dir: str, start_s: float = 0.0, duration_s: float = 30.0,
                 max_side: int = 480, step: int = 1, moving_thresh: float = 0.5, progress=None):
    """
    Stream the clip: flow for every consecutive pair, written straight into the
    visualisation video (original | colour-coded flow | arrows) and per-frame stats.
    The colour scale is fixed from the first second so colours are comparable over time.
    Returns the stats rows, the colour-scale maximum and a sample flow field.
    """
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    gen = iter_frames(path, start_s, duration_s, max_side, step)
    prev, fps = next(gen)
    buffer, rows, writer, max_mag, sample = [], [], None, None, None
    total = int(duration_s * fps)

    def emit(i, f0, f):
        nonlocal writer
        col = flow_to_color(f, max_mag)
        wh = wheel.shape[0]
        col[4:4 + wh, 4:4 + wh] = wheel
        panel = np.hstack([f0, col, draw_arrows(f0, f)])
        cv2.putText(panel, f"t = {i / fps:5.2f} s", (8, panel.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = imageio.get_writer(os.path.join(out_dir, "flow_video.mp4"), fps=fps, codec="libx264",
                                        quality=7, macro_block_size=2, ffmpeg_log_level="error")
        writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        s = flow_statistics(f, moving_thresh)
        s["frame"], s["time_s"] = i, i / fps
        rows.append(s)

    wheel = color_wheel(min(100, prev.shape[0] // 4))
    i = 0
    for cur, _ in gen:
        f = dense_flow(prev, cur)
        if max_mag is None:
            buffer.append((i, prev, f))
            if len(buffer) >= max(2, int(fps)):          # one second to set the colour scale
                max_mag = max(max(np.percentile(np.hypot(b[2][..., 0], b[2][..., 1]), 99) for b in buffer), 0.5)
                for b in buffer:
                    emit(*b)
                buffer = []
        else:
            emit(i, prev, f)
        if sample is None or i == total // 2:
            sample = (prev.copy(), f)
        prev, i = cur, i + 1
        if progress:
            progress(i, total)
    if buffer:
        max_mag = max(max(np.percentile(np.hypot(b[2][..., 0], b[2][..., 1]), 99) for b in buffer), 0.5)
        for b in buffer:
            emit(*b)
    if writer is not None:
        writer.close()
    with open(os.path.join(out_dir, "flow_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows, max_mag, fps, sample


def plot_flow_evidence(rows, sample, max_mag, out_path: str, title: str = ""):
    """Evidence figure: speed and moving area over time, direction histogram, sample flow frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["time_s"] for r in rows])
    fig = plt.figure(figsize=(11, 6.2))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, [r["moving_speed_px"] for r in rows], color="tab:blue", lw=1)
    ax1.set_ylabel("speed of moving pixels\n(px / frame)")
    ax1b = ax1.twinx()
    ax1b.plot(t, [100 * r["moving_fraction"] for r in rows], color="tab:orange", lw=1)
    ax1b.set_ylabel("moving area (%)", color="tab:orange")
    ax1.set_xlabel("time (s)")
    ax1.set_title("How fast and how much is moving")
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    d = np.radians([r["direction_deg"] for r in rows if not np.isnan(r["direction_deg"])])
    ax2.hist(d, bins=24, color="tab:green")
    ax2.set_theta_direction(-1)          # image coordinates: y points down
    ax2.set_title("Dominant motion direction per frame\n(0 deg = right, 90 deg = down)")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, [r["global_u"] for r in rows], label="median u (camera pan x)")
    ax3.plot(t, [r["global_v"] for r in rows], label="median v (camera pan y)")
    ax3.plot(t, [100 * r["divergence"] for r in rows], label="divergence x100 (+ approach)")
    ax3.plot(t, [100 * r["curl"] for r in rows], label="curl x100 (+ clockwise on screen)")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_xlabel("time (s)")
    ax3.legend(fontsize=7)
    ax3.set_title("Camera pan, approach (divergence), rotation (curl)")
    ax4 = fig.add_subplot(2, 2, 4)
    frame, f = sample
    ax4.imshow(cv2.cvtColor(np.hstack([frame, flow_to_color(f, max_mag)]), cv2.COLOR_BGR2RGB))
    ax4.axis("off")
    ax4.set_title("Sample frame and its flow")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bilinear interpolation (derived in the report)
# ---------------------------------------------------------------------------
def bilinear(img, x, y):
    """
    Sample image `img` (H x W float) at real-valued (x, y) arrays.
        x0 = floor(x), a = x - x0,  y0 = floor(y), b = y - y0
        I(x, y) = (1-a)(1-b) I00 + a(1-b) I10 + (1-a) b I01 + a b I11
    where Ixy are the four neighbouring pixels. Coordinates are clamped to the image.
    """
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    x = np.clip(np.asarray(x, np.float64), 0, w - 1.000001)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1.000001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = x - x0, y - y0
    i00, i10 = img[y0, x0], img[y0, x0 + 1]
    i01, i11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (1 - a) * (1 - b) * i00 + a * (1 - b) * i10 + (1 - a) * b * i01 + a * b * i11


# ---------------------------------------------------------------------------
# Lucas-Kanade tracking implemented from the derivation
# ---------------------------------------------------------------------------
def image_gradients(gray):
    """Central-difference (Scharr-smoothed) spatial gradients Ix, Iy."""
    g = gray.astype(np.float64)
    ix = cv2.Scharr(g, cv2.CV_64F, 1, 0) / 32.0
    iy = cv2.Scharr(g, cv2.CV_64F, 0, 1) / 32.0
    return ix, iy


def lk_track_point(i1, i2, ix, iy, p, half: int = 7, iters: int = 20, eps: float = 1e-3,
                   d0=(0.0, 0.0)):
    """
    Iterative Lucas-Kanade for one point p = (x, y) between grey images i1, i2.

    Brightness constancy + first-order Taylor:  Ix u + Iy v + It = 0 for every
    pixel q in the window W(p). Least squares over the window:
        G d = b,   G = sum [Ix^2  IxIy; IxIy  Iy^2],   b = -sum [Ix It; Iy It]
    with It = I2(q + d) - I1(q), sampled by bilinear interpolation, and the
    update d <- d + G^-1 b repeated until |delta| < eps.
    Returns d, G, list of iterations.
    """
    x, y = p
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    qx, qy = x + gx.ravel(), y + gy.ravel()
    t1 = bilinear(i1, qx, qy)
    ixw, iyw = bilinear(ix, qx, qy), bilinear(iy, qx, qy)
    G = np.array([[np.sum(ixw * ixw), np.sum(ixw * iyw)],
                  [np.sum(ixw * iyw), np.sum(iyw * iyw)]])
    d = np.array(d0, np.float64)
    history = []
    for _ in range(iters):
        it = bilinear(i2, qx + d[0], qy + d[1]) - t1
        b = -np.array([np.sum(ixw * it), np.sum(iyw * it)])
        delta = np.linalg.solve(G, b)
        d = d + delta
        history.append((d.copy(), float(np.sqrt(np.mean(it ** 2)))))
        if np.hypot(*delta) < eps:
            break
    return d, G, history


def lk_track_pyramid(g1, g2, p, levels: int = 3, half: int = 7):
    """Coarse-to-fine LK: estimate at the coarsest level, double, refine (handles large motion)."""
    pyr1, pyr2 = [g1.astype(np.float64)], [g2.astype(np.float64)]
    for _ in range(levels - 1):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))
    d = np.zeros(2)
    for lvl in range(levels - 1, -1, -1):
        s = 2 ** lvl
        ix, iy = image_gradients(pyr1[lvl])
        d, G, hist = lk_track_point(pyr1[lvl], pyr2[lvl], ix, iy, (p[0] / s, p[1] / s), half, d0=d)
        if lvl:
            d = d * 2
    return d, G, hist


def measure_by_template(g1, g2, p, half: int = 7, search: int = 25):
    """
    Independent 'actual' location of the patch around p in the next frame:
    normalised cross-correlation over a search window, then sub-pixel
    refinement by fitting a parabola through the peak and its neighbours.
    """
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = g1.shape
    if not (half + search <= x < w - half - search and half + search <= y < h - half - search):
        return None, None
    tpl = g1[y - half:y + half + 1, x - half:x + half + 1].astype(np.float32)
    win = g2[y - half - search:y + half + search + 1, x - half - search:x + half + search + 1].astype(np.float32)
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (mx, my) = cv2.minMaxLoc(r)

    def sub(c_m, c_0, c_p):
        den = c_m - 2 * c_0 + c_p
        return 0.0 if abs(den) < 1e-9 else 0.5 * (c_m - c_p) / den

    ox = sub(r[my, mx - 1], r[my, mx], r[my, mx + 1]) if 0 < mx < r.shape[1] - 1 else 0.0
    oy = sub(r[my - 1, mx], r[my, mx], r[my + 1, mx]) if 0 < my < r.shape[0] - 1 else 0.0
    # sub-pixel offset of the integer start point: p - (x, y)
    actual = np.array([x + (mx + ox - search) + (p[0] - x), y + (my + oy - search) + (p[1] - y)])
    return actual, float(peak)


def track_two_frames(f1_bgr, f2_bgr, n_points: int = 12, half: int = 7, out_dir: str | None = None):
    """
    Validate LK tracking on frames t and t+1:
      - Shi-Tomasi corners (min eigenvalue of G large -> well-conditioned)
      - prediction p + d from our pyramidal LK
      - measured location from template matching, and OpenCV LK for reference.
    """
    g1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2GRAY)
    margin = half + 26
    mask = np.zeros_like(g1)
    mask[margin:-margin, margin:-margin] = 255
    pts = cv2.goodFeaturesToTrack(g1, maxCorners=n_points * 4, qualityLevel=0.01,
                                  minDistance=max(10, g1.shape[1] // 25), mask=mask, blockSize=2 * half + 1)
    if pts is None:
        raise ValueError("No trackable corners found")
    pts = pts.reshape(-1, 2)
    ref, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts.astype(np.float32), None,
                                          winSize=(2 * half + 1, 2 * half + 1), maxLevel=3)
    # Visit moving points first (largest displacement) so the table is not only
    # static background, then the remaining (static) corners.
    order = np.argsort(-np.hypot(*(ref - pts).T) * st.ravel())
    moving = [k for k in order if st[k] and np.hypot(*(ref[k] - pts[k])) > 0.5]
    static = [k for k in order if k not in moving]
    order = moving[: (2 * n_points) // 3] + static
    rows, example = [], None
    for k in order:
        p = pts[k]
        d, G, hist = lk_track_pyramid(g1, g2, p, half=half)
        actual, ncc = measure_by_template(g1, g2, p, half=half)
        if actual is None or ncc < 0.8:
            continue
        pred = p + d
        rows.append({
            "id": len(rows) + 1, "x": p[0], "y": p[1],
            "dx": d[0], "dy": d[1], "pred_x": pred[0], "pred_y": pred[1],
            "actual_x": actual[0], "actual_y": actual[1], "ncc": ncc,
            "error_px": float(np.hypot(*(pred - actual))),
            "opencv_x": float(ref[k, 0]), "opencv_y": float(ref[k, 1]),
            "err_vs_opencv_px": float(np.hypot(*(pred - ref[k]))) if st[k] else float("nan"),
            "lambda_min": float(np.linalg.eigvalsh(G)[0]),
        })
        if example is None and np.hypot(*d) > 0.5:
            example = (p, G, hist)
        if len(rows) >= n_points:
            break

    # Worked 2x2 system for the first moving point, at full resolution, from d = 0
    if example is None:
        example = (np.array([rows[0]["x"], rows[0]["y"]]), None, None)
    p, _, _ = example
    ix, iy = image_gradients(g1)
    d_full, G_full, hist_full = lk_track_point(g1.astype(np.float64), g2.astype(np.float64), ix, iy, p, half)
    hgrid = np.mgrid[-half:half + 1, -half:half + 1]
    qx, qy = p[0] + hgrid[1].ravel(), p[1] + hgrid[0].ravel()
    it0 = bilinear(g2, qx, qy) - bilinear(g1, qx, qy)
    ixw, iyw = bilinear(ix, qx, qy), bilinear(iy, qx, qy)
    b0 = -np.array([np.sum(ixw * it0), np.sum(iyw * it0)])
    system = {"p": p, "G": G_full, "b": b0, "d_first": np.linalg.solve(G_full, b0),
              "iterations": hist_full, "eig": np.linalg.eigvalsh(G_full), "window": 2 * half + 1}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "track_table.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        cv2.imwrite(os.path.join(out_dir, "track_frames.png"), draw_tracks(f1_bgr, f2_bgr, rows))
        with open(os.path.join(out_dir, "track_system.txt"), "w") as fh:
            fh.write(format_system(system))
    return rows, system


def draw_tracks(f1, f2, rows, scale: int = 2):
    a = cv2.resize(f1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(f2, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for r in rows:
        p = (int(r["x"] * scale), int(r["y"] * scale))
        q = (int(r["pred_x"] * scale), int(r["pred_y"] * scale))
        m = (int(r["actual_x"] * scale), int(r["actual_y"] * scale))
        cv2.circle(a, p, 5, (0, 255, 255), 2)
        cv2.putText(a, str(r["id"]), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(b, m, 7, (255, 0, 255), 2)                         # measured (magenta)
        cv2.drawMarker(b, q, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)    # LK prediction (green)
        cv2.arrowedLine(b, p, q, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(b, str(r["id"]), (m[0] + 8, m[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    fs = min(0.7, a.shape[1] / 700)
    for img, txt in ((a, "frame t"), (b, "frame t+1   + LK prediction   o measured")):
        cv2.rectangle(img, (0, 0), (img.shape[1], int(34 * fs / 0.7)), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, int(24 * fs / 0.7)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    return np.hstack([a, b])


def format_system(s) -> str:
    G, b, p = s["G"], s["b"], s["p"]
    lines = [
        f"Point p = ({p[0]:.2f}, {p[1]:.2f}), window {s['window']}x{s['window']}",
        "",
        "G = sum over window [[Ix^2, IxIy], [IxIy, Iy^2]]",
        f"  = [[{G[0, 0]:.1f}, {G[0, 1]:.1f}], [{G[1, 0]:.1f}, {G[1, 1]:.1f}]]",
        f"eigenvalues of G: {s['eig'][0]:.1f}, {s['eig'][1]:.1f}  (both large -> corner, solvable)",
        "",
        "b = -sum over window [Ix It, Iy It]   (It = I2(q) - I1(q), first iteration d = 0)",
        f"  = [{b[0]:.1f}, {b[1]:.1f}]",
        "",
        f"first step d = G^-1 b = [{s['d_first'][0]:.4f}, {s['d_first'][1]:.4f}] px",
        "",
        "iterations (d, rms residual It):",
    ]
    for i, (d, res) in enumerate(s["iterations"], 1):
        lines.append(f"  {i:2d}: d = ({d[0]:8.4f}, {d[1]:8.4f})   rms It = {res:7.3f}")
    d = s["iterations"][-1][0]
    lines.append(f"\nfinal: p' = p + d = ({p[0] + d[0]:.2f}, {p[1] + d[1]:.2f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["flow", "track"])
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="clip start (s)")
    ap.add_argument("--duration", type=float, default=30.0, help="clip length (s)")
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--frame", type=int, default=0, help="track: index of frame t inside the clip")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.mode == "flow":
        rows, max_mag, fps, sample = process_clip(
            a.video, a.out, a.start, a.duration, a.max_side,
            progress=lambda i, n: print(f"\rflow {i}/{n}", end="", flush=True))
        plot_flow_evidence(rows, sample, max_mag, os.path.join(a.out, "flow_evidence.png"),
                           os.path.basename(a.video))
        sp = np.array([r["moving_speed_px"] for r in rows])
        print(f"\n{len(rows)} flow fields @ {fps:.1f} fps; mean moving speed {sp.mean():.2f} px/frame "
              f"({sp.mean() * fps:.1f} px/s). Wrote flow_video.mp4, flow_stats.csv, flow_evidence.png")
    else:
        frames, fps = read_clip(a.video, a.start, a.duration, a.max_side)
        k = min(a.frame, len(frames) - 2)
        rows, system = track_two_frames(frames[k], frames[k + 1], out_dir=a.out)
        print(format_system(system))
        err = np.array([r["error_px"] for r in rows])
        print(f"\n{len(rows)} points: mean |prediction - measured| = {err.mean():.3f} px, max {err.max():.3f} px")


if __name__ == "__main__":
    _main()
