"""
CSc 8830 Computer Vision - Module 4: Human boundary segmentation (classical, no ML/DL)
=====================================================================================

README / HOW TO RUN
-------------------
This file holds the algorithms only. It is used by the Streamlit page
`pages/4_Module_4_Human_Segmentation.py`, but can also be run on its own:

    # RGB image, person inside rectangle x,y,w,h (pixels)
    python modules/m4_segmentation.py rgb  samples/rgb/person.jpg --rect 50 20 300 460

    # Thermal image (white-hot by default)
    python modules/m4_segmentation.py thermal samples/thermal/person.jpg

    # Compare against a SAM2 mask (binary PNG, white = person)
    python modules/m4_segmentation.py thermal img.jpg --sam2 assets/sam2_masks/img.png

Outputs <name>_mask.png and <name>_overlay.png next to the input image and
prints IoU / Dice / precision / recall / boundary-F when a SAM2 mask is given.

Only OpenCV + NumPy image-processing functions are used (GrabCut is an
iterative graph-cut energy minimisation with GMM colour models, not a learned
model; Otsu, morphology, watershed and contours are purely classical).
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask (0/255) by flood-filling the background."""
    h, w = mask.shape
    # Pad so the flood fill starts from guaranteed background.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes)


def keep_components(mask: np.ndarray, min_area_frac: float = 0.01,
                    largest_only: bool = False) -> np.ndarray:
    """Keep connected components whose area >= min_area_frac of the image."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    out = np.zeros_like(mask)
    if largest_only:
        out[labels == 1 + int(np.argmax(areas))] = 255
        return out
    min_area = min_area_frac * mask.size
    for i, a in enumerate(areas, start=1):
        if a >= min_area:
            out[labels == i] = 255
    return out


def smooth_mask(mask: np.ndarray, open_k: int = 5, close_k: int = 9) -> np.ndarray:
    """Morphological opening (remove specks) then closing (bridge small gaps)."""
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def boundary_contours(mask: np.ndarray):
    """Exact outer boundary of each region (CHAIN_APPROX_NONE keeps every pixel)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 0), alpha: float = 0.35, thickness: int = 2) -> np.ndarray:
    """Tint the mask region and draw its boundary on top of the image."""
    out = img_bgr.copy()
    tint = np.zeros_like(out)
    tint[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)[m]
    cv2.drawContours(out, boundary_contours(mask), -1, color, thickness)
    return out


# ---------------------------------------------------------------------------
# Q1: RGB camera  ->  GrabCut + morphology + contour
# ---------------------------------------------------------------------------
def segment_rgb(img_bgr: np.ndarray, rect: tuple[int, int, int, int],
                iterations: int = 5, open_k: int = 5, close_k: int = 11,
                largest_only: bool = False, fg_hint: bool = True,
                max_side: int = 800) -> np.ndarray:
    """
    Segment the person inside `rect` (x, y, w, h) of an RGB/BGR image.

    Steps
      1. Downscale for speed (GrabCut is O(pixels * iterations)).
      2. GrabCut: pixels outside rect = sure background; inside = "probably
         foreground". Two 5-component GMMs model FG/BG colour, and a min-cut
         over the pixel graph (data term + contrast-sensitive smoothness term)
         labels every pixel. Repeated `iterations` times.
      3. Morphological open/close, keep the largest blob, fill holes.
      4. Upscale back and snap edges to the original resolution.
    Returns a uint8 mask (255 = person).
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else img_bgr.copy()
    x, y, rw, rh = [int(round(v * scale)) for v in rect]
    sh, sw = small.shape[:2]
    x, y = max(0, x), max(0, y)
    rw, rh = max(2, min(rw, sw - x - 1)), max(2, min(rh, sh - y - 1))

    gc_mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    # GrabCut initialises its colour GMMs with k-means, which draws from
    # OpenCV's global random generator. Fix the seed so the same image and box
    # always give the same mask (reproducible results in the report and app).
    cv2.setRNGSeed(0)
    cv2.grabCut(small, gc_mask, (x, y, rw, rh), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)

    if fg_hint:
        # Gap-bridging pass. If the first pass split the person into pieces
        # (e.g. head separated from torso by a dark collar), force a thin
        # vertical band to sure-foreground, but ONLY between the topmost and
        # bottommost first-pass foreground pixels on that band. The band is
        # centred on the foreground's own centre of mass, not the box centre,
        # so it never extends into the sky/ground around an off-centre person.
        first_fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        ys, xs = np.nonzero(first_fg)
        n_parts = cv2.connectedComponents(first_fg.astype(np.uint8))[0] - 1
        if len(xs) and n_parts > 1:
            half = max(2, int(0.05 * rw))
            cx = int(np.median(xs))
            cx0, cx1 = max(x, cx - half), min(x + rw, cx + half)
            rows = np.nonzero(first_fg[:, cx0:cx1].any(axis=1))[0]
            if len(rows):
                forced = np.zeros_like(first_fg)
                forced[rows.min():rows.max() + 1, cx0:cx1] = True
                forced &= ~first_fg & (gc_mask != cv2.GC_BGD)
                gc_mask[forced] = cv2.GC_FGD
                cv2.grabCut(small, gc_mask, None, bgd, fgd, max(1, iterations // 2),
                            cv2.GC_INIT_WITH_MASK)
                # The forced strip is only a constraint, not evidence: hand it
                # back to the colour models, but keep that only if the person
                # does not fall apart into more pieces again.
                def parts(gm):
                    # count only regions that survive the final clean-up
                    fg = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8) * 255
                    fg = keep_components(smooth_mask(fg, open_k, close_k), 0.01 * (rw * rh) / (sw * sh))
                    return cv2.connectedComponents((fg > 0).astype(np.uint8))[0] - 1
                bridged = gc_mask.copy()
                gc_mask[forced] = cv2.GC_PR_FGD
                cv2.grabCut(small, gc_mask, None, bgd.copy(), fgd.copy(), 1, cv2.GC_INIT_WITH_MASK)
                if parts(gc_mask) > parts(bridged):
                    gc_mask = bridged

    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = smooth_mask(mask, open_k, close_k)
    mask = keep_components(mask, 0.01 * (rw * rh) / (sw * sh), largest_only)
    mask = fill_holes(mask)

    if scale < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Q2: Thermal camera  ->  intensity thresholding + morphology (+ watershed)
# ---------------------------------------------------------------------------
def thermal_to_intensity(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a thermal frame to a single 'temperature-like' channel.
    Grayscale thermal images are used directly. Pseudo-colour palettes
    (ironbow / inferno) map heat to brightness, so the HSV V channel is a
    good monotonic proxy.
    """
    if img_bgr.ndim == 2:
        return img_bgr
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    if np.abs(b - g).mean() < 3 and np.abs(g - r).mean() < 3:   # already grey
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]


def segment_thermal(img_bgr: np.ndarray, polarity: str = "white-hot",
                    method: str = "otsu", manual_thresh: int = 128,
                    use_clahe: bool = True, blur_k: int = 5,
                    open_k: int = 5, close_k: int = 9,
                    min_area_frac: float = 0.005, use_watershed: bool = False,
                    peak_frac: float = 0.6,
                    return_steps: bool = False):
    """
    Segment warm human bodies in a thermal image.

    Steps
      1. Intensity channel (white-hot: hot = bright; black-hot is inverted).
      2. CLAHE to boost local contrast, Gaussian blur to suppress sensor noise.
      3. Global threshold: Otsu (maximises between-class variance), Triangle,
         or a manual value.
      4. Open/close, drop tiny blobs, fill holes.
      5. Optional distance-transform watershed to split touching people.
    """
    gray = thermal_to_intensity(img_bgr)
    if polarity == "black-hot":
        gray = 255 - gray
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if use_clahe else gray
    if blur_k > 1:
        blur_k = blur_k | 1
        enh = cv2.GaussianBlur(enh, (blur_k, blur_k), 0)

    if method == "otsu":
        t, th = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "triangle":
        t, th = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        t, th = cv2.threshold(enh, manual_thresh, 255, cv2.THRESH_BINARY)

    mask = smooth_mask(th, open_k, close_k)
    mask = keep_components(mask, min_area_frac)
    mask = fill_holes(mask)

    if use_watershed and mask.any():
        mask = watershed_split(mask, peak_frac)

    if return_steps:
        return mask, {"intensity": gray, "enhanced": enh, "threshold": th, "threshold_value": float(t)}
    return mask


def watershed_split(mask: np.ndarray, peak_frac: float = 0.6) -> np.ndarray:
    """
    Split touching blobs (e.g. two people side by side).
    Markers = peaks of the distance transform; the watershed is flooded on the
    inverted distance map, so cuts fall along the narrow 'necks' between blobs
    and the outer boundary is never moved. Watershed lines (1 px) separate people.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, peak_frac * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                      # background label = 1
    markers[(mask > 0) & (sure_fg == 0)] = 0   # unknown = inside mask, not a peak
    relief = cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
    # Give every in-mask pixel the label of its nearest person basin (fills the
    # 1-px watershed lines and any pixels the background basin grabbed).
    people = np.where(markers > 1, markers, 0).astype(np.int32)
    if people.max() <= 2:          # only one person found -> nothing to split
        return mask
    _, nearest = cv2.distanceTransformWithLabels((people == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(people)
    lut = np.zeros(nearest.max() + 1, np.int32)
    lut[nearest[ys, xs]] = people[ys, xs]
    full = np.where(mask > 0, lut[nearest], 0).astype(np.float32)
    # Cut the mask where two different people meet (never along the outer edge).
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(full, k)
    lo = cv2.erode(np.where(full == 0, 1e6, full).astype(np.float32), k)
    split = (full > 0) & (lo < 1e6) & (hi != lo)
    out = mask.copy()
    out[split] = 0
    return out


# ---------------------------------------------------------------------------
# Comparison with SAM2
# ---------------------------------------------------------------------------
def load_binary_mask(path_or_array, shape=None) -> np.ndarray:
    m = cv2.imread(path_or_array, cv2.IMREAD_GRAYSCALE) if isinstance(path_or_array, str) else path_or_array
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if shape is not None and m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def compare_masks(pred: np.ndarray, ref: np.ndarray, tol: int = 3) -> dict:
    """
    Region metrics (IoU, Dice, precision, recall) plus boundary F-score:
    a boundary pixel counts as matched if it lies within `tol` px of the
    other mask's boundary.
    """
    p, r = pred > 0, ref > 0
    tp = np.logical_and(p, r).sum()
    fp = np.logical_and(p, ~r).sum()
    fn = np.logical_and(~p, r).sum()
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    def edge(m):
        return cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    ep, er = edge(p), edge(r)
    dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)
    bp = (dt_r[ep] <= tol).mean() if ep.any() else 0.0
    br = (dt_p[er] <= tol).mean() if er.any() else 0.0
    bf = 2 * bp * br / (bp + br) if (bp + br) else 0.0
    return {"IoU": float(iou), "Dice": float(dice), "Precision": float(prec),
            "Recall": float(rec), f"Boundary F (±{tol}px)": float(bf)}


def diff_visual(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Green = both agree, red = only ours (false positive), blue = only SAM2 (missed)."""
    out = np.zeros((*pred.shape, 3), np.uint8)
    p, r = pred > 0, ref > 0
    out[p & r] = (0, 200, 0)
    out[p & ~r] = (0, 0, 255)
    out[~p & r] = (255, 80, 0)
    return out


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rgb", "thermal"])
    ap.add_argument("image")
    ap.add_argument("--rect", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="RGB only: box around the person (default: 5%% margin)")
    ap.add_argument("--polarity", default="white-hot", choices=["white-hot", "black-hot"])
    ap.add_argument("--sam2", help="path to a SAM2 binary mask for comparison")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"Cannot read {a.image}")
    h, w = img.shape[:2]
    if a.mode == "rgb":
        rect = tuple(a.rect) if a.rect else (int(0.05 * w), int(0.05 * h), int(0.9 * w), int(0.9 * h))
        mask = segment_rgb(img, rect)
    else:
        mask = segment_thermal(img, polarity=a.polarity)

    stem = os.path.splitext(a.image)[0]
    cv2.imwrite(stem + "_mask.png", mask)
    cv2.imwrite(stem + "_overlay.png", draw_overlay(img, mask))
    print("Saved", stem + "_mask.png", "and", stem + "_overlay.png")
    if a.sam2:
        for k, v in compare_masks(mask, load_binary_mask(a.sam2, mask.shape)).items():
            print(f"{k:>22}: {v:.4f}")


if __name__ == "__main__":
    _main()
