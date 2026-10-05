"""
Module 6 Part B - automatic marker detection for structure from motion.

Plain paper has almost no texture, so SIFT features land on the background
(carpet, table) instead of the object. Instead, the object is marked with dark
dots, and this file finds, in every photo:

  1. the dots          - dark, solid, round blobs (handwritten numbers are thin
                         strokes and are rejected by the shape tests);
  2. which dot is which - a homography search: 4 reference dots are tried against
                         every ordered 4-tuple of candidates and the homography
                         that maps the most reference dots onto candidates wins
                         (the dots lie on a plane, so x_k ~ H x_ref);
  3. the paper corners  - GrabCut segmentation in the reference view, then in
                         every view the predicted edges (H applied to the
                         reference corners) are snapped to the strongest image
                         edge nearby and a straight line is fitted per side;
                         corners = intersections of neighbouring lines.

The result is the observation array used by m6_sfm.run_sfm_points.

Command line (regenerates the stored results shown on the website):
    python modules/m6_markers.py photo1.jpg photo2.jpg photo3.jpg photo4.jpg ^
        --camera assets/m2/camera_params.json --edge-cm 21.59 --out assets/m6/sfm
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import string

import cv2
import numpy as np


# ----------------------------------------------------------------- 1. dot detection
def find_dots(img):
    """Centres (x, y) of dark, solid, round blobs."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    big = max(h, w)
    bg = cv2.medianBlur(g, (big // 40) | 1).astype(np.float32)        # local background
    dark = ((bg - g.astype(np.float32)) > 0.35 * bg).astype(np.uint8)  # much darker than it
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    out = []
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if not (2e-6 * h * w < a < 3e-4 * h * w):
            continue
        m = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        A, P = cv2.contourArea(c), cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        if A < 4 or hull == 0:
            continue
        circularity = 4 * np.pi * A / P ** 2          # 1 for a perfect disc
        solidity = A / hull                           # 1 for a filled convex blob
        aspect = max(ww, hh) / max(1, min(ww, hh))
        if circularity > 0.7 and solidity > 0.9 and aspect < 1.6:
            out.append(cen[i])
    return np.array(out, float).reshape(-1, 2)


# ------------------------------------------------------- 2. which dot is which (H search)
def _batch_homography(src, dst):
    """Homographies mapping the 4 points src (4,2) to each dst[m] (M,4,2) - DLT with h33 = 1."""
    M = dst.shape[0]
    A = np.zeros((M, 8, 8))
    b = np.zeros((M, 8))
    for k in range(4):
        x, y = src[k]
        u, v = dst[:, k, 0], dst[:, k, 1]
        A[:, 2 * k, 0:3] = [x, y, 1]
        A[:, 2 * k, 6], A[:, 2 * k, 7] = -u * x, -u * y
        A[:, 2 * k + 1, 3:6] = [x, y, 1]
        A[:, 2 * k + 1, 6], A[:, 2 * k + 1, 7] = -v * x, -v * y
        b[:, 2 * k], b[:, 2 * k + 1] = u, v
    ok = np.abs(np.linalg.det(A)) > 1e-9
    h = np.zeros((M, 8))
    h[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return np.concatenate([h, np.ones((M, 1))], 1).reshape(M, 3, 3), ok


def match_dots(ref, cand, size, chunk=20000):
    """Assign each reference dot to a candidate in another view. Returns (N,2) with NaN = not found,
    and the homography H (reference -> this view)."""
    tol = 0.02 * size
    hull = cv2.convexHull(ref.astype(np.float32)).reshape(-1, 2)
    src = np.array([hull[np.argmin(hull.sum(1))], hull[np.argmax(hull[:, 0] - hull[:, 1])],
                    hull[np.argmax(hull.sum(1))], hull[np.argmin(hull[:, 0] - hull[:, 1])]], float)
    R = np.c_[ref, np.ones(len(ref))]
    best, Hbest = np.inf, None
    perms = itertools.permutations(range(len(cand)), 4)
    while True:
        p = np.array(list(itertools.islice(perms, chunk)))
        if len(p) == 0:
            break
        H, ok = _batch_homography(src, cand[p])
        P = np.einsum("mij,nj->mni", H, R)
        w = P[..., 2]
        P = P[..., :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)[..., None]
        d = np.linalg.norm(P[:, :, None, :] - cand[None, None], axis=-1).min(-1)
        score = np.minimum(d, tol).sum(1)             # truncated error: unmatched dots cost tol
        score[~ok | (w <= 0).any(1)] = np.inf
        i = int(np.argmin(score))
        if score[i] < best:
            best, Hbest = score[i], H[i]
    if Hbest is None:
        raise RuntimeError("could not match the dots between views")
    for _ in range(3):                                # re-estimate H from all matched dots
        P = cv2.perspectiveTransform(ref[None].astype(np.float64), Hbest)[0]
        d = np.linalg.norm(P[:, None] - cand[None], axis=-1)
        j, good = d.argmin(1), d.min(1) < tol
        if good.sum() >= 4:
            Hbest, _ = cv2.findHomography(ref[good], cand[j[good]], 0)
    out = np.full_like(ref, np.nan)
    out[good] = cand[j[good]]
    return out, Hbest


# --------------------------------------------------------------- 3. paper corners
def _intersect(l1, l2):
    (d1, p1), (d2, p2) = l1, l2
    s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + s[0] * d1


def _order_quad(q):
    """Start at the top-left corner, go clockwise (in image coordinates)."""
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)


def segment_paper(img, dots):
    """Rough paper quadrilateral in one view by GrabCut seeded with the dots."""
    h, w = img.shape[:2]
    big = max(h, w)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    poly = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(poly, cv2.convexHull(dots.astype(np.float32)).astype(np.int32), 1)
    grown = cv2.dilate(poly, np.ones((int(0.06 * big),) * 2, np.uint8))
    mask[grown > 0] = cv2.GC_PR_FGD
    mask[poly > 0] = cv2.GC_FGD
    b = max(3, int(0.01 * big))
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = (cv2.GC_BGD,) * 4
    cv2.grabCut(img, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
    fg = cv2.morphologyEx(((mask == 1) | (mask == 3)).astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    cx, cy = np.round(dots.mean(0)).astype(int)
    i = lab[cy, cx] or 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    q = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(q) != 4:
        q = cv2.boxPoints(cv2.minAreaRect(hull))
    return _order_quad(q.astype(float))


def snap_quad(img, quad, search=0.025):
    """Move each side of a predicted quad onto the strongest nearby edge and fit a line to it.
    Corners outside the image are returned as NaN (not visible)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    h, w = g.shape
    r = int(search * max(h, w))
    offs = np.arange(-r, r + 1)
    lines = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        pts = []
        for t in np.linspace(0.08, 0.92, 80):
            p = a + t * L * d + offs[:, None] * nrm   # samples across the edge
            ok = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
            if ok.sum() < len(offs) * 0.8:
                continue
            p = p[ok]
            xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
            s = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1])  # gradient across the edge
            if s.max() > 3 * np.median(s) + 1:
                pts.append(p[int(np.argmax(s))])
        if len(pts) >= 10:
            vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([vx, vy]), np.array([x0, y0])))
        else:
            lines.append((d, a))
    q = np.array([_intersect(lines[k - 1], lines[k]) for k in range(4)])
    outside = (q[:, 0] < 0) | (q[:, 0] >= w) | (q[:, 1] < 0) | (q[:, 1] >= h)
    q[outside] = np.nan
    return q


# ------------------------------------------------------------------------ all views
def detect_markers(images, ref=0):
    """
    Returns obs (N_points x N_views x 2, NaN = not seen), labels, boundary indices and a dict of
    diagnostics. Dots are numbered 1..N in reading order in the reference view; corners are A-D,
    clockwise from the top-left corner of the reference view (A-B = top edge).
    """
    size = max(images[0].shape[:2])
    dots = [find_dots(im) for im in images]
    R = dots[ref]
    if len(R) < 4:
        raise RuntimeError(f"only {len(R)} dots found in the reference photo (need at least 4)")
    R = R[np.lexsort((R[:, 0], np.round(R[:, 1] / (0.05 * size))))]  # reading order
    obs_d, Hs = [], []
    for k, im in enumerate(images):
        if k == ref:
            obs_d.append(R.copy())
            Hs.append(np.eye(3))
        else:
            m, H = match_dots(R, dots[k], size)
            obs_d.append(m)
            Hs.append(H)
    quad_ref = snap_quad(images[ref], segment_paper(images[ref], R))
    corners = []
    for k, im in enumerate(images):
        if k == ref:
            corners.append(quad_ref)
        else:
            pred = cv2.perspectiveTransform(quad_ref[None], Hs[k])[0]
            corners.append(snap_quad(im, pred))
    nd = len(R)
    obs = np.stack([np.vstack([obs_d[k], corners[k]]) for k in range(len(images))], 1)
    labels = [str(i + 1) for i in range(nd)] + list(string.ascii_uppercase[:4])
    info = {"dots_found": [len(d) for d in dots],
            "dots_matched": [int((~np.isnan(o[:, 0])).sum()) for o in obs_d],
            "corners_visible": [int((~np.isnan(c[:, 0])).sum()) for c in corners]}
    return obs, labels, list(range(nd, nd + 4)), info


def draw_markers(images, obs, labels):
    """Detected points drawn on each photo (for checking)."""
    out = []
    for k, im in enumerate(images):
        v = im.copy()
        t = max(1, max(im.shape[:2]) // 400)
        q = obs[-4:, k]
        if not np.isnan(q).any():
            cv2.polylines(v, [q.astype(np.int32)], True, (0, 165, 255), 2 * t)
        for p, lab in zip(obs[:, k], labels):
            if np.isnan(p[0]):
                continue
            c = tuple(int(round(x)) for x in p)
            cv2.circle(v, c, 6 * t, (255, 0, 255), 2 * t)
            cv2.putText(v, lab, (c[0] + 8 * t, c[1] - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t,
                        (255, 0, 255), 2 * t)
        out.append(v)
    return out


def run_auto(images, K, names, edge_cm=None, out_dir=None):
    """Detect markers automatically, then run the SfM pipeline on them."""
    from modules.m6_sfm import run_sfm_points
    obs, labels, boundary, info = detect_markers(images)
    se = (boundary[0], boundary[1], edge_cm) if edge_cm else None
    r = run_sfm_points(obs, K, names, labels, boundary, se, images, out_dir)
    r["detection"] = info
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        marked = draw_markers(images, obs, labels)
        hmax = max(m.shape[0] for m in marked)
        tiles = [cv2.resize(m, (int(m.shape[1] * 600 / m.shape[0]), 600)) for m in marked]
        cv2.imwrite(os.path.join(out_dir, "detected.png"), np.hstack(tiles))
        json.dump({"labels": labels, "boundary": [labels[i] for i in boundary],
                   "detection": info,
                   "points": {n: [None if np.isnan(p[0]) else [float(p[0]), float(p[1])] for p in obs[:, k]]
                              for k, n in enumerate(names)}},
                  open(os.path.join(out_dir, "detected_points.json"), "w"), indent=1)
    return r


def _main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.m6_sfm import load_camera, report_text
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="photos; the first one is the reference view")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--edge-cm", type=float, help="real length of edge A-B (top edge in photo 1)")
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--out", default="results/sfm")
    a = ap.parse_args()
    imgs = [cv2.imread(p) for p in a.images]
    s = min(1.0, a.max_side / max(imgs[0].shape[:2]))
    imgs = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in imgs]
    K, _ = load_camera(a.camera, imgs[0].shape)
    r = run_auto(imgs, K, [os.path.basename(p) for p in a.images], a.edge_cm, a.out)
    print("detection:", r["detection"], "\n")
    print(report_text(r))


if __name__ == "__main__":
    _main()
