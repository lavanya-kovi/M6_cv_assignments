"""
CSc 8830 Computer Vision - Module 6, Part B: structure from motion (4 views, planar object)
==========================================================================================

README / HOW TO RUN
-------------------
    python modules/m6_sfm.py view1.jpg view2.jpg view3.jpg view4.jpg \
        --camera assets/m2/camera_params.json \
        --corners corners.json --width-cm 23.5 --out results/sfm

    --camera   intrinsics K (+ distortion) from the Module 2 calibration. K is
               rescaled automatically if the photos have a different resolution.
    --corners  optional JSON {"view1.jpg": [[x,y] x 4], ...}: the object's four
               corners, clockwise from top-left, in every view. If given, the
               boundary is triangulated from them and features are restricted to
               the object. Without it, the boundary is the hull of the points.
    --width-cm real length of the first edge (corner 1 -> corner 2) to fix the
               metric scale (SfM recovers shape only up to scale).

Pipeline (every step is written out in the report)
    1. SIFT features + ratio test, matched between EVERY pair of views and
       joined into multi-view tracks (union-find).
    2. Two-view initialisation from the best-matched pair: essential matrix E and homography H
       are both estimated; for a planar scene H explains the matches, so the pose
       comes from decomposing H = K (R + t n^T / d) K^-1, else from E = [t]x R.
    3. Linear (DLT) triangulation of the view-1/2 matches.
    4. Remaining views registered one by one by PnP (2-D/3-D matches), best-connected
       first, re-triangulating after each. Finally everything is expressed in the
       frame of view 1 (camera 1 at the origin).
    5. Multi-view DLT triangulation, then bundle adjustment (all poses + points)
       minimising reprojection error.
    6. Plane fit by SVD, points expressed in plane coordinates, boundary =
       triangulated corners (or convex hull), scaled to cm.
Outputs: sfm_report.txt (all matrices and numbers), matches.png, points3d.png,
         boundary.png, reprojection.png, sfm_result.json
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def load_camera(path: str, image_shape):
    """K and distortion, rescaled/rotated to the image size (h, w)."""
    p = json.load(open(path))
    K, dist = np.array(p["K"], float), np.array(p["dist"], float).ravel()
    h, w = image_shape[:2]
    cw, ch = p.get("image_size", [w, h])
    if (cw > ch) != (w > h):      # calibration in the other orientation: swap axes
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1]])
        cw, ch = ch, cw
    sx, sy = w / cw, h / ch
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K, dist


def projection(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(K, R, t, X):
    x = (K @ (R @ X.T + t.reshape(3, 1))).T
    return x[:, :2] / x[:, 2:3]


# ---------------------------------------------------------------------------
# Features and tracks
# ---------------------------------------------------------------------------
def detect(gray, mask=None, n=4000):
    sift = cv2.SIFT_create(nfeatures=n)
    return sift.detectAndCompute(gray, mask)


def match(d1, d2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def build_tracks(images, masks=None, ratio=0.8):
    """
    Match EVERY pair of views (SIFT + ratio test), keep matches consistent with a
    RANSAC fundamental matrix or homography, and join them into multi-view tracks
    with union-find: two features are in the same track if a chain of verified
    matches links them. Tracks that contain two different features from the same
    view are inconsistent and dropped.
    Returns obs (n_tracks x n_views x 2, NaN where unseen), features, the verified
    matches of every pair {(i, j): (pts_i, pts_j)} and the pair inlier counts.
    """
    nv = len(images)
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    feats = [detect(g, None if masks is None else masks[i], n=8000) for i, g in enumerate(grays)]
    offs = np.cumsum([0] + [len(f[0]) for f in feats])
    parent = list(range(offs[-1]))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    pair_matches, counts = {}, np.zeros((nv, nv), int)
    for i in range(nv):
        for j in range(i + 1, nv):
            (ki, di), (kj, dj) = feats[i], feats[j]
            if di is None or dj is None:
                continue
            m = match(di, dj, ratio)
            if len(m) < 8:
                continue
            a = np.float32([ki[x.queryIdx].pt for x in m])
            b = np.float32([kj[x.trainIdx].pt for x in m])
            _, inl = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.999)
            _, inlh = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
            keep = np.zeros(len(m), bool)
            if inl is not None:
                keep |= inl.ravel() > 0
            if inlh is not None:
                keep |= inlh.ravel() > 0
            for x, k in zip(m, keep):
                if k:
                    ru, rv = find(offs[i] + x.queryIdx), find(offs[j] + x.trainIdx)
                    if ru != rv:
                        parent[ru] = rv
            pair_matches[(i, j)] = (a[keep], b[keep])
            counts[i, j] = counts[j, i] = int(keep.sum())

    groups = {}
    for v in range(nv):
        for k in range(len(feats[v][0])):
            groups.setdefault(find(offs[v] + k), []).append((v, k))
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        views = [v for v, _ in members]
        if len(set(views)) != len(views):
            continue                                   # inconsistent track
        o = np.full((nv, 2), np.nan)
        for v, k in members:
            o[v] = feats[v][0][k].pt
        rows.append(o)
    obs = np.array(rows) if rows else np.zeros((0, nv, 2))
    return obs, feats, pair_matches, counts


# ---------------------------------------------------------------------------
# Two-view geometry
# ---------------------------------------------------------------------------
def triangulate_dlt(Ps, xs):
    """
    Linear triangulation from N views: for each view, x ~ P X gives
        x (p3 . X) - (p1 . X) = 0,   y (p3 . X) - (p2 . X) = 0
    Stack into A X = 0 (2N x 4) and take the right singular vector of the
    smallest singular value. Returns X (3,) and A.
    """
    A = []
    for P, (x, y) in zip(Ps, xs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3], A


def triangulate_all(K, poses, obs):
    """Multi-view DLT for every track using every view that sees it."""
    X = np.full((len(obs), 3), np.nan)
    for i, o in enumerate(obs):
        vs = [v for v in range(len(poses)) if poses[v] is not None and not np.isnan(o[v, 0])]
        if len(vs) >= 2:
            X[i], _ = triangulate_dlt([projection(K, *poses[v]) for v in vs], [o[v] for v in vs])
    return X


def init_two_view(K, x1, x2, thr_e=1.0, thr_h=2.0):
    """
    Relative pose of view 2 w.r.t. view 1.
    E: x2^T E x1 = 0 with E = [t]x R (5-point RANSAC).  H: x2 ~ H x1 with
    H = K (R + t n^T / d) K^-1 for a plane n^T X = d.  For a planar scene most
    matches are H-inliers; we take the model with more inliers and keep the
    decomposition with the most points in front of both cameras.
    """
    E, inE = cv2.findEssentialMat(x1, x2, K, cv2.RANSAC, 0.999, thr_e)
    H, inH = cv2.findHomography(x1, x2, cv2.RANSAC, thr_h)
    nE, nH = int(inE.sum()), int(inH.sum())
    candidates = []
    _, R, t, m = cv2.recoverPose(E, x1, x2, K, mask=inE.copy())
    candidates.append(("E", R, t.ravel()))
    nsol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    for R, t in zip(Rs, ts):
        candidates.append(("H", R, t.ravel()))
    best, best_score = None, -1
    P1 = projection(K, np.eye(3), np.zeros(3))
    sel = inH.ravel() > 0
    for name, R, t in candidates:
        if np.linalg.norm(t) < 1e-9:
            continue
        t = t / np.linalg.norm(t)
        P2 = projection(K, R, t)
        Xh = cv2.triangulatePoints(P1, P2, x1[sel].T, x2[sel].T)
        X = (Xh[:3] / Xh[3]).T
        z1 = X[:, 2]
        z2 = (R @ X.T + t.reshape(3, 1))[2]
        front = np.mean((z1 > 0) & (z2 > 0))
        err = np.median(np.hypot(*(project(K, R, t, X) - x2[sel]).T))
        score = front - 0.01 * err
        if score > best_score:
            best_score, best = score, (name, R, t, front, err)
    name, R, t, front, err = best
    return {"E": E, "H": H, "inliers_E": nE, "inliers_H": nH, "model": name,
            "R": R, "t": t, "front_fraction": float(front), "median_reproj": float(err)}


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------
def bundle_adjust(K, poses, X, obs, fix_first=True):
    """Minimise sum ||x_ij - pi(K, R_j, t_j, X_i)||^2 over poses (views 2..N) and points."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    nv, npnt = len(poses), len(X)
    vis = [(i, v) for i in range(npnt) for v in range(nv) if not np.isnan(obs[i, v, 0])]
    cam0 = 1 if fix_first else 0
    ncam = nv - cam0

    def pack():
        cams = []
        for v in range(cam0, nv):
            r, _ = cv2.Rodrigues(poses[v][0])
            cams.append(np.r_[r.ravel(), poses[v][1]])
        return np.r_[np.ravel(cams), X.ravel()]

    def unpack(p):
        cams = p[:6 * ncam].reshape(ncam, 6)
        P = list(poses[:cam0]) + [(cv2.Rodrigues(c[:3])[0], c[3:]) for c in cams]
        return P, p[6 * ncam:].reshape(npnt, 3)

    def resid(p):
        P, Xs = unpack(p)
        r = np.empty(2 * len(vis))
        for k, (i, v) in enumerate(vis):
            r[2 * k:2 * k + 2] = project(K, P[v][0], P[v][1], Xs[i:i + 1])[0] - obs[i, v]
        return r

    A = lil_matrix((2 * len(vis), 6 * ncam + 3 * npnt), dtype=int)
    for k, (i, v) in enumerate(vis):
        if v >= cam0:
            A[2 * k:2 * k + 2, 6 * (v - cam0):6 * (v - cam0) + 6] = 1
        A[2 * k:2 * k + 2, 6 * ncam + 3 * i:6 * ncam + 3 * i + 3] = 1
    p0 = pack()
    r0 = resid(p0)
    sol = least_squares(resid, p0, jac_sparsity=A, x_scale="jac", method="trf", max_nfev=60)
    P, Xs = unpack(sol.x)
    rms = lambda r: float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))  # noqa: E731
    return P, Xs, rms(r0), rms(sol.fun)


# ---------------------------------------------------------------------------
# Plane and boundary
# ---------------------------------------------------------------------------
def fit_plane(X):
    """Least-squares plane by SVD: normal = singular vector of the smallest singular value."""
    c = X.mean(0)
    _, S, Vt = np.linalg.svd(X - c)
    n = Vt[2]
    e1, e2 = Vt[0], Vt[1]
    return c, n, e1, e2, S


def to_plane(X, c, e1, e2):
    return np.c_[(X - c) @ e1, (X - c) @ e2]


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_sfm(images, K, dist=None, corners=None, width_cm=None, names=None, out_dir=None):
    names = names or [f"view{i + 1}" for i in range(len(images))]
    if dist is not None and np.any(dist):
        images = [cv2.undistort(im, K, dist) for im in images]
        if corners is not None:
            corners = [cv2.undistortPoints(np.float32(c).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
                       for c in corners]
    masks = None
    if corners is not None:
        masks = []
        for im, c in zip(images, corners):
            m = np.zeros(im.shape[:2], np.uint8)
            cv2.fillConvexPoly(m, np.int32(c), 255)
            masks.append(cv2.dilate(m, np.ones((15, 15), np.uint8)))

    obs, feats, pair_matches, counts = build_tracks(images, masks)
    nv = len(images)

    # Initial pair: the pair of views with the most verified matches
    i0, j0 = max(pair_matches, key=lambda k: len(pair_matches[k][0]))
    x1, x2 = pair_matches[(i0, j0)]
    init = init_two_view(K, x1, x2)
    init["pair"] = (i0, j0)

    poses = [None] * nv
    poses[i0] = (np.eye(3), np.zeros(3))
    poses[j0] = (init["R"], init["t"])
    X = triangulate_all(K, poses, obs)

    # Register the remaining views one by one (PnP), best-connected first,
    # re-triangulating after each so later views can use the new points.
    pnp = {}
    while any(p is None for p in poses):
        cand = {v: int((~np.isnan(X[:, 0]) & ~np.isnan(obs[:, v, 0])).sum())
                for v in range(nv) if poses[v] is None}
        v = max(cand, key=cand.get)
        if cand[v] < 6:
            raise ValueError(
                f"Too few 2D-3D matches for view {v + 1} ({cand[v]}). Verified matches between views: "
                + ", ".join(f"{a + 1}-{b + 1}: {counts[a, b]}" for a in range(nv) for b in range(a + 1, nv))
                + ". Retake that photo closer in angle to the others, with the whole object sharp.")
        ok = ~np.isnan(X[:, 0]) & ~np.isnan(obs[:, v, 0])
        okp, rvec, tvec, inl = cv2.solvePnPRansac(X[ok].astype(np.float64), obs[ok, v].astype(np.float64),
                                                  K, None, reprojectionError=4.0, iterationsCount=3000,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not okp:
            raise ValueError(f"PnP failed for view {v + 1}")
        poses[v] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        pnp[v] = int(len(inl)) if inl is not None else 0
        X = triangulate_all(K, poses, obs)

    # Express everything in the frame of view 1 (camera 1 at the origin)
    R0, t0 = poses[0]
    poses = [(R @ R0.T, t - R @ R0.T @ t0) for R, t in poses]

    X = triangulate_all(K, poses, obs)
    good = ~np.isnan(X[:, 0])
    # discard gross outliers (behind camera / huge reprojection error)
    for v in range(len(images)):
        z = (poses[v][0] @ X[good].T + poses[v][1].reshape(3, 1))[2]
        tmp = np.where(good)[0]
        good[tmp[z <= 0]] = False
    # discard points whose reprojection error exceeds 3 px in any view (mismatches)
    for i in np.where(good)[0]:
        for v in range(len(images)):
            if not np.isnan(obs[i, v, 0]):
                e = np.hypot(*(project(K, *poses[v], X[i:i + 1])[0] - obs[i, v]))
                if e > 3.0:
                    good[i] = False
                    break
    obs_g, X_g = obs[good], X[good]
    poses_ba, X_ba, rms0, rms1 = bundle_adjust(K, poses, X_g, obs_g)

    # worked triangulation example: the point seen in the most views
    nseen = (~np.isnan(obs_g[..., 0])).sum(1)
    ex = int(np.argmax(nseen))
    vs = [v for v in range(len(images)) if not np.isnan(obs_g[ex, v, 0])]
    Xex, Aex = triangulate_dlt([projection(K, *poses_ba[v]) for v in vs], [obs_g[ex, v] for v in vs])

    # plane + boundary
    c, n, e1, e2, S = fit_plane(X_ba)
    plane_rms = float(S[2] / np.sqrt(len(X_ba)))
    result = {"names": names, "K": K, "poses": poses_ba, "points": X_ba, "obs": obs_g,
              "init": init, "pnp_inliers": pnp, "pair_counts": counts, "rms_before_ba": rms0, "rms_after_ba": rms1,
              "plane": (c, n, e1, e2), "plane_rms": plane_rms,
              "example": {"point": ex, "views": vs, "A": Aex, "X": Xex}}
    if corners is not None:
        Ps = [projection(K, *p) for p in poses_ba]
        Cw = np.array([triangulate_dlt(Ps, [corners[v][j] for v in range(len(images))])[0] for j in range(4)])
        poly = to_plane(Cw, c, e1, e2)
    else:
        poly2 = to_plane(X_ba, c, e1, e2)
        hull = cv2.convexHull(poly2.astype(np.float32)).reshape(-1, 2)
        poly, Cw = hull.astype(float), None
    edges = np.hypot(*np.diff(np.vstack([poly, poly[:1]]), axis=0).T)
    scale = (width_cm / edges[0]) if width_cm else None
    result.update({"corners_3d": Cw, "boundary_plane": poly, "edges_units": edges, "scale_cm_per_unit": scale})
    if scale:
        result["edges_cm"] = edges * scale
        result["area_cm2"] = polygon_area(poly) * scale ** 2
        result["points_cm"] = X_ba * scale
        result["camera_centres_cm"] = [(-p[0].T @ p[1]) * scale for p in poses_ba]
    if out_dir:
        result["pair_counts"] = counts
        save_outputs(result, images, feats, pair_matches, out_dir)
    return result


def run_sfm_points(obs, K, names, labels, boundary=None, scale_edge=None, images=None, out_dir=None,
                   thr_px=6.0):
    """
    Structure from motion from KNOWN point correspondences (e.g. numbered marker
    dots and paper corners): obs is N_points x N_views x 2 (pixels, NaN = unseen).
    Same pipeline as run_sfm: best two-view initialisation (E or H), DLT
    triangulation, PnP for the other views, multi-view triangulation, bundle
    adjustment, plane fit. boundary = indices of the points forming the object's
    outline (in order); scale_edge = (i, j, length_cm) fixes the metric scale.
    """
    nv = obs.shape[1]

    def complete(i0, j0, R, t):
        """Finish the reconstruction from one candidate initial pose; returns (rms, poses, X, rms0, pnp)."""
        poses = [None] * nv
        poses[i0] = (np.eye(3), np.zeros(3))
        poses[j0] = (R, t / np.linalg.norm(t))
        X = triangulate_all(K, poses, obs)
        pnp = {}
        while any(p is None for p in poses):
            cand = {v: int((~np.isnan(X[:, 0]) & ~np.isnan(obs[:, v, 0])).sum()) for v in range(nv) if poses[v] is None}
            v = max(cand, key=cand.get)
            ok = ~np.isnan(X[:, 0]) & ~np.isnan(obs[:, v, 0])
            okp, rvec, tvec, inl = cv2.solvePnPRansac(X[ok].astype(np.float64), obs[ok, v].astype(np.float64), K,
                                                      None, reprojectionError=thr_px * 2, iterationsCount=2000,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if not okp:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(X[ok].astype(np.float64), obs[ok, v].astype(np.float64), K, None,
                                              rvec, tvec)
            poses[v] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
            pnp[v] = int(len(inl)) if inl is not None else int(ok.sum())
            X = triangulate_all(K, poses, obs)
        R0, t0 = poses[0]
        poses = [(Rv @ R0.T, tv - Rv @ R0.T @ t0) for Rv, tv in poses]
        X = triangulate_all(K, poses, obs)
        if np.isnan(X).any():
            return None
        zs = [(Rv @ X.T + tv.reshape(3, 1))[2] for Rv, tv in poses]
        if min(z.min() for z in zs) <= 0:            # cheirality: every point in front of every camera
            return None
        poses_ba, X_ba, rms0, rms1 = bundle_adjust(K, poses, X, obs)
        return rms1, poses_ba, X_ba, rms0, pnp

    # Try every pair and every candidate pose (E -> recoverPose, H -> up to 4 decompositions);
    # keep the reconstruction with the lowest final reprojection error. For a planar object the
    # E solution is ambiguous, so this also selects between the E and H models.
    best, tried = None, []
    for i in range(nv):
        for j in range(i + 1, nv):
            ok = ~np.isnan(obs[:, i, 0]) & ~np.isnan(obs[:, j, 0])
            if ok.sum() < 6:
                continue
            x1, x2 = obs[ok, i].astype(np.float64), obs[ok, j].astype(np.float64)
            init = init_two_view(K, x1, x2, thr_e=thr_px / 2, thr_h=thr_px)
            cands = []
            _, Re, te, _ = cv2.recoverPose(init["E"], x1, x2, K)
            cands.append(("E", Re, te.ravel()))
            _, Rs, ts, _ = cv2.decomposeHomographyMat(init["H"], K)
            cands += [("H", Rh, th.ravel()) for Rh, th in zip(Rs, ts) if np.linalg.norm(th) > 1e-9]
            for model, Rc, tc in cands:
                res = complete(i, j, Rc, tc)
                tried.append((i + 1, j + 1, model, None if res is None else res[0]))
                if res is not None and (best is None or res[0] < best[0]):
                    init_sel = dict(init, model=model, R=Rc, t=tc / np.linalg.norm(tc), pair=(i, j))
                    best = (res[0], res, init_sel)
    _, (rms1, poses_ba, X_ba, rms0, pnp), init = best
    init["candidates_tried"] = tried
    # per-point reprojection errors after BA
    per_view = []
    for v in range(nv):
        e = np.hypot(*(project(K, *poses_ba[v], X_ba) - obs[:, v]).T)
        per_view.append(e)
    ex = 0
    vs = [v for v in range(nv) if not np.isnan(obs[ex, v, 0])]
    Xex, Aex = triangulate_dlt([projection(K, *poses_ba[v]) for v in vs], [obs[ex, v] for v in vs])
    c, n, e1, e2, S = fit_plane(X_ba)
    if boundary is not None and len(boundary) >= 4:
        # align the in-plane axes with the object: e1 along the first boundary edge, and
        # e2 chosen so the object reads like the reference photo (first corner top-left)
        d1 = X_ba[boundary[1]] - X_ba[boundary[0]]
        e1 = d1 - (d1 @ n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        if (X_ba[boundary[-1]] - X_ba[boundary[0]]) @ e2 > 0:
            e2 = -e2
    r = {"names": names, "labels": labels, "K": K, "poses": poses_ba, "points": X_ba, "obs": obs, "init": init,
         "pnp_inliers": pnp, "rms_before_ba": rms0, "rms_after_ba": rms1, "plane": (c, n, e1, e2),
         "plane_rms": float(S[2] / np.sqrt(len(X_ba))), "reproj_per_view": per_view,
         "example": {"point": ex, "views": vs, "A": Aex, "X": Xex}, "pair_counts": None}
    bidx = boundary if boundary is not None else list(range(len(X_ba)))
    poly = to_plane(X_ba[bidx], c, e1, e2)
    edges = np.hypot(*np.diff(np.vstack([poly, poly[:1]]), axis=0).T)
    sc = None
    if scale_edge:
        a, b, L = scale_edge
        sc = L / np.linalg.norm(X_ba[a] - X_ba[b])
    r.update({"corners_3d": X_ba[bidx], "boundary_plane": poly, "edges_units": edges, "scale_cm_per_unit": sc,
              "boundary_idx": bidx})
    if sc:
        r["edges_cm"] = edges * sc
        r["area_cm2"] = polygon_area(poly) * sc ** 2
        r["points_cm"] = X_ba * sc
        r["camera_centres_cm"] = [(-p[0].T @ p[1]) * sc for p in poses_ba]
    if out_dir and images is not None:
        save_outputs_points(r, images, out_dir)
    return r


def save_outputs_points(r, images, out_dir):
    """Figures for the marker-point reconstruction (same style as save_outputs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sfm_report.txt"), "w") as fh:
        fh.write(report_text(r))
    K, sc = r["K"], r.get("scale_cm_per_unit") or 1.0
    unit = "cm" if r.get("scale_cm_per_unit") else "units"
    X = r["points"] * sc
    lab = r["labels"]
    # 3-D
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=25, c="tab:blue")
    for k, p in enumerate(X):
        ax.text(*p, f" {lab[k]}", fontsize=8)
    B = r["corners_3d"] * sc
    ax.plot(*np.vstack([B, B[:1]]).T, color="orange", lw=2)
    span = np.ptp(X, axis=0).max()
    for v, (R, t) in enumerate(r["poses"]):
        C = (-R.T @ t) * sc
        z = R.T @ np.array([0, 0, 1.0])
        ax.scatter(*C, color="red", s=30)
        ax.quiver(*C, *(z * span * 0.25), color="red")
        ax.text(*C, f" C{v + 1}", color="red")
    ax.set_xlabel(f"X ({unit})"); ax.set_ylabel(f"Y ({unit})"); ax.set_zlabel(f"Z ({unit})")
    ax.set_title("Reconstructed points and camera centres (camera 1 at origin)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "points3d.png"), dpi=130); plt.close(fig)
    # plane coordinates
    c, n, e1, e2 = r["plane"]
    P2 = to_plane(r["points"], c, e1, e2) * sc
    poly = r["boundary_plane"] * sc
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    ax.plot(*np.vstack([poly, poly[:1]]).T, color="orange", lw=2, label="estimated boundary (paper edges)")
    ax.scatter(P2[:, 0], P2[:, 1], s=30, color="tab:blue", zorder=3, label="reconstructed points")
    for k, p in enumerate(P2):
        ax.annotate(lab[k], p, xytext=(4, 4), textcoords="offset points", fontsize=9)
    if "edges_cm" in r:
        for (p, q), L in zip(zip(poly, np.roll(poly, -1, 0)), r["edges_cm"]):
            ax.annotate(f"{L:.2f} cm", (p + q) / 2, ha="center", fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_aspect("equal"); ax.set_xlabel(f"plane axis 1 ({unit})"); ax.set_ylabel(f"plane axis 2 ({unit})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    ax.set_title("Object recovered on its fitted plane")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "boundary.png"), dpi=130); plt.close(fig)
    # re-projection into each view + input views
    tiles, tiles_in = [], []
    for v, im in enumerate(images):
        R, t = r["poses"][v]
        o = im.copy(); th = max(4, im.shape[1] // 300)
        q = project(K, R, t, r["corners_3d"])
        cv2.polylines(o, [np.int32(q)], True, (0, 165, 255), th)
        pr = project(K, R, t, r["points"])
        for k, (p, m) in enumerate(zip(pr, r["obs"][:, v])):
            if not np.isnan(m[0]):
                cv2.circle(o, tuple(np.int32(m)), 6 * th, (255, 0, 255), th)      # measured
            cv2.drawMarker(o, tuple(np.int32(p)), (0, 255, 0), cv2.MARKER_CROSS, 10 * th, th)  # reprojected
        cv2.putText(o, f"view {v + 1}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 12)
        w = 420
        tiles.append(cv2.resize(o, (w, int(w * im.shape[0] / im.shape[1]))))
        oi = im.copy()
        cv2.putText(oi, f"view {v + 1}: {r['names'][v]}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
        tiles_in.append(cv2.resize(oi, (w, int(w * im.shape[0] / im.shape[1]))))
    for name, tl in (("reprojection.png", tiles), ("views.png", tiles_in)):
        hmax = max(t.shape[0] for t in tl)
        tl = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tl]
        cv2.imwrite(os.path.join(out_dir, name), np.hstack(tl))
    out = {"names": r["names"], "labels": lab, "K": r["K"].tolist(),
           "poses": [{"R": R.tolist(), "t": t.tolist(), "C": (-R.T @ t).tolist()} for R, t in r["poses"]],
           "points": r["points"].tolist(), "rms_before_ba": r["rms_before_ba"], "rms_after_ba": r["rms_after_ba"],
           "boundary_plane": r["boundary_plane"].tolist(), "edges_units": r["edges_units"].tolist(),
           "scale_cm_per_unit": r["scale_cm_per_unit"],
           "edges_cm": r["edges_cm"].tolist() if "edges_cm" in r else None,
           "points_cm": r["points_cm"].tolist() if "points_cm" in r else None}
    json.dump(out, open(os.path.join(out_dir, "sfm_result.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(M, p=4):
    return np.array2string(np.asarray(M), precision=p, suppress_small=True, max_line_width=120)


def report_text(r) -> str:
    L = []
    K, init = r["K"], r["init"]
    L += ["=== Intrinsics K (from Module 2 calibration, rescaled) ===", fmt(K, 2), ""]
    i0, j0 = init.get("pair", (0, 1))
    pc = r.get("pair_counts")
    if pc is not None:
        L += ["=== Verified feature matches between views ===",
              ", ".join(f"{a + 1}-{b + 1}: {pc[a, b]}" for a in range(len(pc)) for b in range(a + 1, len(pc))), ""]
    L += [f"=== Two-view initialisation (view {i0 + 1} -> view {j0 + 1}, the best-matched pair) ===",
          f"E-inliers {init['inliers_E']}, H-inliers {init['inliers_H']}  -> model used: {init['model']}",
          "E =", fmt(init["E"]), "H =", fmt(init["H"]),
          f"points in front of both cameras: {100 * init['front_fraction']:.1f}%", ""]
    for v, (R, t) in enumerate(r["poses"]):
        C = -R.T @ t
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        L += [f"=== View {v + 1} ({r['names'][v]}) ===", "R =", fmt(R), f"t = {fmt(t)}",
              f"camera centre C = -R^T t = {fmt(C)}" + (f"  = {fmt(C * r['scale_cm_per_unit'], 1)} cm"
                                                         if r.get("scale_cm_per_unit") else ""),
              f"rotation from view 1: {ang:.2f} deg"
              + (f";  PnP inliers {r['pnp_inliers'][v]}" if v in r["pnp_inliers"] else ""), ""]
    ex = r["example"]
    L += ["=== Worked triangulation (DLT) for one point ===",
          f"point #{ex['point']} seen in views {[v + 1 for v in ex['views']]}"]
    for v in ex["views"]:
        L.append(f"  x_{v + 1} = {fmt(r['obs'][ex['point'], v], 2)}")
    L += ["A (rows x*p3 - p1, y*p3 - p2 for each view) =", fmt(ex["A"], 3),
          f"X = last right-singular vector of A, dehomogenised = {fmt(ex['X'])}", ""]
    L += ["=== Reconstruction ===",
          f"{len(r['points'])} 3-D points;  reprojection RMS before BA {r['rms_before_ba']:.3f} px, "
          f"after BA {r['rms_after_ba']:.3f} px",
          f"plane normal n = {fmt(r['plane'][1])};  RMS distance of points to plane = "
          f"{r['plane_rms']:.5f} units ({100 * r['plane_rms'] / np.ptp(r['boundary_plane'][:, 0]):.2f}% of object width)",
          ""]
    L += ["=== Boundary (plane coordinates) ===", fmt(r["boundary_plane"]),
          f"edge lengths (units): {fmt(r['edges_units'])}"]
    if r.get("scale_cm_per_unit"):
        L += [f"scale: {r['scale_cm_per_unit']:.3f} cm per unit (from the known first edge)",
              f"edge lengths (cm): {fmt(r['edges_cm'], 2)}",
              f"area: {r['area_cm2']:.1f} cm^2"]
    return "\n".join(L)


def save_outputs(r, images, feats, pair_matches, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sfm_report.txt"), "w") as fh:
        fh.write(report_text(r))

    # matches of the initial pair
    i0, j0 = r["init"].get("pair", (0, 1))
    a, b = pair_matches[(i0, j0)]
    i1, i2 = images[i0], images[j0]
    h = max(i1.shape[0], i2.shape[0])
    canvas = np.zeros((h, i1.shape[1] + i2.shape[1], 3), np.uint8)
    canvas[:i1.shape[0], :i1.shape[1]] = i1
    canvas[:i2.shape[0], i1.shape[1]:] = i2
    rng = np.random.default_rng(0)
    for k in rng.choice(len(a), min(80, len(a)), replace=False):
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        p, q = a[k], b[k] + [i1.shape[1], 0]
        cv2.line(canvas, tuple(np.int32(p)), tuple(np.int32(q)), col, 2, cv2.LINE_AA)
    s = 1400 / canvas.shape[1]
    cv2.imwrite(os.path.join(out_dir, "matches.png"), cv2.resize(canvas, None, fx=s, fy=s))

    # 3-D points + cameras
    X = r.get("points_cm", r["points"])
    unit = "cm" if "points_cm" in r else "units"
    sc = r.get("scale_cm_per_unit") or 1.0
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=X[:, 2], cmap="viridis")
    span = np.ptp(X, axis=0).max()
    for v, (R, t) in enumerate(r["poses"]):
        C = (-R.T @ t) * sc
        z = R.T @ np.array([0, 0, 1.0])
        ax.scatter(*C, color="red", s=30)
        ax.quiver(*C, *(z * span * 0.3), color="red")
        ax.text(*C, f" C{v + 1}", color="red")
    if r["corners_3d"] is not None:
        Cw = r["corners_3d"] * sc
        ax.plot(*np.vstack([Cw, Cw[:1]]).T, color="orange", lw=2)
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title("Reconstructed points and camera centres")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "points3d.png"), dpi=130)
    plt.close(fig)

    # boundary in plane coordinates
    c, n, e1, e2 = r["plane"]
    P2 = to_plane(r["points"], c, e1, e2) * sc
    poly = r["boundary_plane"] * sc
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P2[:, 0], P2[:, 1], s=2, color="steelblue", label="reconstructed points")
    ax.plot(*np.vstack([poly, poly[:1]]).T, color="orange", lw=2, label="estimated boundary")
    if "edges_cm" in r:
        for (p, q), L in zip(zip(poly, np.roll(poly, -1, 0)), r["edges_cm"]):
            ax.annotate(f"{L:.1f} cm", (p + q) / 2, ha="center", fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel(f"plane axis 1 ({unit})")
    ax.set_ylabel(f"plane axis 2 ({unit})")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Object recovered on its fitted plane")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boundary.png"), dpi=130)
    plt.close(fig)

    # reprojection of boundary + points into every view
    tiles = []
    for v, im in enumerate(images):
        R, t = r["poses"][v]
        o = im.copy()
        pts = project(r["K"], R, t, r["points"])
        for p in pts[:: max(1, len(pts) // 400)]:
            cv2.circle(o, tuple(np.int32(p)), 4, (255, 200, 0), -1)
        if r["corners_3d"] is not None:
            q = project(r["K"], R, t, r["corners_3d"])
            cv2.polylines(o, [np.int32(q)], True, (0, 165, 255), max(3, im.shape[1] // 300))
        cv2.putText(o, f"view {v + 1}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        tiles.append(cv2.resize(o, (400, int(400 * im.shape[0] / im.shape[1]))))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(os.path.join(out_dir, "reprojection.png"), np.hstack(tiles))

    # the four input views, labelled
    tiles = []
    for v, im in enumerate(images):
        o = im.copy()
        cv2.putText(o, f"view {v + 1}: {r['names'][v]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 255), 4)
        tiles.append(cv2.resize(o, (400, int(400 * im.shape[0] / im.shape[1]))))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(os.path.join(out_dir, "views.png"), np.hstack(tiles))

    out = {"names": r["names"], "K": r["K"].tolist(),
           "poses": [{"R": R.tolist(), "t": t.tolist(), "C": (-R.T @ t).tolist()} for R, t in r["poses"]],
           "rms_before_ba": r["rms_before_ba"], "rms_after_ba": r["rms_after_ba"],
           "boundary_plane": r["boundary_plane"].tolist(), "edges_units": r["edges_units"].tolist(),
           "scale_cm_per_unit": r["scale_cm_per_unit"],
           "edges_cm": r["edges_cm"].tolist() if "edges_cm" in r else None}
    json.dump(out, open(os.path.join(out_dir, "sfm_result.json"), "w"), indent=2)


def run_from_points_json(cfg, K, names, images=None, out_dir=None, scale=1.0):
    """Run run_sfm_points from a points JSON ({labels, boundary, scale_edge, points:{photo: [[x,y],..]}}).
    names = the photos to use (keys of cfg["points"]); scale = factor applied to the stored pixel
    coordinates when the images were resized (K must already match the resized images)."""
    labels = [str(l) for l in cfg["labels"]]
    idx = {l: i for i, l in enumerate(labels)}
    obs = np.full((len(labels), len(names), 2), np.nan)
    for j, n in enumerate(names):
        for i, xy in enumerate(cfg["points"][n]):
            if xy is not None:
                obs[i, j] = np.array(xy, float) * scale
    boundary = [idx[str(b)] for b in cfg["boundary"]] if cfg.get("boundary") else None
    se = cfg.get("scale_edge")
    scale_edge = (idx[str(se[0])], idx[str(se[1])], float(se[2])) if se else None
    return run_sfm_points(obs, K, names, labels, boundary, scale_edge, images, out_dir)


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs=4)
    ap.add_argument("--points", help="points JSON with marked correspondences (recommended for plain objects)")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--corners")
    ap.add_argument("--width-cm", type=float)
    ap.add_argument("--max-side", type=int, default=1600, help="downscale photos for speed (K follows)")
    ap.add_argument("--out", default="results/sfm")
    a = ap.parse_args()

    imgs = [cv2.imread(p) for p in a.images]
    if a.points:
        # marker-dot pipeline at full resolution; distortion is NOT applied (see report)
        K, _ = load_camera(a.camera, imgs[0].shape)
        r = run_from_points_json(json.load(open(a.points)), K, [os.path.basename(p) for p in a.images],
                                 imgs, a.out)
        print(report_text(r))
        return
    s = min(1.0, a.max_side / max(imgs[0].shape[:2]))
    full = imgs[0].shape
    imgs = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in imgs]
    K, dist = load_camera(a.camera, imgs[0].shape)
    corners = None
    if a.corners:
        cj = json.load(open(a.corners))
        corners = [np.array(cj[os.path.basename(p)], float) * s for p in a.images]
    r = run_sfm(imgs, K, None, corners, a.width_cm, [os.path.basename(p) for p in a.images], a.out)
    print(f"(photos {full[1]}x{full[0]} processed at {imgs[0].shape[1]}x{imgs[0].shape[0]})\n")
    print(report_text(r))


if __name__ == "__main__":
    _main()
