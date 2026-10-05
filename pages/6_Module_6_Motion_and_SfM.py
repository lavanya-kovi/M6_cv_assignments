"""
Module 6 page - Part A: optical flow and tracking; Part B: structure from motion.

Algorithms: modules/m6_optical_flow.py and modules/m6_sfm.py (run them from the
command line to regenerate the stored results in assets/m6/).
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules import m6_optical_flow as of  # noqa: E402
from modules import m6_sfm as sfm  # noqa: E402
from modules import m6_markers as mk  # noqa: E402

M6 = os.path.join(ROOT, "assets", "m6")
CAMERA = os.path.join(ROOT, "assets", "m2", "camera_params.json")

st.set_page_config(page_title="Module 6 · Motion & SfM", layout="wide")
st.title("Module 6 · Optical flow, tracking and structure from motion")
st.caption("Part A: dense optical flow, Lucas-Kanade tracking validated against measured pixel locations. "
           "Part B: 4-view structure from motion of a planar object using the Module 2 camera calibration.")


def show_file(path, kind="image", caption=None):
    if os.path.exists(path):
        if kind == "video":
            st.video(path)
        elif kind == "text":
            st.code(open(path).read(), language=None)
        else:
            st.image(path, caption=caption, width="stretch")
        return True
    return False


def show_flow_results(d, label):
    if not os.path.isdir(d):
        return False
    st.markdown(f"#### {label}")
    show_file(os.path.join(d, "flow_video.mp4"), "video")
    st.caption("Left: original · Middle: flow colour (hue = direction, brightness = speed, see wheel) · "
               "Right: flow arrows")
    show_file(os.path.join(d, "flow_evidence.png"))
    if os.path.exists(os.path.join(d, "flow_stats.csv")):
        s = pd.read_csv(os.path.join(d, "flow_stats.csv"))
        c = st.columns(4)
        c[0].metric("Mean speed of moving pixels", f"{s.moving_speed_px.mean():.2f} px/frame")
        c[1].metric("Mean moving area", f"{100 * s.moving_fraction.mean():.1f} %")
        c[2].metric("Median camera pan (u, v)", f"({s.global_u.median():.2f}, {s.global_v.median():.2f})")
        c[3].metric("Mean divergence", f"{s.divergence.mean():+.4f}")
    if os.path.exists(os.path.join(d, "track_table.csv")):
        st.markdown("**Two-frame tracking validation**")
        show_file(os.path.join(d, "track_frames.png"))
        t = pd.read_csv(os.path.join(d, "track_table.csv"))
        st.dataframe(t[["id", "x", "y", "dx", "dy", "pred_x", "pred_y", "actual_x", "actual_y",
                        "error_px", "ncc"]].round(3), hide_index=True, width="stretch")
        st.metric("Mean |LK prediction − measured location|", f"{t.error_px.mean():.3f} px")
        with st.expander("Lucas-Kanade system for one point (worked numbers)"):
            show_file(os.path.join(d, "track_system.txt"), "text")
    return True


tabA, tabB = st.tabs(["Part A · Optical flow & tracking", "Part B · Structure from motion"])

# ======================================================================= Part A
with tabA:
    shown = False
    for name in sorted(os.listdir(M6)) if os.path.isdir(M6) else []:
        if name.startswith("video"):
            lab = os.path.join(M6, name, "label.txt")
            title = open(lab).read().strip() if os.path.exists(lab) else f"My {name.replace('video', 'video ')}"
            shown |= show_flow_results(os.path.join(M6, name), title)
    if not shown:
        st.info("Stored results for my two videos will appear here.")

    st.divider()
    st.markdown("#### Run on your own video")
    up = st.file_uploader("Video (mp4 / mov / avi)", type=["mp4", "mov", "avi", "m4v"], key="vid")
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Start (s)", 0.0, 3600.0, 0.0, 1.0)
    dur = c2.slider("Duration (s)", 2, 30, 10, help="30 s clips take a while on the free server")
    side = c3.select_slider("Processing size (longest side, px)", [240, 320, 400, 480], 320)
    if up is not None and st.button("Compute optical flow", type="primary"):
        tmp = tempfile.mkdtemp()
        vpath = os.path.join(tmp, up.name)
        open(vpath, "wb").write(up.read())
        bar = st.progress(0.0, "Computing flow…")
        rows, max_mag, fps, sample = of.process_clip(
            vpath, tmp, start, dur, side,
            progress=lambda i, n: bar.progress(min(1.0, i / max(n, 1)), f"Flow {i}/{n}"))
        of.plot_flow_evidence(rows, sample, max_mag, os.path.join(tmp, "flow_evidence.png"), up.name)
        frames = [f for f, _ in of.iter_frames(vpath, start + dur / 2, 0.2, side)]
        if len(frames) >= 2:
            of.track_two_frames(frames[0], frames[1], out_dir=tmp)
        bar.empty()
        show_flow_results(tmp, f"Results for {up.name}")

    with st.expander("What the optical flow tells us"):
        st.markdown(r"""
* **Where** things move: pixels with $|\mathbf{v}| > 0.5$ px/frame form the moving-region mask (bright areas).
* **Which way**: hue = direction $\theta = \operatorname{atan2}(v, u)$; the polar histogram gives the dominant direction.
* **How fast**: brightness = speed $|\mathbf{v}|$ in px/frame; multiply by fps for px/s.
* **Camera motion**: the median flow of the whole frame is the camera pan (the background dominates).
* **Approach / recede**: divergence $\partial u/\partial x + \partial v/\partial y > 0$ means expansion (object getting closer), $<0$ moving away.
* **Rotation**: curl $\partial v/\partial x - \partial u/\partial y \neq 0$.
* **Object boundaries**: sharp changes in flow (colour edges in the middle panel) mark where independently moving objects end.
""")

# ======================================================================= Part B
LETTER = [21.59, 27.94, 21.59, 27.94]


def show_sfm(d, r):
    show_file(os.path.join(d, "detected.png"),
              caption="Automatically detected dots (1-8) and paper corners (A-D), same labels in every view")
    show_file(os.path.join(d, "reprojection.png"),
              caption="Detected points (magenta) and reconstructed 3-D points re-projected (green); "
                      "orange = recovered paper boundary")
    c1, c2 = st.columns(2)
    with c1:
        show_file(os.path.join(d, "points3d.png"))
    with c2:
        show_file(os.path.join(d, "boundary.png"))
    if r.get("edges_cm"):
        c = st.columns(4)
        for i, (n, e) in enumerate(zip(["A-B (scale)", "B-C", "C-D", "D-A"], r["edges_cm"])):
            c[i].metric(n, f"{e:.2f} cm", None if i == 0 else f"{e - LETTER[i]:+.2f} cm vs Letter",
                        delta_color="off")
    st.metric("Reprojection RMS (before -> after bundle adjustment)",
              f"{r['rms_before_ba']:.2f} -> {r['rms_after_ba']:.2f} px")


with tabB:
    d = os.path.join(M6, "sfm")
    if os.path.exists(os.path.join(d, "sfm_result.json")):
        st.markdown("#### My planar object: a US Letter sheet with 8 marker dots, 4 viewpoints")
        st.caption("Plain paper has no texture, so SIFT features land on the background. The code instead "
                   "detects the dark dots and the paper corners in every photo and matches them with a "
                   "homography search (modules/m6_markers.py).")
        show_sfm(d, json.load(open(os.path.join(d, "sfm_result.json"))))
        with st.expander("All matrices and numbers (K, E, H, R, t, camera centres, DLT triangulation)"):
            show_file(os.path.join(d, "sfm_report.txt"), "text")
    else:
        st.info("Stored results for my object will appear here.")

    st.divider()
    st.markdown("#### Run on your own 4 photos")
    st.caption("Put dark dots (at least 4, not symmetric) on a flat sheet and photograph it from 4 positions. "
               "The first photo is the reference: edge A-B is its top edge. K comes from my Module 2 "
               "calibration (lens distortion not applied).")
    ups = st.file_uploader("Four photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                           key="sfm_imgs")
    edge = st.number_input("Real length of edge A-B, the top edge in photo 1 (cm, 0 = unknown)",
                           0.0, 500.0, 21.59, 0.01)
    if not os.path.exists(CAMERA):
        st.error("Camera calibration not found: assets/m2/camera_params.json (from Module 2).")
    elif ups and len(ups) == 4 and st.button("Reconstruct", type="primary"):
        with st.spinner("Detecting dots and corners, estimating poses, triangulating, bundle adjustment..."):
            imgs = [cv2.imdecode(np.frombuffer(u.read(), np.uint8), cv2.IMREAD_COLOR) for u in ups]
            s = min(1.0, 1200 / max(imgs[0].shape[:2]))
            imgs = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in imgs]
            K, _ = sfm.load_camera(CAMERA, imgs[0].shape)
            tmp = tempfile.mkdtemp()
            try:
                r = mk.run_auto(imgs, K, [u.name for u in ups], edge or None, tmp)
                st.write("Detection:", r["detection"])
                show_sfm(tmp, json.load(open(os.path.join(tmp, "sfm_result.json"))))
                st.code(sfm.report_text(r), language=None)
            except Exception as e:  # noqa: BLE001
                st.error(f"Reconstruction failed: {e}")
    elif ups and len(ups) != 4:
        st.warning("Please upload exactly four photos.")
