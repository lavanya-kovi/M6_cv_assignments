"""
Module 4 page - Human boundary segmentation (RGB + thermal) vs SAM2.

Run the whole site locally with:   streamlit run Home.py
Algorithms live in modules/m4_segmentation.py (classical OpenCV only).
SAM2 reference masks are produced offline by tools/sam2_generate_masks.py
and stored in assets/sam2_masks/<image-name>.png.
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.m4_segmentation import (boundary_contours, compare_masks, diff_visual,  # noqa: E402
                                     draw_overlay, load_binary_mask, segment_rgb,
                                     segment_thermal)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = {"RGB": os.path.join(ROOT, "samples", "rgb"),
           "Thermal": os.path.join(ROOT, "samples", "thermal")}
SAM2_DIR = os.path.join(ROOT, "assets", "sam2_masks")

st.set_page_config(page_title="Module 4 · Human Segmentation", layout="wide")
st.title("Module 4 · Human boundary segmentation")
st.caption("Classical OpenCV pipelines (no machine learning) for RGB and thermal images, "
           "compared against SAM2 reference masks.")


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def list_samples(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and "_mask" not in f)


# ------------------------------------------------------------------ input
mode = st.radio("Camera type", ["RGB", "Thermal"], horizontal=True,
                help="Question 1 = RGB camera, Question 2 = thermal camera")

src_col, up_col = st.columns(2)
samples = list_samples(SAMPLES[mode])
with src_col:
    choice = st.selectbox("Sample image", ["—"] + samples)
with up_col:
    upload = st.file_uploader("…or upload your own", type=["png", "jpg", "jpeg", "bmp"])

img, name = None, None
if upload is not None:
    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    name = os.path.splitext(upload.name)[0]
elif choice != "—":
    img = cv2.imread(os.path.join(SAMPLES[mode], choice), cv2.IMREAD_COLOR)
    name = os.path.splitext(choice)[0]

if img is None:
    st.info("Pick a sample or upload an image to begin.")
    st.stop()

H, W = img.shape[:2]

# ------------------------------------------------------------------ parameters
with st.sidebar:
    st.header("Parameters")
    if mode == "RGB":
        st.subheader("Box around the person (%)")
        x0, x1 = st.slider("Horizontal", 0, 100, (5, 95))
        y0, y1 = st.slider("Vertical", 0, 100, (2, 99))
        iters = st.slider("GrabCut iterations", 1, 10, 5)
        fg_hint = st.checkbox("Centre-axis foreground hint", True,
                              help="Forces a thin vertical band in the box centre to foreground "
                                   "so the head/legs are not cut off.")
        largest = st.checkbox("Keep only the largest region", False)
        open_k = st.slider("Opening kernel", 1, 15, 5, step=2)
        close_k = st.slider("Closing kernel", 1, 25, 11, step=2)
    else:
        polarity = st.radio("Palette polarity", ["white-hot", "black-hot"],
                            help="white-hot: warm = bright (most datasets)")
        method = st.selectbox("Threshold", ["otsu", "triangle", "manual"])
        manual_t = st.slider("Manual threshold", 0, 255, 128, disabled=method != "manual")
        clahe = st.checkbox("CLAHE contrast enhancement", True)
        blur_k = st.slider("Gaussian blur kernel", 1, 21, 5, step=2)
        open_k = st.slider("Opening kernel", 1, 15, 5, step=2)
        close_k = st.slider("Closing kernel", 1, 25, 9, step=2)
        min_area = st.slider("Min region area (% of image)", 0.0, 5.0, 0.5, 0.1)
        ws = st.checkbox("Watershed split of touching people", False)
        peak = st.slider("Watershed peak level", 0.3, 0.9, 0.6, 0.05, disabled=not ws)

# ------------------------------------------------------------------ segment
if mode == "RGB":
    rect = (int(x0 / 100 * W), int(y0 / 100 * H), int((x1 - x0) / 100 * W), int((y1 - y0) / 100 * H))
    with st.spinner("Running GrabCut…"):
        mask = segment_rgb(img, rect, iters, open_k, close_k, largest, fg_hint)
    preview = img.copy()
    cv2.rectangle(preview, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 200, 255), max(2, W // 300))
    steps = None
else:
    mask, steps = segment_thermal(img, polarity, method, manual_t, clahe, blur_k, open_k, close_k,
                                  min_area / 100, ws, peak, return_steps=True)
    preview = img

contours = boundary_contours(mask)

c1, c2, c3 = st.columns(3)
c1.image(rgb(preview), caption="Input" + (" + box" if mode == "RGB" else ""))
c2.image(mask, caption="Binary mask")
c3.image(rgb(draw_overlay(img, mask)), caption=f"Boundary · {len(contours)} region(s)")

if steps is not None:
    with st.expander("Intermediate steps"):
        s1, s2, s3 = st.columns(3)
        s1.image(steps["intensity"], caption="Intensity channel")
        s2.image(steps["enhanced"], caption="CLAHE + blur")
        s3.image(steps["threshold"], caption=f"Threshold (t = {steps['threshold_value']:.0f})")

d1, d2 = st.columns(2)
d1.download_button("Download mask (PNG)", cv2.imencode(".png", mask)[1].tobytes(), f"{name}_mask.png")
d2.download_button("Download overlay (PNG)", cv2.imencode(".png", draw_overlay(img, mask))[1].tobytes(),
                   f"{name}_overlay.png")

# ------------------------------------------------------------------ SAM2 comparison
st.divider()
st.subheader("Comparison with SAM2")

ref, ref_src = None, None
auto = os.path.join(SAM2_DIR, name + ".png")
if os.path.exists(auto):
    ref, ref_src = load_binary_mask(auto, mask.shape), "precomputed SAM2 mask"
up_ref = st.file_uploader("Upload a SAM2 mask for this image (white = person)", type=["png", "jpg"],
                          key="sam2")
if up_ref is not None:
    ref = load_binary_mask(cv2.imdecode(np.frombuffer(up_ref.read(), np.uint8), cv2.IMREAD_GRAYSCALE),
                           mask.shape)
    ref_src = "uploaded mask"

if ref is None:
    st.info("No SAM2 mask for this image yet. Generate one with `tools/sam2_generate_masks.py` "
            "(Colab) and upload it here, or save it as `assets/sam2_masks/" + name + ".png`.")
else:
    m = compare_masks(mask, ref)
    if m["IoU"] > 0.999:
        st.warning("The reference mask is identical to the classical result. It is probably this app's "
                   "own output (e.g. a `*_mask.png` file), not a SAM2 mask. Upload the mask made by SAM2.")
    cols = st.columns(len(m))
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")
    r1, r2, r3 = st.columns(3)
    r1.image(rgb(draw_overlay(img, ref, color=(255, 120, 0))), caption=f"SAM2 ({ref_src})")
    r2.image(rgb(draw_overlay(img, mask)), caption="Ours (classical)")
    r3.image(rgb(diff_visual(mask, ref)),
             caption="Green = agree · Red = only ours · Blue = only SAM2")

with st.expander("How it works"):
    if mode == "RGB":
        st.markdown("""
1. **GrabCut** (graph-cut energy minimisation). Pixels outside the box are fixed as background; two
   Gaussian mixture colour models (foreground/background) are fitted, and a min-cut on the pixel graph
   balances *how well a pixel fits each colour model* against *a smoothness penalty that is cheap to cut
   across strong image edges*. The fit → cut loop repeats for the chosen number of iterations.
2. **Centre-axis hint**: a second GrabCut pass with a thin vertical band forced to foreground.
3. **Morphology**: opening removes specks, closing bridges small gaps; tiny regions are dropped and holes filled.
4. **Boundary**: `cv2.findContours(..., CHAIN_APPROX_NONE)` returns every boundary pixel.
""")
    else:
        st.markdown("""
1. **Intensity**: a human body (~30–37 °C skin/clothing surface) is usually warmer than the scene, so it is
   brighter in a white-hot image. Pseudo-colour palettes are converted through the HSV value channel.
2. **CLAHE + Gaussian blur** raise local contrast and suppress sensor noise.
3. **Otsu threshold** chooses the grey level *t* that maximises the between-class variance
   σ²_B(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]² of the histogram (Triangle is better for small, very hot targets).
4. **Morphology + area filter + hole filling** clean the mask.
5. **Optional watershed** on the distance transform splits people who touch.
""")
