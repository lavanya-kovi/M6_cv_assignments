import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import io
try:
    import pymupdf as fitz  # PyMuPDF, for rendering PDF pages to images
except ImportError:
    import fitz

st.set_page_config(page_title="CV Module 2 - Camera Calibration & Measurement", layout="wide")

# --- Web-portal additions: bundled sample data (photos downscaled 1/3 from the originals) ---
import glob
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M2_SAMPLES = os.path.join(_ROOT, "samples", "m2")
M2_ASSETS = os.path.join(_ROOT, "assets", "m2")
REPO_URL = "https://github.com/lavanya-kovi/cv-module2-calibration"


def load_sample_calibration_images():
    files = sorted(glob.glob(os.path.join(M2_SAMPLES, "calibration_images", "*.jpg")))
    return [(os.path.basename(f), cv2.imread(f)) for f in files]


def sample_test_images():
    """{file name: distance in m} for the bundled object photos, from results.csv."""
    files = sorted(glob.glob(os.path.join(M2_SAMPLES, "test_images", "*.jpg")))
    dist = {}
    csv_path = os.path.join(M2_ASSETS, "results.csv")
    if os.path.exists(csv_path):
        for _, r in pd.read_csv(csv_path).iterrows():
            dist[os.path.splitext(os.path.basename(str(r["image"])))[0]] = float(r["distance_m"])
    return {os.path.basename(f): dist.get(os.path.splitext(os.path.basename(f))[0], 2.5) for f in files}

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "pdf"]
def decode_uploaded_image(uploaded_file):
    """Convert a Streamlit UploadedFile into an OpenCV BGR image array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def is_pdf(uploaded_file):
    name = uploaded_file.name.lower()
    return name.endswith(".pdf") or uploaded_file.type == "application/pdf"


def pdf_to_cv2_images(uploaded_file, dpi=200):
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    zoom = dpi / 72.0  # PDF points are 72 per inch
    mat = fitz.Matrix(zoom, zoom)

    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:  # grayscale
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        pages.append((f"page{i + 1}", img_bgr))
    doc.close()
    return pages


def load_images_from_upload(uploaded_file, dpi=200):
    if is_pdf(uploaded_file):
        pages = pdf_to_cv2_images(uploaded_file, dpi=dpi)
        return [(f"{uploaded_file.name}_{label}", img) for label, img in pages]
    else:
        return [(uploaded_file.name, decode_uploaded_image(uploaded_file))]


def run_calibration(images, checkerboard, square_x_mm, square_y_mm, progress_callback=None):
    
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp[:, 0] *= square_x_mm
    objp[:, 1] *= square_y_mm

    objpoints, imgpoints, used_names, blur_scores = [], [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_shape = None
    skipped_names = []

    detect_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                     + cv2.CALIB_CB_NORMALIZE_IMAGE)
    max_detect_dim = 1600

    for i, (name, img) in enumerate(images):
        if progress_callback:
            progress_callback(i, len(images), name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        blur_scores.append((name, cv2.Laplacian(gray, cv2.CV_64F).var()))

        h, w = gray.shape
        scale = min(max_detect_dim / w, max_detect_dim / h, 1.0)
        small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray

        found, corners = cv2.findChessboardCorners(small, checkerboard, detect_flags)
        if found:
            if scale < 1.0:
                corners = corners / scale  # map back to full-resolution coordinates
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used_names.append(name)
        else:
            skipped_names.append(name)

    if len(objpoints) < 1:
        return {"error": f"No images had detectable checkerboard corners. "
                          f"Skipped: {skipped_names}"}

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    per_image_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        pts1 = imgpoints[i].reshape(-1, 2).astype(np.float32)
        pts2 = imgpoints2.reshape(-1, 2).astype(np.float32)
        rmse = np.sqrt(np.mean(np.sum((pts1 - pts2) ** 2, axis=1)))
        per_image_errors.append((used_names[i], rmse))

    return {
        "error": None,
        "K": K, "dist": dist, "ret": ret,
        "image_size": gray_shape,
        "used_names": used_names, "skipped_names": skipped_names,
        "blur_scores": sorted(blur_scores, key=lambda x: x[1]),
        "per_image_errors": sorted(per_image_errors, key=lambda x: x[1], reverse=True),
    }


def fix_orientation(img, calib_size):
    img_h, img_w = img.shape[:2]
    calib_w, calib_h = calib_size
    if calib_w is None:
        return img, False
    if (calib_w > calib_h) != (img_w > img_h):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
    return img, False


def scale_K_for_image(K, calib_size, img_w, img_h):
    calib_w, calib_h = calib_size
    if calib_w is None:
        return K.copy()
    if img_w == calib_w and img_h == calib_h:
        return K.copy()

    K_new = K.copy()
    if img_w == calib_w and img_h != calib_h:
        K_new[1, 2] -= (calib_h - img_h) / 2.0
    elif img_h == calib_h and img_w != calib_w:
        K_new[0, 2] -= (calib_w - img_w) / 2.0
    else:
        K_new[0, 0] *= img_w / calib_w
        K_new[0, 2] *= img_w / calib_w
        K_new[1, 1] *= img_h / calib_h
        K_new[1, 2] *= img_h / calib_h
    return K_new


def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, new_K), new_K


def compute_real_world_size(w_px, h_px, Z_m, fx, fy):
    Z_mm = Z_m * 1000.0
    return (w_px * Z_mm) / fx / 10.0, (h_px * Z_mm) / fy / 10.0  # returns cm


def compute_errors(df):
    rows, abs_errors, pct_errors, est_vals, gt_vals = [], [], [], [], []
    for _, r in df.iterrows():
        try:
            est_w, gt_w = float(r["est_width_cm"]), float(r["gt_width_cm"])
            est_h, gt_h = float(r["est_height_cm"]), float(r["gt_height_cm"])
        except (ValueError, TypeError):
            continue
        err_w, err_h = est_w - gt_w, est_h - gt_h
        pct_w = abs(err_w) / gt_w * 100 if gt_w else float("nan")
        pct_h = abs(err_h) / gt_h * 100 if gt_h else float("nan")
        rows.append({"label": r["label"], "est_width_cm": est_w, "gt_width_cm": gt_w,
                      "width_error_cm": round(err_w, 2), "width_pct_error": round(pct_w, 2),
                      "est_height_cm": est_h, "gt_height_cm": gt_h,
                      "height_error_cm": round(err_h, 2), "height_pct_error": round(pct_h, 2)})
        abs_errors += [abs(err_w), abs(err_h)]
        pct_errors += [pct_w, pct_h]
        est_vals += [est_w, est_h]
        gt_vals += [gt_w, gt_h]
    return pd.DataFrame(rows), np.array(abs_errors), np.array(pct_errors), np.array(est_vals), np.array(gt_vals)

if "camera_params" not in st.session_state:
    st.session_state.camera_params = None  # dict: K, dist, image_size
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "label", "distance_m", "w_px", "h_px",
        "est_width_cm", "est_height_cm", "gt_width_cm", "gt_height_cm"])


st.title("Camera Calibration & Real-World Measurement")
st.caption("CSc 8830 Computer Vision - Module 2 Assignment · "
           f"[Source code]({REPO_URL})")

tab1, tab2, tab3 = st.tabs(["Step 1: Calibration", "Step 2: Measurement", "Step 3: Validation"])

with tab1:
    st.header("Step 1: Camera Calibration")
    st.write("Upload 15+ checkerboard photos taken with your phone camera "
             "(varied distance, angle, rotation, and position in frame), "
             "or run it on my own 14 checkerboard photos.")

    col1, col2 = st.columns(2)
    with col1:
        cb_cols = st.number_input("Checkerboard internal corners (columns)", min_value=2, value=9)
        cb_rows = st.number_input("Checkerboard internal corners (rows)", min_value=2, value=6)
    with col2:
        sq_x = st.number_input("Square width (mm)", min_value=1.0, value=23.0, step=0.1)
        sq_y = st.number_input("Square height (mm)", min_value=1.0, value=23.5, step=0.1)

    use_sample_calib = st.checkbox("Use my 14 sample checkerboard photos (downscaled to 1428×1904)", value=True)
    uploaded_calib_imgs = None
    if not use_sample_calib:
        uploaded_calib_imgs = st.file_uploader(
            "Upload checkerboard images (JPG/PNG/PDF — a multi-page PDF contributes one image per page)",
            type=ACCEPTED_TYPES,
            accept_multiple_files=True, key="calib_uploader")

    if st.button("Run Calibration", type="primary"):
        if not use_sample_calib and (not uploaded_calib_imgs or len(uploaded_calib_imgs) < 1):
            st.error("Please upload at least 1 checkerboard image.")
        else:
            with st.spinner("Loading images (rendering any PDFs)..."):
                images = []
                if use_sample_calib:
                    images = load_sample_calibration_images()
                else:
                    for f in uploaded_calib_imgs:
                        images.extend(load_images_from_upload(f))

            progress_bar = st.progress(0, text="Starting corner detection...")

            def update_progress(i, total, name):
                progress_bar.progress((i + 1) / total, text=f"Detecting corners: {name} ({i + 1}/{total})")

            result = run_calibration(images, (int(cb_cols), int(cb_rows)), sq_x, sq_y,
                                      progress_callback=update_progress)
            progress_bar.empty()

            if result["error"]:
                st.error(result["error"])
            else:
                st.session_state.camera_params = {
                    "K": result["K"].tolist(),
                    "dist": result["dist"].tolist(),
                    "reprojection_error": result["ret"],
                    "image_size": result["image_size"],
                }
                st.success(f"Calibration complete. Used {len(result['used_names'])} / "
                           f"{len(images)} images.")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Reprojection error (RMSE, px)", f"{result['ret']:.3f}")
                    st.write("**Intrinsic matrix K:**")
                    st.dataframe(pd.DataFrame(result["K"]))
                    st.write("**Distortion coefficients:**")
                    st.write(result["dist"].ravel())

                with c2:
                    st.write("**Blur scores (lower = blurrier):**")
                    st.dataframe(pd.DataFrame(result["blur_scores"], columns=["image", "blur_score"]))

                st.write("**Per-image reprojection error:**")
                st.dataframe(pd.DataFrame(result["per_image_errors"], columns=["image", "rmse_px"]))

                if result["skipped_names"]:
                    st.warning(f"Corners not found in: {result['skipped_names']}")

                st.download_button(
                    "Download camera_params.json",
                    data=json.dumps(st.session_state.camera_params, indent=4),
                    file_name="camera_params.json", mime="application/json")

    if st.session_state.camera_params:
        st.info("Calibration loaded in this session — you can proceed to Step 2.")


with tab2:
    st.header("Step 2: Real-World Object Measurement")

    SAVED_SAMPLES = "My saved calibration (matches the sample photos)"
    SAVED_FULL = "My saved calibration (full-resolution 4284×5712 phone photos)"
    params_source = st.radio("Camera parameters source",
                              [SAVED_SAMPLES, "Use calibration from Step 1 (this session)",
                               "Upload camera_params.json", SAVED_FULL])

    camera_params = None
    if params_source == SAVED_SAMPLES:
        with open(os.path.join(M2_ASSETS, "camera_params_samples.json")) as f:
            camera_params = json.load(f)
    elif params_source == SAVED_FULL:
        with open(os.path.join(M2_ASSETS, "camera_params.json")) as f:
            camera_params = json.load(f)
    elif params_source == "Use calibration from Step 1 (this session)":
        if st.session_state.camera_params:
            camera_params = st.session_state.camera_params
        else:
            st.warning("No calibration in this session yet — run Step 1, or upload a file instead.")
    else:
        uploaded_params = st.file_uploader("Upload camera_params.json", type=["json"])
        if uploaded_params:
            camera_params = json.load(uploaded_params)

    if camera_params:
        K = np.array(camera_params["K"])
        dist = np.array(camera_params["dist"])
        calib_size = tuple(camera_params.get("image_size", (None, None)))

        img_source = st.radio("Object photo", ["Sample photo", "Upload my own"], horizontal=True)
        img = None
        if img_source == "Sample photo":
            samples = sample_test_images()
            sample_name = st.selectbox("Sample object", list(samples))
            default_label = os.path.splitext(sample_name)[0]
            default_dist = samples[sample_name]
        else:
            default_label, default_dist = "object1", 2.5

        obj_label = st.text_input("Object label", value=default_label)
        distance_m = st.number_input("Camera-to-object distance (m)", min_value=0.1,
                                     value=float(default_dist), step=0.1)

        if img_source == "Sample photo":
            img = cv2.imread(os.path.join(M2_SAMPLES, "test_images", sample_name))
        else:
            uploaded_obj_img = st.file_uploader(
                "Upload object photo (JPG/PNG/PDF)", type=ACCEPTED_TYPES, key="obj_uploader")
            if uploaded_obj_img:
                obj_pages = load_images_from_upload(uploaded_obj_img)
                if len(obj_pages) > 1:
                    page_names = [name for name, _ in obj_pages]
                    chosen = st.selectbox("This PDF has multiple pages — pick one:", page_names)
                    img = dict(obj_pages)[chosen]
                else:
                    img = obj_pages[0][1]

        if img is not None:
            img, rotated = fix_orientation(img, calib_size)
            img_h, img_w = img.shape[:2]
            K_used = scale_K_for_image(K, calib_size, img_w, img_h)
            undistorted, new_K = undistort_image(img, K_used, dist)
            fx, fy = new_K[0, 0], new_K[1, 1]

            if rotated:
                st.info("Auto-rotated image 90 deg to match calibration orientation.")
            if (img_w, img_h) != calib_size and calib_size[0] is not None:
                st.info(f"Resolution differs from calibration ({calib_size} vs "
                        f"{(img_w, img_h)}) — applied crop/resize correction to K.")

            disp_w = undistorted.shape[1]
            disp_h = undistorted.shape[0]
            st.write("Adjust the sliders to draw a bounding box around the object:")

            c1, c2 = st.columns(2)
            with c1:
                x = st.slider("Box left (x)", 0, disp_w - 1, disp_w // 4)
                y = st.slider("Box top (y)", 0, disp_h - 1, disp_h // 4)
            with c2:
                w_box = st.slider("Box width", 1, disp_w - x, disp_w // 2)
                h_box = st.slider("Box height", 1, disp_h - y, disp_h // 2)

            preview = Image.fromarray(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(preview)
            draw.rectangle([x, y, x + w_box, y + h_box], outline="lime", width=max(2, disp_w // 300))
            st.image(preview, caption="Preview - adjust sliders until the box tightly fits the object",
                     width="stretch")

            if st.button("Compute real-world size", type="primary"):
                W_cm, H_cm = compute_real_world_size(w_box, h_box, distance_m, fx, fy)
                st.session_state["last_measurement"] = {
                    "label": obj_label, "distance_m": distance_m,
                    "w_px": w_box, "h_px": h_box,
                    "est_width_cm": round(W_cm, 3), "est_height_cm": round(H_cm, 3)}
                st.success(f"Estimated size: **{W_cm:.2f} cm wide x {H_cm:.2f} cm tall**")

        if "last_measurement" in st.session_state:
            m = st.session_state["last_measurement"]
            st.write(f"Last estimate for **{m['label']}**: {m['est_width_cm']} cm x {m['est_height_cm']} cm")
            gc1, gc2 = st.columns(2)
            with gc1:
                gt_w = st.number_input("Ground truth width (cm)", min_value=0.0, step=0.1, key="gt_w_input")
            with gc2:
                gt_h = st.number_input("Ground truth height (cm)", min_value=0.0, step=0.1, key="gt_h_input")

            if st.button("Add to results table"):
                new_row = {**m, "gt_width_cm": gt_w, "gt_height_cm": gt_h}
                st.session_state.results_df = pd.concat(
                    [st.session_state.results_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"Added {m['label']} to results table ({len(st.session_state.results_df)} total).")
                del st.session_state["last_measurement"]

    st.subheader("Logged measurements")
    st.dataframe(st.session_state.results_df, width="stretch")
    if not st.session_state.results_df.empty:
        st.download_button("Download results.csv",
                            data=st.session_state.results_df.to_csv(index=False),
                            file_name="results.csv", mime="text/csv")
        uploaded_prior_csv = st.file_uploader("...or load a previously saved results.csv", type=["csv"])
        if uploaded_prior_csv:
            st.session_state.results_df = pd.read_csv(uploaded_prior_csv)
            st.success("Loaded results.csv into this session.")


with tab3:
    st.header("Step 3: Validation & Error Analysis")

    if st.button("Load my 20 measured objects (results.csv)"):
        st.session_state.results_df = pd.read_csv(os.path.join(M2_ASSETS, "results.csv"))
        st.success("Loaded 20 objects measured on the full-resolution photos.")

    df = st.session_state.results_df
    if df.empty or df["gt_width_cm"].isna().all():
        st.info("No complete measurements with ground truth yet. Add some in Step 2, "
                "or upload a results.csv there.")
    else:
        if st.button("Run Validation", type="primary"):
            report_df, abs_errors, pct_errors, est_vals, gt_vals = compute_errors(df)

            if report_df.empty:
                st.error("No rows have complete ground-truth data.")
            else:
                st.subheader("Per-object error report")
                st.dataframe(report_df, width="stretch")
                st.download_button("Download validation_report.csv",
                                    data=report_df.to_csv(index=False),
                                    file_name="validation_report.csv", mime="text/csv")

                st.subheader("Aggregate error statistics")
                mae = abs_errors.mean()
                rmse = np.sqrt((abs_errors ** 2).mean())
                std = abs_errors.std()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE (cm)", f"{mae:.2f}")
                m2.metric("RMSE (cm)", f"{rmse:.2f}")
                m3.metric("Std Dev (cm)", f"{std:.2f}")
                m4.metric("Mean % error", f"{pct_errors.mean():.1f}%")

                st.subheader("Estimated vs Ground Truth")
                fig, ax = plt.subplots(figsize=(6, 6))
                max_val = max(est_vals.max(), gt_vals.max()) * 1.1
                ax.plot([0, max_val], [0, max_val], "k--", label="Ideal (y = x)")
                ax.scatter(gt_vals, est_vals, alpha=0.7, label="Measurements")
                ax.set_xlabel("Ground truth (cm)")
                ax.set_ylabel("Estimated (cm)")
                ax.legend()
                ax.axis("equal")
                st.pyplot(fig)