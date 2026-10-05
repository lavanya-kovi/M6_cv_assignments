import time

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage

# Kernel builders
def box_kernel(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()

# (A) Spatial-domain circular convolution
def circular_convolution_spatial(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(image, kernel, mode="wrap")

# (B) Frequency-domain circular convolution
def circular_convolution_frequency(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    padded_kernel = np.zeros_like(image, dtype=np.float64)
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)

    F = np.fft.fft2(image)
    H = np.fft.fft2(padded_kernel)
    result = np.fft.ifft2(F * H)
    return np.real(result)

# Sample / uploaded image loading
def synthetic_pattern(n: int = 256) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    img = (((xx // 16) + (yy // 16)) % 2) * 255.0
    img[100:156, 100:156] = 255.0
    return img


@st.cache_data
def load_sample(name: str) -> np.ndarray:
    if name == "Synthetic pattern":
        return synthetic_pattern()
    from skimage import data, color

    if name == "Cameraman":
        return np.asarray(data.camera(), dtype=np.float64)
    if name == "Baboon (high texture)":
        try:
            return np.asarray(data.mandrill(), dtype=np.float64)
        except AttributeError:
            return color.rgb2gray(data.astronaut()).astype(np.float64) * 255.0
    raise ValueError(name)


def load_uploaded(file) -> np.ndarray:
    img = Image.open(file).convert("L")
    return np.asarray(img, dtype=np.float64)


def to_display(arr: np.ndarray, lo=None, hi=None) -> np.ndarray:
    lo = arr.min() if lo is None else lo
    hi = arr.max() if hi is None else hi
    rng = (hi - lo) or 1.0
    out = np.clip((arr - lo) / rng * 255.0, 0, 255).astype(np.uint8)
    return out

# Streamlit UI
st.set_page_config(page_title="Spatial vs Frequency Blurring", layout="wide")

st.title("Spatial Filtering vs. Frequency-Domain Filtering")
st.caption("CSc 8830 Computer Vision - Module 3 Assignment · "
           "[Source code](https://github.com/lavanya-kovi/cv-week-3-image-blur-fourier)")
st.caption(
    "Blurs an image two independent ways and checks that they agree: "
    "direct convolution in the spatial domain, vs. multiplying Fourier "
    "transforms and inverting. Both use a wrap-around (circular) boundary "
    "so the two methods are mathematically identical, not just similar — "
    "this is an experimental validation of the Convolution Theorem: "
    "f∗h ⟺ F(u,v)·H(u,v)."
)

with st.sidebar:
    st.header("Settings")

    source = st.radio("Image source", ["Sample image", "Upload my own"])
    if source == "Sample image":
        sample_name = st.selectbox(
            "Sample", ["Synthetic pattern", "Cameraman", "Baboon (high texture)"]
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
        sample_name = None

    kernel_type = st.selectbox("Kernel type", ["gaussian", "box"])
    ksize = st.slider("Kernel size (odd)", min_value=3, max_value=31, value=15, step=2)
    sigma = st.slider("Gaussian sigma", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                       disabled=(kernel_type != "gaussian"))

    run_clicked = st.button("Run comparison", type="primary", width="stretch")

# Resolve the image to use
if source == "Sample image":
    image = load_sample(sample_name)
elif uploaded_file is not None:
    image = load_uploaded(uploaded_file)
else:
    st.info("Upload an image in the sidebar, or switch to a sample image, then click Run comparison.")
    image = None

if image is not None:
    st.subheader("Original")
    st.image(to_display(image, 0, 255), width=280, clamp=True)

if run_clicked and image is not None:
    kernel = gaussian_kernel(ksize, sigma) if kernel_type == "gaussian" else box_kernel(ksize)

    t0 = time.perf_counter()
    spatial = circular_convolution_spatial(image, kernel)
    t1 = time.perf_counter()
    frequency = circular_convolution_frequency(image, kernel)
    t2 = time.perf_counter()

    diff = np.abs(spatial - frequency)
    mse = float(np.mean(diff ** 2))
    max_err = float(np.max(diff))

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(to_display(image, 0, 255), caption="Original", width="stretch")
    with c2:
        st.image(to_display(spatial, 0, 255), caption="Spatial-domain blur", width="stretch")
    with c3:
        st.image(to_display(frequency, 0, 255), caption="Frequency-domain blur", width="stretch")
    with c4:
        st.image(to_display(diff), caption="|Difference| (contrast-stretched)", width="stretch")

    st.subheader("Numerical Validation")
    m1, m2, m3 = st.columns(3)
    m1.metric("MSE (spatial vs frequency)", f"{mse:.3e}")
    m2.metric("Max absolute pixel error", f"{max_err:.3e}")
    m3.metric("Compute time", f"{(t1-t0)*1000:.1f} ms / {(t2-t1)*1000:.1f} ms")

    if mse < 1e-4:
        st.success(
            "MSE ≈ 0 (floating-point noise only): spatial convolution and "
            "frequency-domain multiplication produced the same image, "
            "confirming the Convolution Theorem."
        )
    else:
        st.warning("Results differ more than expected — check kernel/boundary settings.")
