"""
CSc 8830 Computer Vision - assignment portal (Streamlit multipage app)

HOW TO RUN LOCALLY
    pip install -r requirements.txt
    streamlit run Home.py

Every file in pages/ becomes one entry in the sidebar, so each module's
assignment is reachable from this single website. Module 6 is pulled in
from its own repository at build time (see render.yaml).
"""

import streamlit as st

st.set_page_config(page_title="CSc 8830 · Computer Vision", layout="wide")

st.title("CSc 8830 · Computer Vision")
st.subheader("Assignment portal — Lavanya · Georgia State University")
st.write("Use the sidebar to open each module's working demo.")

MODULES = [
    ("Module 2", "Camera calibration and real-world object measurement",
     "https://github.com/lavanya-kovi/cv-module2-calibration"),
    ("Module 3", "Image blurring: spatial vs frequency-domain filtering (Convolution Theorem)",
     "https://github.com/lavanya-kovi/cv-week-3-image-blur-fourier"),
    ("Module 4", "Human boundary segmentation in RGB and thermal images, compared with SAM2",
     "https://github.com/lavanya-kovi/CV-assignment-module-4"),
    ("Module 6", "Optical flow, motion tracking and 4-view structure from motion",
     "https://github.com/lavanya-kovi/M6_cv_assignments"),
]
for title, topic, repo in MODULES:
    st.write(f"**{title}** — {topic} · [source code]({repo})")
