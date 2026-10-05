"""
CSc 8830 Computer Vision - Module 6 (Streamlit app)

HOW TO RUN LOCALLY
    pip install -r requirements.txt
    streamlit run Home.py
"""

import streamlit as st

st.set_page_config(page_title="CSc 8830 · Module 6", layout="wide")

st.title("CSc 8830 · Computer Vision — Module 6")
st.subheader("Optical flow, motion tracking and structure from motion — Lavanya · Georgia State University")
st.write("Open **Module 6 Motion and SfM** in the sidebar for the working demo.")
st.markdown("""
* **Part A:** dense optical flow on two videos (watering a plant, a ceiling fan), what the flow tells us
  (speed, moving area, camera pan, divergence, curl), and Lucas-Kanade tracking validated on two
  consecutive frames against the measured pixel locations.
* **Part B:** structure from motion of a planar object (a US Letter sheet with 8 marker dots) from 4
  viewpoints, using the camera matrix K from my Module 2 calibration. Dots and paper corners are
  detected and matched automatically; the paper boundary is recovered in centimetres.
""")
st.caption("Earlier modules (2, 3, 4) are in the "
           "[CV-assignment-module-4](https://github.com/lavanya-kovi/CV-assignment-module-4) repository.")
