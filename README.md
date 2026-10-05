# CSc 8830 · Computer Vision — Module 6

Optical flow, motion tracking and 4-view structure from motion.

- **Live demo:** https://cv-assignments-7a1l.onrender.com/Module_6_Motion_and_SfM
- **Source:** https://github.com/lavanya-kovi/M6_cv_assignments
- Earlier modules (2, 3, 4): https://github.com/lavanya-kovi/CV-assignment-module-4

## Run locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Repository layout

```
Home.py                               landing page
pages/6_Module_6_Motion_and_SfM.py    Module 6 web demo
modules/m6_optical_flow.py            Part A: dense flow, Lucas-Kanade tracking, bilinear interpolation
modules/m6_sfm.py                     Part B: pose (E / H), DLT triangulation, PnP, bundle adjustment
modules/m6_markers.py                 Part B: automatic dot + paper-corner detection and matching
assets/m2/camera_params.json          camera matrix K from the Module 2 calibration
assets/m6/video1, assets/m6/video2    stored optical-flow and tracking results for my two videos
assets/m6/sfm/                        stored structure-from-motion results for my 4 photos
```

## Method and results

**Part A: optical flow and tracking.** Dense Farneback optical flow on two videos (watering a plant, 26 s; a ceiling fan switching on, 30 s), shown as a side-by-side video (original | colour-coded flow | arrows). From the flow field the app computes speed, moving area, camera pan, divergence (approach) and curl (rotation). Lucas-Kanade tracking is implemented from the derivation (2x2 system `G d = b`, iterated, with our own bilinear interpolation). It is validated on two consecutive frames per video against the measured pixel locations (sub-pixel template matching): mean error 0.108 px (plant) and 0.089 px (fan).

**Part B: structure from motion.** A US Letter sheet marked with 8 dark dots, photographed from 4 viewpoints with the Module 2 calibrated phone camera (K from Module 2). Plain paper has no texture, so the code detects the dots (solid, round, dark blobs) and the paper corners (GrabCut, then a line fit per edge). 


## Deploying (Render)

`render.yaml` holds the whole deployment configuration:

- a free Python web service on Python 3.11.9;
- `pip install -r requirements.txt` as the build step;
- `streamlit run Home.py` bound to Render's `$PORT` as the start command.
