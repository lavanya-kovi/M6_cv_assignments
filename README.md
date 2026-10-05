# CSc 8830 · Computer Vision — assignment portal

One public web app for every assignment. Each module is a page in the sidebar.

- **Live demo:** https://cv-assignments.onrender.com (replace with your Render URL)
- **Source:** https://github.com/lavanya-kovi/CV-assignment-module-4

## Run locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Modules on the site

| Page | Assignment | Original repo |
|---|---|---|
| Module 2 | Camera calibration and real-world object measurement | [cv-module2-calibration](https://github.com/lavanya-kovi/cv-module2-calibration) |
| Module 3 | Image blurring: spatial vs frequency-domain filtering | [cv-week-3-image-blur-fourier](https://github.com/lavanya-kovi/cv-week-3-image-blur-fourier) |
| Module 4 | Human boundary segmentation (RGB + thermal) vs SAM2 | this repo |

The Module 2 and 3 pages are those repos' Streamlit apps with their logic unchanged. The Module 2 page also offers bundled sample data:

- the 14 checkerboard photos and 20 object photos, downscaled by 1/3 (`samples/m2/`);
- the saved calibration, plus a copy scaled to the downscaled photos (`assets/m2/`);
- the 20-object `results.csv`, for Step 3.

## Repository layout

```
Home.py                               landing page
pages/2_Module_2_Camera_Calibration.py Module 2 web demo
pages/3_Module_3_Fourier_Blur.py      Module 3 web demo
pages/4_Module_4_Human_Segmentation.py Module 4 web demo
modules/m4_segmentation.py            Module 4 algorithms (also runnable from the command line)
tools/sam2_generate_masks.py          offline SAM2 reference-mask generator (Colab)
tools/sam2_boxes.json                 box prompts used for SAM2
samples/rgb, samples/thermal          test images
assets/sam2_masks/                    SAM2 masks, one PNG per sample (same file stem)
```

## Module 4 — Human boundary segmentation

**Q1: RGB camera.** GrabCut graph-cut segmentation initialised from a box around the person, a second pass with a centre-axis foreground hint, then morphological opening/closing, small-region removal, hole filling, and pixel-exact contours (`CHAIN_APPROX_NONE`).

**Q2: Thermal camera.** Intensity channel (HSV value for pseudo-colour palettes), CLAHE, Gaussian blur, Otsu/Triangle/manual threshold, then morphology, area filter, and hole filling. An optional distance-transform watershed separates touching people.

No machine-learning or deep-learning code is used in either pipeline.

**Comparison with SAM2.** Reference masks are generated offline with SAM2 using the same box prompt. The app reports:

- IoU, Dice, precision and recall;
- boundary F-score within ±3 px;
- a colour-coded disagreement map.

### Command-line use

```bash
python modules/m4_segmentation.py rgb samples/rgb/astronaut.png --rect 20 0 350 511 --sam2 assets/sam2_masks/astronaut.png
python modules/m4_segmentation.py thermal samples/thermal/person.jpg --sam2 assets/sam2_masks/person.png
```

### Adding SAM2 masks

1. Put the images in `samples/rgb` and `samples/thermal`, and add one box per person to `tools/sam2_boxes.json`.
2. Run `tools/sam2_generate_masks.py` in Colab. Instructions are at the top of that file.
3. Commit the resulting `assets/sam2_masks/*.png`.

## Deploying (Render)

`render.yaml` holds the whole deployment configuration:

- a free Python web service on Python 3.11.9;
- `pip install -r requirements.txt` as the build step;
- `streamlit run Home.py` bound to Render's `$PORT` as the start command.

To deploy:

1. Push this repo to GitHub.
2. On render.com, click **New → Blueprint**, connect GitHub, and pick this repo. Render reads `render.yaml`; click **Apply**.
3. The first build takes a few minutes. After that, every push to `main` redeploys automatically.

On the free plan the service sleeps after about 15 minutes without traffic. The next visit wakes it up, which takes roughly a minute.

## Credits

The sample RGB image `astronaut.png` is the NASA public-domain portrait of Eileen Collins, as distributed with scikit-image.
