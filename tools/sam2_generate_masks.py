"""
Generate SAM2 reference masks for the Module 4 comparison.
=========================================================

SAM2 is a deep-learning model, so it is NOT part of the classical solution.
It runs offline (Google Colab with a free GPU works well) and the resulting
binary masks are committed to assets/sam2_masks/<image-name>.png, where the
web app picks them up automatically.

HOW TO RUN (Colab: Runtime -> Change runtime type -> T4 GPU)
    !git clone https://github.com/lavanya-kovi/CV-assignment-module-4.git
    %cd CV-assignment-module-4
    !pip install -q "git+https://github.com/facebookresearch/sam2.git" huggingface_hub
    !python tools/sam2_generate_masks.py --images samples/rgb samples/thermal \
            --boxes tools/sam2_boxes.json --out assets/sam2_masks
    # then download assets/sam2_masks/ and commit it (or push from Colab)

PROMPTS
    SAM2 needs a prompt. Give one box per person in tools/sam2_boxes.json:
        { "astronaut.png": [[20, 0, 370, 511]],
          "thermal_01.jpg": [[100, 40, 220, 400], [260, 50, 380, 410]] }
    Boxes are [x0, y0, x1, y1] in pixels. Use the SAME box you give the
    classical GrabCut method on the RGB page so the comparison is fair.
    Images without an entry get a single box with a 5 % margin.
    The masks of all boxes in an image are merged (union) into one PNG.
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="image files or folders")
    ap.add_argument("--boxes", default=None, help="JSON file: {image_name: [[x0,y0,x1,y1], ...]}")
    ap.add_argument("--out", default="assets/sam2_masks")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large",
                    help="Hugging Face id, e.g. facebook/sam2.1-hiera-small for less memory")
    a = ap.parse_args()

    boxes = json.load(open(a.boxes)) if a.boxes and os.path.exists(a.boxes) else {}
    files = []
    for p in a.images:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.lower().endswith(EXTS) and "_mask" not in f]
        else:
            files.append(p)
    os.makedirs(a.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained(a.model, device=device)
    print(f"Loaded {a.model} on {device}; {len(files)} image(s)")

    for path in files:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("skip (unreadable):", path)
            continue
        h, w = bgr.shape[:2]
        name = os.path.basename(path)
        bxs = boxes.get(name) or [[int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h)]]

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            union = np.zeros((h, w), bool)
            for b in bxs:
                masks, scores, _ = predictor.predict(box=np.array(b, dtype=np.float32),
                                                     multimask_output=False)
                union |= masks[0] > 0

        out = os.path.join(a.out, os.path.splitext(name)[0] + ".png")
        cv2.imwrite(out, union.astype(np.uint8) * 255)
        print(f"{name}: {len(bxs)} box(es) -> {out}  (fg {union.mean():.1%})")


if __name__ == "__main__":
    main()
