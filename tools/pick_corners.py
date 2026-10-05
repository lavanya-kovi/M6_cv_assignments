"""
Click the 4 corners of the flat object in each photo (for Module 6 Part B).

HOW TO RUN (on your own computer, not the web server)
    python tools/pick_corners.py view1.jpg view2.jpg view3.jpg view4.jpg --out corners.json

For each photo a window opens. Click the corners in the SAME order in every
photo: top-left, top-right, bottom-right, bottom-left (as the object lies, so
corner 1 -> corner 2 is the edge whose real length you pass as --width-cm).
Keys: u = undo last click, Enter/Space = next photo, Esc = quit.
Coordinates are saved in full-resolution pixels.
"""
import argparse
import json
import os

import cv2

LABELS = ["1 top-left", "2 top-right", "3 bottom-right", "4 bottom-left"]


def pick(path, max_side=1000):
    img = cv2.imread(path)
    s = min(1.0, max_side / max(img.shape[:2]))
    disp = cv2.resize(img, None, fx=s, fy=s)
    pts = []

    def redraw():
        o = disp.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(o, (int(x * s), int(y * s)), 6, (0, 0, 255), -1)
            cv2.putText(o, LABELS[i], (int(x * s) + 8, int(y * s) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        nxt = LABELS[len(pts)] if len(pts) < 4 else "done - press Enter"
        cv2.putText(o, f"click: {nxt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, o)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / s, y / s))
            redraw()

    win = os.path.basename(path)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
            redraw()
        elif k in (13, 32) and len(pts) == 4:
            break
        elif k == 27:
            raise SystemExit("cancelled")
    cv2.destroyWindow(win)
    return [[round(x, 1), round(y, 1)] for x, y in pts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="corners.json")
    a = ap.parse_args()
    res = {os.path.basename(p): pick(p) for p in a.images}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")
