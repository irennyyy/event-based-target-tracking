#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, glob, os
from pathlib import Path
import numpy as np
import cv2

def parse_tus(p):
    m = re.search(r'_(\d{10,})\.png$', p)
    return int(m.group(1)) if m else None

def main(out_dir):
    ov = Path(out_dir)/"overlays_kf"
    imgs = sorted(glob.glob(str(ov/"kf_*.png")), key=parse_tus)
    assert imgs, f"no overlays in: {ov}"
    # Estimated FPS
    t_us = np.array([parse_tus(p) for p in imgs], dtype=np.int64)
    dt_med = np.median(np.diff(t_us)) if len(t_us)>1 else 20_000  # us
    fps = float(1e6/max(1,dt_med))
    # Read the first frame to determine the resolution
    fr0 = cv2.imread(imgs[0])
    h,w = fr0.shape[:2]
    outp = str(Path(out_dir)/"demo_kf.mp4")
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for p in imgs:
        im = cv2.imread(p)
        if im is None: continue
        vw.write(im)
    vw.release()
    print(f"[INFO] saved video: {outp}  (fps≈{fps:.2f}, frames={len(imgs)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.out_dir)
