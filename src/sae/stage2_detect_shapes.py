#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: SAE-based thresholding + contour/centroid detection
The input directory (--sae_dir) supports two naming：
  - sae_pos_{idx}_{t_us}.npy + sae_neg_{idx}_{t_us}.npy   (float32, 0..1)
  - or sae_fuse_{idx}_{t_us}.npy / .png                   (float32 0..1 or 8bit pic)

output：
  - <out_dir>/detections.csv     t_us,x,y,w,h,score,pol_mode
  - <out_dir>/overlays/*.png     Frame extraction visualization (every N frames)
"""

import re
import csv
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import cv2

PAT = {
    "pos": re.compile(r"sae_pos_(\d+)_(\d+)\.(npy|npz)$", re.IGNORECASE),
    "neg": re.compile(r"sae_neg_(\d+)_(\d+)\.(npy|npz)$", re.IGNORECASE),
    "fnpy": re.compile(r"sae_fuse_(\d+)_(\d+)\.(npy|npz)$", re.IGNORECASE),
    "fpng": re.compile(r"sae_fuse_(\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE),
}

def list_frames(d: Path):
    d = Path(d)
    pos, neg, fuse = {}, {}, {}
    for p in d.glob("sae_pos_*.*"):
        m = PAT["pos"].match(p.name)
        if m: pos[(int(m.group(1)), int(m.group(2)))] = p
    for p in d.glob("sae_neg_*.*"):
        m = PAT["neg"].match(p.name)
        if m: neg[(int(m.group(1)), int(m.group(2)))] = p
    for p in d.glob("sae_fuse_*.*"):
        m1 = PAT["fnpy"].match(p.name); m2 = PAT["fpng"].match(p.name)
        if m1: fuse[(int(m1.group(1)), int(m1.group(2)))] = p
        elif m2: fuse[(int(m2.group(1)), int(m2.group(2)))] = p

    recs = []
    keys = set(pos) | set(neg) | set(fuse)
    for k in sorted(keys):
        idx, tus = k
        r = {"idx": idx, "t_us": tus, "pos": pos.get(k), "neg": neg.get(k), "fuse": fuse.get(k)}
        if (r["pos"] is not None and r["neg"] is not None) or (r["fuse"] is not None):
            recs.append(r)
    return recs

def load_sae(p: Path) -> np.ndarray:
    p = Path(p)
    if p.suffix.lower() in (".npy", ".npz"):
        a = np.load(p)
        if isinstance(a, np.lib.npyio.NpzFile):
            for k in ["arr_0","sae","data"]:
                if k in a: a = a[k]; break
        a = a.astype(np.float32, copy=False)
        return np.clip(a, 0.0, 1.0)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Read Failure：{p}")
    return (img.astype(np.float32) / 255.0)

def fuse_polarity(Spos: np.ndarray, Sneg: np.ndarray, mode: str="max", alpha: float=0.5) -> np.ndarray:
    if mode == "max":
        return np.maximum(Spos, Sneg)
    elif mode == "weighted":
        return alpha * Spos + (1.0 - alpha) * Sneg
    else:
        raise ValueError(mode)

def temporal_filter(buf: deque, K: int) -> np.ndarray:
    if K <= 1 or len(buf) == 1:
        return buf[-1]
    arr = np.stack(list(buf), axis=0)
    med = np.median(arr, axis=0)
    return med.astype(np.float32, copy=False)

def threshold01(S: np.ndarray, mode: str="otsu", q: float=0.9) -> np.ndarray:
    S8 = (np.clip(S, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mode == "otsu":
        _, bw = cv2.threshold(S8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "quantile":
        t = int(np.quantile(S8, q))
        _, bw = cv2.threshold(S8, t, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(mode)
    return bw

def morph_open(bw: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return bw
    ker = np.ones((k, k), dtype=np.uint8)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)

def largest_with_gate(contours, prev_area: float, adapt: bool, rmin: float, rmax: float):
    if not contours: return None, 0.0
    areas = [cv2.contourArea(c) for c in contours]
    if (not adapt) or prev_area <= 0.0:
        i = int(np.argmax(areas)); return contours[i], areas[i]
    lo, hi = prev_area*rmin, prev_area*rmax
    gated = [(c,a) for c,a in zip(contours,areas) if lo <= a <= hi]
    if gated:
        gated.sort(key=lambda ca: abs(ca[1]-prev_area))
        return gated[0]
    i = int(np.argmax(areas)); return contours[i], areas[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", required=True, type=Path, help="Stage1 Output Directory（sae_frames）")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output Directory（Advice Data/<seq>）")
    ap.add_argument("--fuse_mode", default="max", choices=["max","weighted"], help="if there is pos/neg，Two-channel fusion method")
    ap.add_argument("--fuse_alpha", type=float, default=0.5)
    ap.add_argument("--K_temporal", type=int, default=3, help="Time series median window (1 means off)")
    ap.add_argument("--morph_kernel", type=int, default=3, help="Binary image kernel size (1 means closed)")
    ap.add_argument("--thresh_mode", choices=["otsu","quantile"], default="otsu")
    ap.add_argument("--quantile_q", type=float, default=0.90)
    ap.add_argument("--area_adapt", action="store_true", help="Whether to enable area adaptive gating")
    ap.add_argument("--area_r_min", type=float, default=0.7)
    ap.add_argument("--area_r_max", type=float, default=1.4)
    ap.add_argument("--save_every", type=int, default=50, help="Overlay frame interval (0 to disable)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    manifest = list_frames(args.sae_dir)
    if not manifest:
        raise RuntimeError(f"SAE frame not found：{args.sae_dir}")

    det_path = args.out_dir / "detections.csv"
    with open(det_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us","x","y","w","h","score","pol_mode"])

        tbuf = deque(maxlen=max(1, args.K_temporal))
        prev_area = 0.0

        for rec in manifest:
            idx, tus = rec["idx"], rec["t_us"]

            if rec["pos"] is not None and rec["neg"] is not None:
                Sp = load_sae(rec["pos"])
                Sn = load_sae(rec["neg"])
                S  = fuse_polarity(Sp, Sn, args.fuse_mode, args.fuse_alpha)
            else:
                S  = load_sae(rec["fuse"])
                Sp = Sn = None

            tbuf.append(S)
            Sf = temporal_filter(tbuf, args.K_temporal)

            bw = threshold01(Sf, args.thresh_mode, args.quantile_q)
            bw = morph_open(bw, args.morph_kernel)

            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour, area = largest_with_gate(contours, prev_area, args.area_adapt, args.area_r_min, args.area_r_max)

            if contour is not None and len(contour) >= 5:
                x, y, wbox, hbox = cv2.boundingRect(contour)
                m = cv2.moments(contour)
                if m["m00"] > 1e-6:
                    cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]
                else:
                    cx, cy = x + wbox/2.0, y + hbox/2.0

                mask = np.zeros_like(Sf, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                score = float(Sf[mask == 255].mean()) if np.any(mask) else 0.0

                if Sp is not None and Sn is not None and score > 0.0:
                    sp = float(Sp[mask == 255].mean())
                    sn = float(Sn[mask == 255].mean())
                    pol_mode = "pos" if sp >= sn else "neg"
                else:
                    pol_mode = ""

                w.writerow([tus, f"{cx:.3f}", f"{cy:.3f}", f"{wbox:.3f}", f"{hbox:.3f}", f"{score:.6f}", pol_mode])

                if args.save_every and (idx % args.save_every == 0):
                    vis = (np.clip(Sf, 0.0, 1.0) * 255).astype(np.uint8)
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis, (x, y), (x+wbox, y+hbox), (255,255,255), 1)
                    cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, (255,255,255), -1)
                    outp = args.out_dir / "overlays" / f"sae_{idx:06d}_{tus}.png"
                    cv2.imwrite(str(outp), vis)

                prev_area = area
            else:
                w.writerow([tus, "", "", "", "", "0.0", ""])


    print(f"[INFO] Detection Results Written in：{det_path}")
    print(f"[INFO] Overlays Output：{args.out_dir/'overlays'}（per {args.save_every} frame）")

if __name__ == "__main__":
    main()
