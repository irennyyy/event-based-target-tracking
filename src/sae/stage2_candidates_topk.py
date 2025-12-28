#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, csv, argparse
from pathlib import Path
import numpy as np
import cv2

def parse_ts_us(path: Path) -> int:
    m = re.search(r'_(\d{10,})\.npy$', path.name)
    if not m:
        raise ValueError(f"Unable to parse timestamp from filename: {path}")
    return int(m.group(1))

def nms_points(cands, radius: float, topk: int):
    keep = []; r2 = radius * radius
    for c in sorted(cands, key=lambda x: x[2], reverse=True):
        cy, cx, score, w, h, area = c
        ok = True
        for sel in keep:
            dy = cy - sel[0]; dx = cx - sel[1]
            if dx*dx + dy*dy <= r2:
                ok = False; break
        if ok: keep.append(c)
        if len(keep) >= topk: break
    return keep

def extract_candidates(img: np.ndarray,
                       thr: float, close_ks: int,
                       min_area: int, nms_radius: float, topk: int):
    img = np.clip(img.astype(np.float32), 0, 1)
    mask = (img >= thr).astype(np.uint8) * 255
    if close_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:  # Filter small noise
            continue
        cx, cy = centroids[i]
        mean_val = cv2.mean(img, mask=(labels == i).astype(np.uint8))[0]
        score = float(mean_val * area)
        cands.append((cy, cx, score, w, h, area))

    if not cands:
        return []
    return nms_points(cands, nms_radius, topk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", required=True, type=Path, help="Directory containing sae_fuse_*.npy")
    ap.add_argument("--pattern", default="sae_fuse_*.npy", type=str)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--min_area", type=int, default=12)
    ap.add_argument("--nms_radius", type=float, default=8.0)
    ap.add_argument("--limit_frames", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.sae_dir).glob(args.pattern))
    if args.limit_frames > 0:
        files = files[:args.limit_frames]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_us", "k", "x", "y", "score", "w", "h", "area"])
        for p in files:
            ts = parse_ts_us(p)
            img = np.load(str(p))
            cands = extract_candidates(img,
                thr=args.thr, close_ks=args.close,
                min_area=args.min_area, nms_radius=args.nms_radius, topk=args.topk)
            for k, (cy, cx, score, w, h, area) in enumerate(cands):
                wr.writerow([ts, k, f"{cx:.2f}", f"{cy:.2f}", f"{score:.3f}", w, h, area])
    print(f"[INFO] Candidates have been written to：{args.out_csv}")

if __name__ == "__main__":
    main()
