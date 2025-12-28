#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, shutil
from pathlib import Path
import numpy as np

def main(traj_csv, overlays_dir, out_dir, N=200):
    traj_csv = Path(traj_csv); overlays_dir=Path(overlays_dir); out_dir=Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    with open(traj_csv) as f:
        for r in csv.DictReader(f):
            if r["det_x"]=="" or r["det_y"]=="" or r["accepted"]!="1": continue
            dx=float(r["det_x"]) - float(r["pred_x"])
            dy=float(r["det_y"]) - float(r["pred_y"])
            rows.append((int(r["frame_idx"]), int(float(r["t_us"])), (dx*dx+dy*dy)**0.5))
    rows.sort(key=lambda x: x[2], reverse=True)
    def find_overlay(tus):
        c=list(overlays_dir.glob(f"kf_*_{tus}.png"))
        return c[0] if c else None
    picked=0
    for i,tus,d in rows:
        p=find_overlay(tus)
        if p:
            shutil.copy(p, out_dir/f"{i:06d}_{tus}_{d:.1f}.png")
            picked+=1
            if picked>=N: break
    print("[INFO] copy：", picked, "=>", out_dir)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--traj_csv", required=True)
    ap.add_argument("--overlays_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topN", type=int, default=200)
    args=ap.parse_args()
    main(args.traj_csv, args.overlays_dir, args.out_dir, args.topN)
