#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json
from pathlib import Path
import numpy as np

# Headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def contiguous_segments(accept_flags):
    segs=[]; s=None
    for i,a in enumerate(accept_flags):
        if a and s is None: s=i
        if (not a) and s is not None:
            segs.append((s,i-1)); s=None
    if s is not None: segs.append((s,len(accept_flags)-1))
    return segs

def main(traj_csv, metrics_json, out_dir):
    traj_csv = Path(traj_csv); out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Read trajectory and compute innovation (only for accepted frames)
    t=[]; pred=[]; det=[]; acc=[]
    with open(traj_csv) as f:
        for r in csv.DictReader(f):
            t.append(int(float(r["t_us"])))
            acc.append(int(r["accepted"])==1)
            if r["det_x"]!="" and r["det_y"]!="":
                det.append([float(r["det_x"]), float(r["det_y"])])
            else:
                det.append([np.nan, np.nan])
            pred.append([float(r["pred_x"]), float(r["pred_y"])])

    t=np.array(t); acc=np.array(acc, bool)
    det=np.array(det, dtype=np.float64); pred=np.array(pred, dtype=np.float64)
    innov = np.linalg.norm(det - pred, axis=1)
    innov = innov[np.isfinite(innov) & acc]

    # 1) Innovation histogram
    if innov.size > 0:
        plt.figure()
        plt.hist(innov, bins=50)
        p50, p90, p95 = np.percentile(innov,[50,90,95])
        for v,label in [(p50,"P50"),(p90,"P90"),(p95,"P95")]:
            plt.axvline(v, linestyle="--")
            ymax = plt.gca().get_ylim()[1]
            plt.text(v, ymax*0.9, label, rotation=90, va="top")
        plt.xlabel("innovation |z - Hx| (px)"); plt.ylabel("frames")
        plt.title("Innovation distribution (accepted)")
        plt.tight_layout(); plt.savefig(out_dir/"plot_innovation_hist.png", dpi=150)
        plt.close()

    # 2) Continuity (track length histogram)
    segs = contiguous_segments(acc.tolist())
    lengths = [e-s+1 for s,e in segs]
    plt.figure()
    if lengths:
        plt.hist(lengths, bins=30)
    plt.xlabel("track length (frames)"); plt.ylabel("count")
    plt.title("Track length distribution")
    plt.tight_layout(); plt.savefig(out_dir/"plot_tracklen_hist.png", dpi=150)
    plt.close()

    # 3) By-speed plots (from metrics JSON)
    mj = Path(metrics_json)
    if mj.exists():
        with open(mj) as f: m=json.load(f)
        buckets = m.get("by_speed",{}).get("buckets",[])
        if buckets:
            names=[f"{b['range'][0]}–{b['range'][1]}" for b in buckets]
            adopt=[b.get("adoption_rate", np.nan) for b in buckets]
            improv=[b.get("improvement", np.nan) for b in buckets]
            x=np.arange(len(buckets))
            plt.figure()
            plt.bar(x-0.2, adopt, width=0.4, label="adoption")
            plt.bar(x+0.2, improv, width=0.4, label="improvement")
            plt.xticks(x, names, rotation=20)
            plt.ylim(0,1.05); plt.legend()
            plt.title("By-speed buckets: adoption & improvement")
            plt.tight_layout(); plt.savefig(out_dir/"plot_by_speed.png", dpi=150)
            plt.close()

    print("[INFO] Plots written to:", out_dir)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--traj_csv", required=True)
    ap.add_argument("--metrics_json", required=False, default="")
    ap.add_argument("--out_dir", required=True)
    args=ap.parse_args()
    main(args.traj_csv, args.metrics_json, args.out_dir)
