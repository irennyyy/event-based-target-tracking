#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, argparse
from pathlib import Path
import numpy as np

def load_det(p: Path):
    with open(p) as f:
        rdr = csv.DictReader(f); rows = list(rdr)
    def col(name):
        return np.array([float(r[name]) if r.get(name,"") not in ("","nan") else np.nan for r in rows], dtype=np.float64)
    t = np.array([float(r.get("t_us","nan")) for r in rows], dtype=np.float64)
    return {"t_us": t, "x": col("x"), "y": col("y"), "w": col("w"), "h": col("h"), "score": col("score")}

def load_speed(p: Path, shift_us: float = 0.0):
    with open(p) as f:
        rdr = csv.DictReader(f); rows = [r for r in rdr if r.get("t_us","")!=""]
    t = np.array([float(r["t_us"]) for r in rows], dtype=np.float64) + shift_us
    v = np.array([float(r["v_lin"]) if r.get("v_lin","") not in ("","nan") else np.nan for r in rows], dtype=np.float64)
    return t, v

def interp_to(ts_det, ts_gt, val_gt):
    order = np.argsort(ts_gt); ts_gt=ts_gt[order]; val_gt=val_gt[order]
    out = np.full_like(ts_det, np.nan, dtype=np.float64)
    mask = (ts_det>=ts_gt[0]) & (ts_det<=ts_gt[-1])
    out[mask] = np.interp(ts_det[mask], ts_gt, val_gt)
    return out

def rms(arr):
    arr = arr[~np.isnan(arr)]
    return float(np.sqrt(np.mean(arr*arr))) if arr.size else float("nan")

def segments(mask):
    idx = np.where(mask)[0]
    if idx.size==0: return []
    brk = np.where(np.diff(idx)>1)[0]
    starts = np.r_[0, brk+1]; ends = np.r_[brk+1, idx.size]
    return [(idx[s], idx[e-1]+1) for s,e in zip(starts, ends)]

def compute_jitter(t, x, y, valid):
    vals=[]
    for s,e in segments(valid):
        if e-s<3: continue
        d2x = x[s+2:e] - 2*x[s+1:e-1] + x[s:e-2]
        d2y = y[s+2:e] - 2*y[s+1:e-1] + y[s:e-2]
        vals.append(np.hypot(d2x,d2y))
    if not vals: return float("nan")
    return rms(np.concatenate(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_csv", required=True, type=Path)
    ap.add_argument("--gt_speed_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--bins", type=str, default="0,0.05,0.2,999",
                    help="Linear speed bins (m/s)")
    ap.add_argument("--gt_shift_us", type=float, default=0.0,
                    help="Global time shift added to GT speed (microseconds, can be negative)")
    ap.add_argument("--align_first", action="store_true",
                    help="Auto-align: set GT first timestamp equal to detections' first timestamp (overrides gt_shift_us)")
    args = ap.parse_args()

    det = load_det(args.detections_csv)
    t_det = det["t_us"]

    shift = float(args.gt_shift_us)
    if args.align_first:
        # Read unshifted GT to compute the initial time difference
        t_gt_raw, _ = load_speed(args.gt_speed_csv, shift_us=0.0)
        shift = float(np.nanmin(t_det) - np.nanmin(t_gt_raw))
        print(f"[INFO] align_first: using automatic shift gt_shift_us={shift:.0f}")

    t_gt, v_gt = load_speed(args.gt_speed_csv, shift_us=shift)
    v_det = interp_to(t_det, t_gt, v_gt)

    bins = np.array([float(x) for x in args.bins.split(",")], dtype=np.float64)
    valid_det = (~np.isnan(det["x"])) & (~np.isnan(det["y"]))
    area = det["w"] * det["h"]

    out = {"bin_edges_mps": bins.tolist(), "gt_shift_us": shift, "bins": []}
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (v_det>=lo) & (v_det<hi)
        n_total = int(np.sum(mask))
        n_valid = int(np.sum(mask & valid_det))
        det_rate = n_valid / n_total if n_total else float("nan")
        jitter = compute_jitter(t_det, det["x"], det["y"], mask & valid_det)
        area_mask = area[mask & valid_det]
        area_cv = float(np.nanstd(area_mask)/np.nanmean(area_mask)) if area_mask.size and np.nanmean(area_mask)>1e-6 else float("nan")
        score_mean = float(np.nanmean(det["score"][mask & valid_det])) if n_valid else float("nan")
        out["bins"].append({
            "range_mps": [lo, hi],
            "frames_total": n_total,
            "frames_detected": n_valid,
            "detection_rate": det_rate,
            "jitter_rms_px": jitter,
            "bbox_area_cv": area_cv,
            "score_mean": score_mean,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outp = args.out_dir / "metrics_stage2_by_speed.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[INFO] Written to：{outp}")

if __name__ == "__main__":
    main()
