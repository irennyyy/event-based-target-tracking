#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, argparse
from pathlib import Path
import numpy as np

def load_traj(p: Path):
    rows=[]
    with open(p) as f:
        rows = list(csv.DictReader(f))
    def col(name):
        out=[]
        for r in rows:
            s=r.get(name,"")
            out.append(float(s) if s!="" else np.nan)
        return np.array(out, dtype=np.float64)
    return {
        "t_us": col("t_us"),
        "pred_x": col("pred_x"), "pred_y": col("pred_y"),
        "kf_x": col("kf_x"), "kf_y": col("kf_y"),
        "det_x": col("det_x"), "det_y": col("det_y"),
        "accepted": col("accepted"), "dt_ms": col("dt_ms"),
    }

def load_topk_ts(p: Path):
    """Read detections_topk.csv and return the set of t_us where candidates appeared."""
    if p is None or not p.exists(): return None
    S=set()
    with open(p) as f:
        for r in csv.DictReader(f):
            ts = r.get("t_us","")
            if ts!="":
                S.add(int(float(ts)))
    return S

def jitter_rms(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x=x[m]; y=y[m]
    if x.size<3: return float("nan")
    d2x = x[2:] - 2*x[1:-1] + x[:-2]
    d2y = y[2:] - 2*y[1:-1] + y[:-2]
    v = np.hypot(d2x,d2y)
    return float(np.sqrt(np.mean(v*v)))

def innovation_stats(pred_x, pred_y, det_x, det_y, accepted):
    m = np.isfinite(det_x) & np.isfinite(det_y) & (accepted>0.5)
    if not np.any(m):
        return {"N":0,"P50":np.nan,"P90":np.nan,"P95":np.nan,"mean":np.nan}
    d = np.hypot(det_x[m]-pred_x[m], det_y[m]-pred_y[m])
    return {
        "N": int(m.sum()),
        "P50": float(np.percentile(d,50)),
        "P90": float(np.percentile(d,90)),
        "P95": float(np.percentile(d,95)),
        "mean": float(np.nanmean(d)),
    }

def segments(mask):
    idx = np.where(mask)[0]
    if idx.size==0: return []
    cut = np.where(np.diff(idx)>1)[0]
    s = np.r_[0, cut+1]; e = np.r_[cut+1, idx.size]
    return [(idx[si], idx[ei-1]) for si,ei in zip(s,e)]

def continuity_metrics(accepted):
    acc = accepted>0.5
    n = len(acc)
    zeros = segments(~acc)
    ones  = segments(acc)
    gaps  = [e-s+1 for (s,e) in zeros]
    tracks= [e-s+1 for (s,e) in ones]
    return {
        "frames_total": int(n),
        "gaps_count": int(len(gaps)),
        "gap_len_mean": float(np.mean(gaps)) if gaps else np.nan,
        "gap_len_max": int(np.max(gaps)) if gaps else 0,
        "tracks_count": int(len(tracks)),
        "track_len_mean": float(np.mean(tracks)) if tracks else np.nan,
        "track_len_max": int(np.max(tracks)) if tracks else 0,
    }

def load_speed(p: Path, shift_us=0.0):
    if p is None or not p.exists(): return None, None
    rows=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("t_us","")!="":
                rows.append(r)
    t = np.array([float(r["t_us"])+shift_us for r in rows], np.float64)
    v = np.array([float(r["v_lin"]) if r.get("v_lin","") not in ("","nan") else np.nan for r in rows], np.float64)
    return t, v

def interp_to(ts_det, ts_gt, val_gt):
    order = np.argsort(ts_gt); ts_gt=ts_gt[order]; val_gt=val_gt[order]
    out = np.full_like(ts_det, np.nan, np.float64)
    if ts_gt.size==0: return out
    m = (ts_det>=ts_gt[0]) & (ts_det<=ts_gt[-1])
    out[m] = np.interp(ts_det[m], ts_gt, val_gt)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_csv", required=True, type=Path)
    ap.add_argument("--out_dir",  required=True, type=Path)
    ap.add_argument("--detections_topk_csv", type=Path, default=None,
                    help="If provided, use this to count frames_with_det (more accurate)")
    ap.add_argument("--gt_speed_csv", type=Path, default=None)
    ap.add_argument("--align_first", action="store_true")
    ap.add_argument("--gt_shift_us", type=float, default=0.0)
    ap.add_argument("--speed_bins", type=str, default="0,0.05,0.2,999")
    args = ap.parse_args()

    T = load_traj(args.traj_csv)
    topk_ts = load_topk_ts(args.detections_topk_csv)

    # Denominator for adoption rate (frames with measurements)
    if topk_ts is not None:
        # Use intersection of Top-K timestamps and trajectory timestamps
        traj_ts = T["t_us"].astype(np.int64)
        frames_with_det = int(np.isfinite(T["t_us"]).sum() and sum(int(t) in topk_ts for t in traj_ts))
    else:
        # Fallback: approximate using non-empty det_x
        frames_with_det = int(np.isfinite(T["det_x"]).sum())

    frames_accepted = int((T["accepted"]>0.5).sum())
    adoption_rate = float(frames_accepted / max(1, frames_with_det))

    out = {}
    out["adoption"] = {
        "frames_with_det": frames_with_det,
        "frames_accepted": frames_accepted,
        "adoption_rate": adoption_rate
    }

    det_j = jitter_rms(T["det_x"], T["det_y"])  # Note: only meaningful on accepted frames (conservative)
    kf_j  = jitter_rms(T["kf_x"],  T["kf_y"])
    out["jitter"] = {
        "det_rms_px": det_j,
        "kf_rms_px": kf_j,
        "improvement": float(1 - kf_j/max(1e-6,det_j))
    }

    out["innovation"] = innovation_stats(T["pred_x"], T["pred_y"], T["det_x"], T["det_y"], T["accepted"])
    out["continuity"] = continuity_metrics(T["accepted"])

    # speed-binned metrics
    bins = np.array([float(x) for x in args.speed_bins.split(",")], np.float64)
    if args.gt_speed_csv is not None:
        t_gt, v_gt = load_speed(args.gt_speed_csv, 0.0)
        if t_gt is not None:
            shift = 0.0
            if args.align_first and t_gt.size>0:
                shift = float(np.nanmin(T["t_us"]) - np.nanmin(t_gt))
            else:
                shift = float(args.gt_shift_us)
            v_det = interp_to(T["t_us"], t_gt+shift, v_gt)
            bucket=[]
            for i in range(len(bins)-1):
                lo,hi=bins[i],bins[i+1]
                m = (v_det>=lo) & (v_det<hi)
                if not np.any(m):
                    bucket.append({"range":[lo,hi]})
                    continue
                acc = (T["accepted"][m]>0.5)
                j_det = jitter_rms(T["det_x"][m], T["det_y"][m])
                j_kf  = jitter_rms(T["kf_x"][m],  T["kf_y"][m])
                bucket.append({
                    "range":[lo,hi],
                    "frames": int(m.sum()),
                    "adoption_rate": float(acc.mean()),
                    "det_jitter_rms_px": j_det,
                    "kf_jitter_rms_px": j_kf,
                    "improvement": float(1 - j_kf/max(1e-6,j_det)),
                })
            out["by_speed"] = {"bins": bins.tolist(), "buckets": bucket, "gt_shift_us": shift}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outp = args.out_dir/"metrics_stage3_tracking.json"
    with open(outp,"w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("[INFO] Written to：", outp)

if __name__ == "__main__":
    main()
