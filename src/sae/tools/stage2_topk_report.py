#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 Top-K report (robust)
- Supports two CSV structures:
  A) Wide: one row = one frame, columns cand{i}_x / cand{i}_y / cand{i}_score
  B) Long: one row = one candidate, columns t_us, x, y, score
- Outputs:
  <out_dir>/metrics_stage2_topk.json
  <out_dir>/plot_topk_hist.png
  <out_dir>/plot_topk_timeseries.png
  <optional> <out_dir>/plot_score_hist.png (if score exists)
"""

import csv, json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _safe_float(s):
    try:
        v = float(s)
        if np.isnan(v): return np.nan
        return v
    except:
        return np.nan

def load_topk_any(csv_path: Path):
    """Return (t_us: np.array[N], K_per_frame: np.array[N], bestscore_per_frame: np.array[N or nan])"""
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        return np.array([]), np.array([]), np.array([])

    # Detect wide table: presence of cand{i}_x columns
    fields = rows[0].keys()
    idxs = sorted({int(m.group(1)) for k in fields
                   for m in [re.match(r"cand(\d+)_x", k)] if m})

    if idxs:
        # Wide table: one row = one frame
        ts = []
        Ks = []
        bests = []
        for r in rows:
            ts.append(_safe_float(r.get("t_us","nan")))
            k = 0
            best = np.nan
            for i in idxs:
                xs, ys = r.get(f"cand{i}_x",""), r.get(f"cand{i}_y","")
                if xs!="" and ys!="":
                    k += 1
                    sc = _safe_float(r.get(f"cand{i}_score","nan"))
                    if not np.isnan(sc):
                        best = sc if np.isnan(best) else max(best, sc)
            Ks.append(k)
            bests.append(best)
        return np.array(ts, dtype=np.float64), np.array(Ks, dtype=np.float64), np.array(bests, dtype=np.float64)

    # Otherwise try long table: one row per candidate, columns x,y,score (or cand_x/cand_y/score)
    # Group by t_us and compute K and best score per frame
    # Score may be missing
    by_ts = {}
    for r in rows:
        t = _safe_float(r.get("t_us","nan"))
        if np.isnan(t):  # skip rows without timestamp
            continue
        xs = r.get("x", r.get("cand_x",""))
        ys = r.get("y", r.get("cand_y",""))
        if xs=="" or ys=="":
            continue
        sc = _safe_float(r.get("score","nan"))
        if t not in by_ts:
            by_ts[t] = {"k":0, "best":np.nan}
        by_ts[t]["k"] += 1
        if not np.isnan(sc):
            if np.isnan(by_ts[t]["best"]):
                by_ts[t]["best"] = sc
            else:
                by_ts[t]["best"] = max(by_ts[t]["best"], sc)

    if not by_ts:
        # Fallback: unrecognized format
        return np.array([]), np.array([]), np.array([])

    ts_sorted = np.array(sorted(by_ts.keys()), dtype=np.float64)
    Ks = np.array([by_ts[t]["k"] for t in ts_sorted], dtype=np.float64)
    bests = np.array([by_ts[t]["best"] for t in ts_sorted], dtype=np.float64)
    return ts_sorted, Ks, bests

def save_hist(vals, bins, title, xlabel, ylabel, out_png, vlines=None):
    plt.figure(figsize=(12,8))
    plt.hist(vals, bins=bins)
    if vlines:
        for x, name in vlines:
            plt.axvline(x, ls="--")
            plt.text(x, plt.ylim()[1]*0.9, name, rotation=90, va="top")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def save_series(y, title, xlabel, ylabel, out_png):
    plt.figure(figsize=(12,6))
    if y.size == 0:
        plt.title(title + " (no data)")
    else:
        plt.plot(np.arange(y.size), y)
        plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main(csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ts, Ks, bests = load_topk_any(csv_path)

    frames_total = int(Ks.size)
    metrics = {
        "frames_total": frames_total,
        "k_mean": float(np.nanmean(Ks)) if frames_total>0 else None,
        "k_pcts": {
            "P50": float(np.nanpercentile(Ks,50)) if frames_total>0 else None,
            "P90": float(np.nanpercentile(Ks,90)) if frames_total>0 else None,
            "P95": float(np.nanpercentile(Ks,95)) if frames_total>0 else None,
            "P99": float(np.nanpercentile(Ks,99)) if frames_total>0 else None,
        },
        "bestscore_mean": (float(np.nanmean(bests)) if np.isfinite(bests).any() else None),
        "bestscore_pcts": {
            "P50": float(np.nanpercentile(bests,50)) if np.isfinite(bests).any() else None,
            "P90": float(np.nanpercentile(bests,90)) if np.isfinite(bests).any() else None,
            "P95": float(np.nanpercentile(bests,95)) if np.isfinite(bests).any() else None,
            "P99": float(np.nanpercentile(bests,99)) if np.isfinite(bests).any() else None,
        },
    }

    # Plot: K distribution
    if frames_total>0:
        kmax = int(np.nanmax(Ks))
        save_hist(Ks, bins=np.arange(-0.5, kmax+1.5, 1),
                  title="Candidates per frame (K)",
                  xlabel="K", ylabel="frames",
                  out_png=str(out_dir/"plot_topk_hist.png"),
                  vlines=[(metrics["k_pcts"]["P50"], "P50"),
                          (metrics["k_pcts"]["P90"], "P90"),
                          (metrics["k_pcts"]["P95"], "P95")] if metrics["k_pcts"]["P50"] is not None else None)
        # Plot: K time series
        save_series(Ks, title="K over frames", xlabel="frame idx", ylabel="K",
                    out_png=str(out_dir/"plot_topk_timeseries.png"))
        # Plot: best score distribution (if available)
        if np.isfinite(bests).any():
            save_hist(bests[np.isfinite(bests)], bins=50,
                      title="Best candidate score per frame",
                      xlabel="best score", ylabel="frames",
                      out_png=str(out_dir/"plot_bestscore_hist.png"))
    else:
        # Still output empty plot to avoid pipeline errors
        save_series(np.array([]), title="K over frames", xlabel="frame idx", ylabel="K",
                    out_png=str(out_dir/"plot_topk_timeseries.png"))

    with open(out_dir/"metrics_stage2_topk.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Written:", out_dir/"metrics_stage2_topk.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_topk_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    main(args.detections_topk_csv, args.out_dir)
