#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman Tracker (Top-K candidates)
- Visualization colors: red circle (gating), blue trajectory, yellow cross (measurement)
- Supports pixel radius gating (--gate_px) and Mahalanobis distance gating (--gate_mahal)
- Adaptive Q: based on GT speed (if provided) or KF’s own velocity
- Candidate selection strategy: nearest / score / hybrid (combined cost)
Outputs:
  <out_dir>/trajectory_kf.csv
  <out_dir>/overlays_kf/kf_XXXXXX_<t_us>.png
"""

import argparse, csv, os, re, json
from pathlib import Path
import numpy as np
import cv2

# ===Visualization colors (BGR)
COLOR_GATE  = (0, 0, 255)     # red circle
COLOR_TRACK = (255, 0, 0)     # blue trajectory
COLOR_MEAS  = (0, 255, 255)   # yellow cross

def draw_circle(img, c, r, color, t=2):
    cv2.circle(img, c, r, (0,0,0), t+2, lineType=cv2.LINE_AA)
    cv2.circle(img, c, r, color,  t,   lineType=cv2.LINE_AA)

def draw_polyline(img, pts, color, t=2):
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (0,0,0), t+2, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, color,    t,   lineType=cv2.LINE_AA)

def draw_x(img, x, y, color, size=6, t=2):
    p1=(x-size, y-size); p2=(x+size, y+size)
    p3=(x-size, y+size); p4=(x+size, y-size)
    cv2.line(img, p1, p2, (0,0,0), t+2, cv2.LINE_AA)
    cv2.line(img, p3, p4, (0,0,0), t+2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color,   t,   cv2.LINE_AA)
    cv2.line(img, p3, p4, color,   t,   cv2.LINE_AA)

# Data loading
def load_topk_csv(p: Path):
    """
   Compatible with two formats:
    1) Long table (from stage2_candidates_topk.py): multiple rows per frame
       Columns include at least: t_us, k, x, y, (score)
    2) Wide table: one row contains multiple cand{i}_x/cand{i}_y/cand{i}_score
    Returns:
       t_us:   np.ndarray[float64]  (sorted by time)
       topk:   List[List[Tuple[x,y,score]]]   aligned with t_us
    """
    with open(p) as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr if r.get("t_us","")!=""]
    if not rows:
        raise ValueError(f"CSV is empty：{p}")

    fieldset = set(rows[0].keys())

    # Case A: wide table (cand0_x, cand1_x, ...)
    cand_cols = sorted(
        {int(m.group(1))
         for col in fieldset
         for m in [re.match(r"cand(\d+)_x", col or "")]
         if m}
    )
    if cand_cols:
        t_list = []
        C = []
        for r in rows:
            t = float(r.get("t_us","nan"))
            cands = []
            for i in cand_cols:
                xs = r.get(f"cand{i}_x",""); ys = r.get(f"cand{i}_y",""); ss = r.get(f"cand{i}_score","")
                if xs!="" and ys!="":
                    s = float(ss) if ss not in ("","nan") else float("nan")
                    cands.append((float(xs), float(ys), s))
            t_list.append(t)
            C.append(cands)
        #  If t_us not monotonic, sort and realign
        order = np.argsort(np.array(t_list, dtype=np.float64))
        t_us = np.array(t_list, dtype=np.float64)[order]
        topk = [C[i] for i in order]
        return t_us, topk

    # Case B: long table (t_us, k, x, y, score)
    need = {"t_us","k","x","y"}
    if need.issubset(fieldset):
        by_t = {}
        for r in rows:
            try:
                t = float(r["t_us"])
                x = float(r["x"]); y = float(r["y"])
                s_raw = r.get("score","")
                s = float(s_raw) if s_raw not in ("","nan") else float("nan")
            except Exception:
                # skip invalid row
                continue
            by_t.setdefault(t, []).append((x, y, s))
        ts = sorted(by_t.keys())
        t_us = np.array(ts, dtype=np.float64)
        topk = [by_t[t] for t in ts]
        return t_us, topk

    # Other: unrecognized column format
    raise ValueError(
        f"Unrecognized CSV format. Expected one of:\n"
        f"  - Long table: t_us,k,x,y,(score)\n"
        f"  - Wide table: cand0_x,cand0_y,cand0_score,... + t_us\n"
        f"Got columns: {sorted(fieldset)}"
    )


def load_sae_frame(sae_dir: Path, idx: int, t: int, fallback_shape=(180,240)):
    """
    Load SAE fuse frame:
      1) Exact name: sae_fuse_{idx:06d}_{t}.npy
      2) Loose name: sae_fuse_{idx:06d}_*.npy / sae_pos_... / sae_neg_...
      3) If not found: search within ±10 frames for nearest available
      4) If still not found: return None (caller decides to skip visualization)
    """
    # 1) Exact filename
    exact = sae_dir / f"sae_fuse_{idx:06d}_{int(t)}.npy"
    if exact.exists():
        a = np.load(str(exact))
        img = (np.clip(a*255.0, 0, 255).astype(np.uint8))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 2) Same idx, any timestamp (fuse/pos/neg)
    pats = [
        f"sae_fuse_{idx:06d}_*.npy",
        f"sae_pos_{idx:06d}_*.npy",
        f"sae_neg_{idx:06d}_*.npy",
    ]
    for pat in pats:
        ms = sorted(sae_dir.glob(pat))
        if ms:
            a = np.load(str(ms[0]))
            img = (np.clip(a*255.0, 0, 255).astype(np.uint8))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 3) Search ±10 frames
    for d in range(1, 11):
        for j in (idx-d, idx+d):
            if j < 0: continue
            for pat in [f"sae_fuse_{j:06d}_*.npy", f"sae_pos_{j:06d}_*.npy", f"sae_neg_{j:06d}_*.npy"]:
                ms = sorted(sae_dir.glob(pat))
                if ms:
                    a = np.load(str(ms[0]))
                    img = (np.clip(a*255.0, 0, 255).astype(np.uint8))
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 4) Nothing found
    return None


def load_speed_csv(p: Path, shift_us: float = 0.0):
    """read ground_truth_speed.csv"""
    with open(p) as f:
        rdr = csv.DictReader(f); rows = [r for r in rdr if r.get("t_us","")!=""]
    t = np.array([float(r["t_us"]) for r in rows], dtype=np.float64) + shift_us
    v = np.array([float(r["v_lin"]) if r.get("v_lin","") not in ("","nan") else np.nan for r in rows], dtype=np.float64)
    return t, v

def interp_to(ts_det, ts_gt, val_gt):
    order = np.argsort(ts_gt); ts_gt=ts_gt[order]; val_gt=val_gt[order]
    out = np.full_like(ts_det, np.nan, dtype=np.float64)
    if ts_gt.size==0: return out
    mask = (ts_det>=ts_gt[0]) & (ts_det<=ts_gt[-1])
    out[mask] = np.interp(ts_det[mask], ts_gt, val_gt)
    return out

# ==KF
def kf_mats(dt: float):
    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1, 0],
                  [0,0,0, 1]], dtype=np.float64)
    H = np.array([[1,0,0,0],
                  [0,1,0,0]], dtype=np.float64)
    G = np.array([[0.5*dt*dt, 0],
                  [0, 0.5*dt*dt],
                  [dt, 0],
                  [0, dt]], dtype=np.float64)
    return F, H, G

def q_from_params(q_base: float, q_alpha: float, v: float, q_min: float, q_max: float):
    q = q_base * (1.0 + q_alpha * max(0.0, float(v)))
    return float(np.clip(q, q_min, q_max))

def sigma_from_score(r_base: float, r_min: float, r_max: float, score: float,
                     alpha: float, s0: float):
    if np.isnan(score): return r_base
    sig = r_base * np.exp(-alpha*(score - s0))
    return float(np.clip(sig, r_min, r_max))

# ======Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_topk_csv", required=True, type=Path)
    ap.add_argument("--out_dir",              required=True, type=Path)
    ap.add_argument("--dt_fallback_ms", type=float, default=2.0)

    # KF noise
    ap.add_argument("--q_process", type=float, default=10.0)
    ap.add_argument("--use_adaptive_q", action="store_true",
                    help="Adaptive Q based on speed：Q = q * (1 + q_alpha * v)")
    ap.add_argument("--q_alpha", type=float, default=2.0)
    ap.add_argument("--q_min", type=float, default=1.0)
    ap.add_argument("--q_max", type=float, default=30.0)

    ap.add_argument("--r_base", type=float, default=3.0)
    ap.add_argument("--r_min",  type=float, default=0.5)
    ap.add_argument("--r_max",  type=float, default=10.0)
    ap.add_argument("--use_score_R", action="store_true",
                    help="Scale R exponentially based on score")
    ap.add_argument("--r_alpha", type=float, default=2.0)
    ap.add_argument("--r_s0",    type=float, default=0.5)

    # Candidate Selection
    ap.add_argument("--prefer", choices=["nearest","score","hybrid"], default="nearest")
    ap.add_argument("--prefer_weight", type=float, default=0.7,
                    help="Weight for hybrid: cost=α*distance + (1-α)*(1-normalized score)")

    # Gating
    ap.add_argument("--gate_px", type=float, default=40.0)
    ap.add_argument("--gate_mahal", type=float, default=0.0,
                    help=">0 enables Mahalanobis gating (2D χ² thresholds e.g. 5.99/9.21)")

    # Adaptive Q can use GT speed
    ap.add_argument("--gt_speed_csv", type=Path, default=None)
    ap.add_argument("--gt_shift_us", type=float, default=0.0)
    ap.add_argument("--align_first", action="store_true")

    # visualisation
    ap.add_argument("--sae_dir", type=Path, default=None)
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "overlays_kf"
    vis_dir.mkdir(parents=True, exist_ok=True)

    t_us, topk = load_topk_csv(args.detections_topk_csv)

    # Speed (for adaptive Q)
    v_series = np.full_like(t_us, np.nan)
    if args.use_adaptive_q and args.gt_speed_csv is not None and args.gt_speed_csv.exists():
        t_gt, v_gt = load_speed_csv(args.gt_speed_csv, shift_us=0.0)
        if args.align_first and t_gt.size>0:
            shift = float(np.nanmin(t_us) - np.nanmin(t_gt))
        else:
            shift = args.gt_shift_us
        t_gt_s = t_gt + shift
        v_series = interp_to(t_us, t_gt_s, v_gt)

    # KF init
    N = len(t_us)
    dt_fallback = args.dt_fallback_ms / 1000.0
    x = np.zeros((4,1), dtype=np.float64)   # [x,y,vx,vy]
    P = np.eye(4, dtype=np.float64) * 1000.0
    inited = False

    # record
    traj = []  # per frame：t_us, pred_x, pred_y, kf_x, kf_y, det_x, det_y, score, accepted(0/1), sigma_obs, dt_ms
    track_hist = []

    accepted_cnt=0; det_present_cnt=0

    for i in range(N):
        t = t_us[i]
        dt = dt_fallback if i==0 or np.isnan(t_us[i-1]) else max(1e-6, (t_us[i]-t_us[i-1])/1e6)
        F,H,G = kf_mats(dt)

        # Prediction step
        if not inited:
            # Initialize with first frame candidates
            if len(topk[i])>0:
                x[:2,0] = topk[i][0][0], topk[i][0][1]
                P = np.eye(4)*50.0
                inited = True
            pred_x, pred_y = float(x[0,0]), float(x[1,0])
        else:
            # Q：adaptive or constant
            if args.use_adaptive_q:
                if not np.isnan(v_series[i]):
                    v_for_q = v_series[i]
                else:
                    v_for_q = float(np.hypot(x[2,0], x[3,0]))  # fallback: KF’s own velocity
                q = q_from_params(args.q_process, args.q_alpha, v_for_q, args.q_min, args.q_max)
            else:
                q = args.q_process
            Q = (G @ (np.eye(2)*q) @ G.T)

            x = F @ x
            P = F @ P @ F.T + Q
            pred_x, pred_y = float(x[0,0]), float(x[1,0])

        # Candidate selection
        cand = topk[i]
        det_x = det_y = score = np.nan
        accepted = 0
        sigma_obs = np.nan

        if len(cand)>0:
            det_present_cnt += 1
            zs = np.array([[c[0], c[1]] for c in cand], dtype=np.float64)   # Kx2
            ss = np.array([c[2] for c in cand], dtype=np.float64)           # K
            # Distances/Mahalanobis
            Hx = np.array([pred_x, pred_y], dtype=np.float64)
            diffs = zs - Hx[None,:]

            in_gate = np.ones(len(cand), dtype=bool)
            if args.gate_mahal > 0 and inited:
                # Approximate R using mean score (only for gating threshold, not update)
                s_mean = float(np.nanmean(ss)) if np.isfinite(ss).any() else np.nan
                sig_tmp = sigma_from_score(args.r_base, args.r_min, args.r_max,
                                           s_mean, args.r_alpha, args.r_s0) if args.use_score_R else args.r_base
                R_tmp = np.eye(2)* (sig_tmp**2)
                S = H @ P @ H.T + R_tmp
                Sinv = np.linalg.inv(S)
                d2 = np.array([d.T @ Sinv @ d for d in diffs])  # K
                in_gate &= (d2 <= args.gate_mahal)
                # Visualization radius: largest eigenvalue of S * threshold
                eigvals,_ = np.linalg.eig(S)
                gate_draw_px = int(np.sqrt(float(np.max(eigvals)) * args.gate_mahal))
            else:
                gate_draw_px = int(args.gate_px)
                if args.gate_px > 0:
                    d2 = np.sum(diffs*diffs, axis=1)
                    in_gate &= (d2 <= (args.gate_px**2))

            # choose candidate
            if np.any(in_gate):
                idxs = np.where(in_gate)[0]
                if args.prefer == "nearest":
                    j = idxs[np.argmin(np.sum(diffs[idxs]**2, axis=1))]
                elif args.prefer == "score":

                    j = idxs[np.nanargmax(ss[idxs])]
                else:
                    # # hybrid: cost = α*normalized distance + (1-α)*(1-normalized score)
                    d = np.sqrt(np.sum(diffs[idxs]**2, axis=1))
                    d = (d - d.min()) / max(1e-6, (d.max()-d.min()))
                    s = ss[idxs]

                    if np.isfinite(s).any():
                        s = (s - np.nanmin(s)) / max(1e-6, (np.nanmax(s)-np.nanmin(s)))
                        s = np.nan_to_num(s, nan=0.5)
                    else:
                        s = np.full_like(d, 0.5)
                    cost = args.prefer_weight*d + (1.0-args.prefer_weight)*(1.0-s)
                    j = idxs[np.argmin(cost)]

                det_x, det_y, score = float(zs[j,0]), float(zs[j,1]), float(ss[j])
                accepted = 1

        # uodate step
        if inited and accepted==1:
            # noise measurement
            if args.use_score_R:
                sigma_obs = sigma_from_score(args.r_base, args.r_min, args.r_max,
                                             score, args.r_alpha, args.r_s0)
            else:
                sigma_obs = args.r_base
            R = np.eye(2)*(sigma_obs**2)

            z = np.array([[det_x],[det_y]], dtype=np.float64)
            y = z - (H @ x)                       # innovation
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ y)
            P = (np.eye(4) - K @ H) @ P

        # record trajectory
        kfx, kfy = float(x[0,0]), float(x[1,0])
        track_hist.append((kfx,kfy))
        traj.append({
            "frame_idx": i, "t_us": int(t),
            "pred_x": pred_x, "pred_y": pred_y,
            "kf_x": kfx, "kf_y": kfy,
            "det_x": det_x, "det_y": det_y,
            "score": score, "accepted": int(accepted),
            "sigma_obs": sigma_obs, "dt_ms": dt*1000.0
        })
        if accepted==1: accepted_cnt += 1

        # visualisation
        if args.sae_dir is not None and (i % max(1,args.save_every) == 0):
            img = load_sae_frame(args.sae_dir, i, int(t))
            if img is None:

                continue
            # red circle
            gate_r = int(args.gate_px) if args.gate_mahal<=0 else int(gate_draw_px)
            if gate_r>0:
                draw_circle(img, (int(kfx), int(kfy)), gate_r, COLOR_GATE, t=2)
            # blue trajectory
            pts = np.array([[int(px),int(py)] for (px,py) in track_hist[-300:]], np.int32).reshape(-1,1,2)
            draw_polyline(img, pts, COLOR_TRACK, t=2)
            # Yellow
            if accepted==1 and np.isfinite(det_x) and np.isfinite(det_y):
                draw_x(img, int(det_x), int(det_y), COLOR_MEAS, size=6, t=2)

            outp = vis_dir / f"kf_{i:06d}_{int(t)}.png"
            cv2.imwrite(str(outp), img)

    # save trajectory
    out_csv = args.out_dir / "trajectory_kf.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader(); w.writerows(traj)
    print(f"[INFO] KF trajectory written to：{out_csv}")
    print(f"[INFO] visualisation output to：{vis_dir}")
    if det_present_cnt>0:
        print(f"[INFO] statistics {det_present_cnt}，adoption {accepted_cnt}，adoption rate {accepted_cnt/det_present_cnt:.2f}")

if __name__ == "__main__":
    main()
