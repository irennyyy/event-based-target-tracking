#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Kalman Filter (Constant Velocity) + Gating
Input:  detections.csv  columns: t_us,x,y,w,h,score,pol_mode (x/y may be empty)
Output: trajectory_kf.csv; optional overlays_kf/*.png visualizations
"""

import csv
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import glob


# ---------- IO ----------

def read_detections(p: Path):
    """Read detections.csv, sort by time & deduplicate (keep last entry per t_us)."""
    with open(p, "r") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    def col(name):
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v in ("", "nan", None):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
        return np.array(vals, dtype=np.float64)

    t = col("t_us").astype(np.float64)
    x = col("x"); y = col("y")
    w = col("w"); h = col("h")
    score = col("score")

    order = np.argsort(t)
    t, x, y, w, h, score = t[order], x[order], y[order], w[order], h[order], score[order]

    uniq = {}
    for ti, xi, yi, wi, hi, si in zip(t, x, y, w, h, score):
        uniq[ti] = (xi, yi, wi, hi, si)

    ts = np.array(sorted(uniq.keys()), dtype=np.float64)
    X  = np.array([uniq[ti][0] for ti in ts])
    Y  = np.array([uniq[ti][1] for ti in ts])
    W  = np.array([uniq[ti][2] for ti in ts])
    H  = np.array([uniq[ti][3] for ti in ts])
    S  = np.array([uniq[ti][4] for ti in ts])
    return ts, X, Y, W, H, S


# ---------- KF primitives ----------

def build_F_Q(dt: float, q: float):
    """F and Q for CV model (white noise acceleration spectral density q, units ~ px^2/s^3)."""
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0,  dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float64)
    dt2 = dt*dt; dt3 = dt2*dt; dt4 = dt2*dt2
    Q = q*np.array([[dt4/4,   0,     dt3/2, 0     ],
                    [0,     dt4/4,   0,     dt3/2 ],
                    [dt3/2,   0,     dt2,   0     ],
                    [0,     dt3/2,   0,     dt2   ]], dtype=np.float64)
    return F, Q


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


# ---------- Overlay visualization ----------

def find_sae_frame(sae_dir: Path, t_us: int) -> Optional[Tuple[Path, ...]]:
    """Find SAE by timestamp (prefer fuse *.npy, then png; fallback to pos/neg)."""
    p = glob.glob(str(sae_dir / f"sae_fuse_*_{t_us}.npy"))
    if p: return (Path(p[0]),)
    p = glob.glob(str(sae_dir / f"sae_fuse_*_{t_us}.png"))
    if p: return (Path(p[0]),)
    ppos = glob.glob(str(sae_dir / f"sae_pos_*_{t_us}.npy"))
    pneg = glob.glob(str(sae_dir / f"sae_neg_*_{t_us}.npy"))
    if ppos and pneg: return (Path(ppos[0]), Path(pneg[0]))
    return None


def render_overlay(sae_any: Tuple[Path, ...],
                   det_xy, kf_xy,
                   track_pts: List[Tuple[int, int]],
                   out_path: Path):
    """Overlay KF/detections on SAE and save."""
    import cv2

    if len(sae_any) == 1:
        p = sae_any[0]
        if p.suffix.lower() in (".npy", ".npz"):
            S = np.load(p)
            if isinstance(S, np.lib.npyio.NpzFile):
                for k in ["arr_0", "sae", "data"]:
                    if k in S:
                        S = S[k]; break
            S = S.astype(np.float32)
        else:
            S = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    else:
        a = np.load(sae_any[0]).astype(np.float32)
        b = np.load(sae_any[1]).astype(np.float32)
        S = np.maximum(a, b)

    S = np.clip(S, 0, 1)
    img = (S * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if len(track_pts) >= 2:
        cv2.polylines(img, [np.array(track_pts, dtype=np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)

    if kf_xy is not None:
        cx, cy = map(int, np.round(kf_xy))
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    if det_xy is not None:
        dx, dy = map(int, np.round(det_xy))
        cv2.drawMarker(img, (dx, dy), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1)

    cv2.imwrite(str(out_path), img)


# ---------- Main Process ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    # KF hyperparameters
    ap.add_argument("--dt_fallback_ms", type=float, default=2.0, help="Fallback Δt (ms) when timestamps are abnormal")
    ap.add_argument("--q_process", type=float, default=10.0, help="Process noise spectral density q (larger allows more flexibility)")
    ap.add_argument("--r_base", type=float, default=3.0, help="Base observation noise σ0 (pixels)")
    ap.add_argument("--r_min", type=float, default=0.5, help="Lower bound on observation noise σ_min")
    ap.add_argument("--r_max", type=float, default=10.0, help="Upper bound on observation noise σ_max")
    ap.add_argument("--use_score_R", action="store_true", help="Use score to adapt observation noise R: σ = clip(r_base/score)")

    #  Gating parameters
    ap.add_argument("--gate_px", type=float, default=25.0,
                    help="Pixel distance gating，|z - Hx_pred| > gate_px then reject measurement（0 disables）")
    ap.add_argument("--gate_mahal", type=float, default=0.0,
                    help="Mahalanobis distance gating threshold (χ²(2) value, e.g. 5.99≈95%%; 0 disables)")

    ap.add_argument("--miss_max_frames", type=int, default=50, help="Mark as long-missed after this many consecutive missed frames (record only)")

    # Optional visualization
    ap.add_argument("--sae_dir", type=Path, default=None, help="If provided, generate overlay visualizations")
    ap.add_argument("--save_every", type=int, default=50, help="Save one visualization every N frames")
    ap.add_argument("--history_len", type=int, default=200, help="Number of past points to show in overlay trajectory history")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    if args.sae_dir is not None:
        ensure_dir(args.out_dir / "overlays_kf")

    # Read detections
    t_us, dx, dy, dw, dh, sc = read_detections(args.detections_csv)
    n = t_us.size
    if n == 0:
        raise RuntimeError("detections.csv is empty")

    out_csv = args.out_dir / "trajectory_kf.csv"
    fw = open(out_csv, "w", newline="")
    wr = csv.writer(fw)
    wr.writerow(["t_us","x_kf","y_kf","vx_kf","vy_kf","det_x","det_y","det_score","sigma_obs","missed","dt_s"])

    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    xk = None; Pk = None; prev_t = None
    miss_cnt = 0
    track_pts : List[Tuple[int,int]] = []

    for i in range(n):
        t = float(t_us[i])

        # Δt
        if prev_t is None:
            dt = args.dt_fallback_ms/1000.0
        else:
            dt = (t - prev_t)/1e6
            if not np.isfinite(dt) or dt <= 1e-9 or dt > 0.1:
                dt = args.dt_fallback_ms/1000.0
        F, Q = build_F_Q(dt, args.q_process)

        # Initialization: wait for first valid detection
        if xk is None:
            det_valid = (not np.isnan(dx[i])) and (not np.isnan(dy[i]))
            prev_t = t
            if det_valid:
                xk = np.array([dx[i], dy[i], 0.0, 0.0], dtype=np.float64)
                Pk = np.diag([100.0, 100.0, 1000.0, 1000.0]).astype(np.float64)
                miss_cnt = 0
                wr.writerow([int(t), f"{xk[0]:.3f}", f"{xk[1]:.3f}", f"{xk[2]:.3f}", f"{xk[3]:.3f}",
                             f"{dx[i]:.3f}", f"{dy[i]:.3f}", f"{(0.0 if np.isnan(sc[i]) else sc[i]):.6f}",
                             "", 0, f"{dt:.6f}"])
                track_pts.append((int(round(xk[0])), int(round(xk[1]))))
            else:
                wr.writerow([int(t), "", "", "", "", "", "", "", "", 1, f"{dt:.6f}"])
            continue

        # Prediction
        x_pred = F @ xk
        P_pred = F @ Pk @ F.T + Q

        det_valid = (not np.isnan(dx[i])) and (not np.isnan(dy[i]))
        if det_valid:
            # Adaptive observation noise
            if args.use_score_R and not np.isnan(sc[i]):
                sigma = args.r_base / max(1e-6, sc[i])
            else:
                sigma = args.r_base
            sigma = float(np.clip(sigma, args.r_min, args.r_max))
            R = np.array([[sigma*sigma, 0.0],[0.0, sigma*sigma]], dtype=np.float64)

            z = np.array([dx[i], dy[i]], dtype=np.float64)
            y = z - (H @ x_pred)              # innovation (pixels)

            # ------------ Gate 1: pixel distance ------------
            if args.gate_px > 0.0 and np.linalg.norm(y) > args.gate_px:
                det_valid = False

            # ------------ Gate 2: Mahalanobis distance ------------
            if det_valid and args.gate_mahal > 0.0:
                S = H @ P_pred @ H.T + R
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    Sinv = np.linalg.pinv(S)
                mahal2 = float(y.T @ Sinv @ y)
                if mahal2 > args.gate_mahal:
                    det_valid = False

            if det_valid:
                S = H @ P_pred @ H.T + R
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    Sinv = np.linalg.pinv(S)
                K = P_pred @ H.T @ Sinv
                xk = x_pred + K @ y
                Pk = (I - K @ H) @ P_pred
                miss_cnt = 0
                sigma_used = sigma
            else:
                # Reject measurement: prediction only
                xk = x_pred
                Pk = P_pred
                miss_cnt += 1
                sigma_used = math.nan
        else:
            # No measurement: prediction only
            xk = x_pred
            Pk = P_pred
            miss_cnt += 1
            sigma_used = math.nan

        prev_t = t
        wr.writerow([int(t),
                     f"{xk[0]:.3f}", f"{xk[1]:.3f}", f"{xk[2]:.3f}", f"{xk[3]:.3f}",
                     "" if np.isnan(dx[i]) else f"{dx[i]:.3f}",
                     "" if np.isnan(dy[i]) else f"{dy[i]:.3f}",
                     "" if np.isnan(sc[i]) else f"{sc[i]:.6f}",
                     "" if np.isnan(sigma_used) else f"{sigma_used:.3f}",
                     miss_cnt, f"{dt:.6f}"])

        # Visualization
        if args.sae_dir is not None and args.save_every > 0 and (i % args.save_every == 0):
            sae = find_sae_frame(args.sae_dir, int(t))
            det_xy = None if np.isnan(dx[i]) or np.isnan(dy[i]) else (dx[i], dy[i])
            kf_xy = (xk[0], xk[1])
            track_pts.append((int(round(xk[0])), int(round(xk[1]))))
            if len(track_pts) > args.history_len:
                track_pts = track_pts[-args.history_len:]
            if sae is not None:
                outp = args.out_dir / "overlays_kf" / f"kf_{i:06d}_{int(t)}.png"
                render_overlay(sae, det_xy, kf_xy, track_pts, outp)

    fw.close()
    print(f"[INFO] KF trajectory written：{out_csv}")
    if args.sae_dir is not None:
        print(f"[INFO] Visualization output directory：{args.out_dir/'overlays_kf'}")


if __name__ == "__main__":
    main()
