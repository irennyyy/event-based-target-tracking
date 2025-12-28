#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import csv

def load_gt_txt(path: Path):
    # groundtruth.txt: timestamp[s] px py pz qx qy qz qw, possible #
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts[:8]])
    data = np.array(rows, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 8:
        raise ValueError(f"Unexpected columns in {path}, expect 8, got {data.shape[1]}")
    t_s = data[:, 0]
    pose = data[:, 1:]  # px py pz qx qy qz qw
    return t_s, pose

def quat_to_yaw(qx, qy, qz, qw):
    # ZYX (yaw-pitch-roll), only take yaw
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return np.arctan2(siny_cosp, cosy_cosp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_txt", required=True, type=Path,
                    help="groundtruth.txt (timestamp px py pz qx qy qz qw), time in seconds")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output directory (e.g. Data/shapes_rotation)")
    ap.add_argument("--make_speeds", action="store_true",
                    help="Also export linear/angular velocity estimates (simple differences)")
    args = ap.parse_args()

    t_s, pose = load_gt_txt(args.gt_txt)
    px, py, pz, qx, qy, qz, qw = pose.T
    t_us = (t_s * 1e6).astype(np.int64)

    # 1) Pose CSV (microsecond t_us, easier to align with detections)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pose = args.out_dir / "ground_truth_pose.csv"
    with open(out_pose, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us","px","py","pz","qx","qy","qz","qw"])
        for i in range(len(t_us)):
            w.writerow([int(t_us[i]), f"{px[i]:.9f}", f"{py[i]:.9f}", f"{pz[i]:.9f}",
                        f"{qx[i]:.9f}", f"{qy[i]:.9f}", f"{qz[i]:.9f}", f"{qw[i]:.9f}"])
    print(f"[INFO] Written: {out_pose}")

    if args.make_speeds:
        dt = np.diff(t_s)
        # Linear velocity (m/s): difference in world coordinates
        vx = np.r_[np.nan, np.diff(px)/dt]
        vy = np.r_[np.nan, np.diff(py)/dt]
        vz = np.r_[np.nan, np.diff(pz)/dt]
        # Yaw angular velocity (rad/s): take yaw from quaternion, then differentiate
        yaw = quat_to_yaw(qx, qy, qz, qw)
        wyaw = np.r_[np.nan, np.diff(yaw)/dt]
        out_spd = args.out_dir / "ground_truth_speed.csv"
        with open(out_spd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_us","v_lin","vx","vy","vz","yaw_rate"])
            for i in range(len(t_us)):
                v_lin = np.nan
                if i>0:
                    v_lin = (vx[i]**2 + vy[i]**2 + vz[i]**2)**0.5
                w.writerow([int(t_us[i]),
                            "" if np.isnan(v_lin) else f"{v_lin:.6f}",
                            "" if np.isnan(vx[i]) else f"{vx[i]:.6f}",
                            "" if np.isnan(vy[i]) else f"{vy[i]:.6f}",
                            "" if np.isnan(vz[i]) else f"{vz[i]:.6f}",
                            "" if np.isnan(wyaw[i]) else f"{wyaw[i]:.6f}"])
        print(f"[INFO] Written: {out_spd}")

if __name__ == "__main__":
    main()
