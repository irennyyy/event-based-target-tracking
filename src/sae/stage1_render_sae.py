#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: SAE / Time Surface rendering
Input:  four-column CSV [timestamp_us, x, y, polarity]
Output: under Data/<seq>/sae_frames/
  - sae_pos_{frameIdx}_{t_us}.npy / sae_neg_{...}.npy  (float32, [0,1])
  - optional: sae_fuse_{frameIdx}_{t_us}.npy           (float32, [0,1])
  - optional: keyframes/sae_{frameIdx}_{t_us}.png      (preview)
"""

import csv
import argparse
from pathlib import Path
import numpy as np

# Optional dependency: OpenCV is only used to export PNG previews.
# If not available, .npy saving is unaffected.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def fuse_polarity(S_pos: np.ndarray, S_neg: np.ndarray, mode: str = "max", alpha: float = 0.5) -> np.ndarray:
    if mode == "max":
        return np.maximum(S_pos, S_neg)
    elif mode == "weighted":
        return alpha * S_pos + (1.0 - alpha) * S_neg
    else:
        raise ValueError(f"Unknown fuse mode: {mode}")


def render_surface_numpy(tk_us: int, tau_us: float, T_pos: np.ndarray, T_neg: np.ndarray):
    """Render the time surface at timestamp tk_us; returns float32 arrays in [0,1]."""
    dtp = (tk_us - T_pos).astype(np.float64)
    dtn = (tk_us - T_neg).astype(np.float64)
    S_pos = np.zeros_like(dtp, dtype=np.float32)
    S_neg = np.zeros_like(dtn, dtype=np.float32)
    mp = dtp >= 0
    mn = dtn >= 0
    if np.any(mp):
        S_pos[mp] = np.exp(-(dtp[mp] / tau_us)).astype(np.float32)
    if np.any(mn):
        S_neg[mn] = np.exp(-(dtn[mn] / tau_us)).astype(np.float32)
    return S_pos, S_neg


def save_keyframe_png(out_dir: Path, frame_idx: int, t_us: int, S: np.ndarray):
    if not _HAS_CV2:
        return
    vis = (np.clip(S, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_path = out_dir / f"sae_{frame_idx:06d}_{t_us}.png"
    cv2.imwrite(str(out_path), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_csv", required=True, type=Path, help="Path to the four-column CSV")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory (recommend Data/<seq>/sae_frames)")
    ap.add_argument("--width",  type=int, default=240, help="Sensor width (DAVIS240=240, DAVIS346=346)")
    ap.add_argument("--height", type=int, default=180, help="Sensor height (DAVIS240=180, DAVIS346=260)")
    ap.add_argument("--dt_ms",  type=float, default=2.0, help="Rendering step Δt in milliseconds")
    ap.add_argument("--tau_ms", type=float, default=10.0, help="Exponential decay constant τ in milliseconds")
    ap.add_argument("--save_mode", choices=["posneg","fuse","both"], default="both", help="Save pos/neg, fused, or both")
    ap.add_argument("--fuse_mode", choices=["max","weighted"], default="max", help="Fusion mode for polarity")
    ap.add_argument("--fuse_alpha", type=float, default=0.5, help="Weight for 'weighted' fusion mode")
    ap.add_argument("--limit_frames", type=int, default=0, help="Save at most N frames (0 = no limit)")
    ap.add_argument("--keyframe_every", type=int, default=50, help="Save one PNG preview every N frames (0 = disable)")
    ap.add_argument("--start_us", type=int, default=None, help="Process only events with timestamp >= start_us (optional)")
    ap.add_argument("--end_us",   type=int, default=None, help="Process only events with timestamp <= end_us (optional)")
    args = ap.parse_args()

    W = int(args.width); H = int(args.height)
    tau_us = float(args.tau_ms * 1000.0)
    dt_us  = int(args.dt_ms * 1000.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.keyframe_every and _HAS_CV2:
        (out_dir / "keyframes").mkdir(parents=True, exist_ok=True)

    # Most recent event timestamps (in microseconds)
    T_pos = np.zeros((H, W), dtype=np.int64)
    T_neg = np.zeros((H, W), dtype=np.int64)

    frame_idx = 0
    saved = 0

    with open(args.events_csv, "r") as f:
        rdr = csv.reader(f)
        _ = next(rdr, None)  # skip header

        # Find the first event that satisfies start_us
        first = None
        for row in rdr:
            ts = int(row[0]); x = int(row[1]); y = int(row[2]); p = int(row[3])
            if args.start_us is not None and ts < args.start_us:
                continue
            first = (ts, x, y, p); break

        if first is None:
            print("[WARN] No events found in the specified time range."); return

        t0, x0, y0, p0 = first
        if 0 <= x0 < W and 0 <= y0 < H:
            (T_pos if p0==1 else T_neg)[y0, x0] = t0

        next_render_ts = t0

        # Main loop
        for row in rdr:
            ts = int(row[0]); x = int(row[1]); y = int(row[2]); p = int(row[3])
            if args.end_us is not None and ts > args.end_us:
                break

            # Render whenever the current event time surpasses the next rendering timestamp
            while ts >= next_render_ts:
                S_pos, S_neg = render_surface_numpy(next_render_ts, tau_us, T_pos, T_neg)

                if args.save_mode in ("posneg","both"):
                    np.save(out_dir / f"sae_pos_{frame_idx:06d}_{next_render_ts}.npy", S_pos)
                    np.save(out_dir / f"sae_neg_{frame_idx:06d}_{next_render_ts}.npy", S_neg)

                if args.save_mode in ("fuse","both"):
                    S_fuse = fuse_polarity(S_pos, S_neg, args.fuse_mode, args.fuse_alpha)
                    np.save(out_dir / f"sae_fuse_{frame_idx:06d}_{next_render_ts}.npy", S_fuse)

                if args.keyframe_every and _HAS_CV2 and (frame_idx % args.keyframe_every == 0):
                    vis = S_fuse if args.save_mode in ("fuse","both") else np.maximum(S_pos, S_neg)
                    save_keyframe_png(out_dir / "keyframes", frame_idx, next_render_ts, vis)

                frame_idx += 1
                saved += 1
                if args.limit_frames and saved >= args.limit_frames:
                    print(f"[INFO] Reached limit_frames={args.limit_frames}, stopping early."); return
                next_render_ts += dt_us

            # Update most recent timestamps with the current event
            if 0 <= x < W and 0 <= y < H:
                (T_pos if p == 1 else T_neg)[y, x] = ts

    print(f"[DONE] Saved {saved} frames to: {out_dir}")

if __name__ == "__main__":
    main()
