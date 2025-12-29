# Project Overview — Event-based Target Tracking (DAVIS)

## Goal
Build a practical single-target tracking pipeline for high-speed motion scenarios using event-based data.
The focus is on **how to use event data** once it is converted into a consistent intermediate form (e.g., SAE frames).

## Why event data
Frame cameras suffer motion blur at high speed. Event streams provide high temporal resolution and low latency,
making them suitable for fast motion tracking.

## Pipeline (3 stages)
1) **SAE framing (Stage 1)**  
   Convert event streams into time-sliced frames with time-decay (SAE-style representation).

2) **Candidate detection (Stage 2)**  
   Extract Top-K candidate detections per frame using thresholding + morphology + connected components + NMS.
   Output: `detections_topk.csv`.

3) **Tracking (Stage 3)**  
   Run a constant-velocity Kalman Filter (CV-KF) with:
   - nearest-neighbour association
   - pixel-radius gating (e.g., 45–50 px)
   - optional score-adaptive measurement noise
   Output: `trajectory_kf*.csv`, `metrics_stage3*.json`, overlays and plots.

## Outputs (portfolio artifacts)
For each sequence, the repo keeps lightweight artifacts under `assets/<SEQ>/`:
- demo video (`*.mp4`)
- diagnostic plots (`plot_*.png`)
- structured metrics (`metrics_*.json`)
- final trajectory (`trajectory_*.csv`)

## Reproducibility
A one-command run script is provided:
- `scripts/run_tracking_topk_kf.sh`
