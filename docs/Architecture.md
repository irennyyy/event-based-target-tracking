# Architecture & Data Contract

This repo is designed as a **reusable pipeline**. Upstream parsing is intentionally decoupled.

## Data contract (what you need to provide)
For each sequence `<SEQ>`:

Data/<SEQ>/
  sae_frames/                 # REQUIRED: SAE frames (png/jpg), ordered
  ground_truth_speed.csv      # OPTIONAL: required only for speed-bucket evaluation

The repo will generate:
- Data/<SEQ>/detections_topk.csv
- Data/<SEQ>_exp/... (tracking outputs, metrics, plots, overlays)

## Stage 1 — SAE framing (optional / adapter)
Input: raw event streams (bag/txt/SDK export)  
Output: `sae_frames/` (images)

Reason for decoupling:
- different event cameras / simulators export different formats
- the tracking pipeline should remain stable regardless of upstream format

## Stage 2 — Top-K candidates
Input: `sae_frames/`  
Output:
- `detections_topk.csv` (candidate detections)
- candidate metrics JSON (optional)
- plots for detection diagnostics (optional)

Key operations:
- thresholding on SAE frame intensity
- morphology (close)
- connected components + filtering by area
- NMS + keep Top-K per frame

## Stage 3 — Kalman tracking (Top-K driven)
Input:
- `detections_topk.csv`
- `sae_frames/` (optional, for overlays)
Output:
- `trajectory_kf*.csv` (final track)
- `metrics_stage3_tracking*.json`
- overlays and plots

Key ideas:
- constant-velocity Kalman Filter (CV-KF)
- nearest-neighbour association
- pixel gating radius (gate_px)
- optional score-adaptive measurement noise to improve robustness

## Entry points
- Main pipeline scripts: `src/sae/`
- Reports/tools: `src/sae/tools/`
- One-command run: `scripts/run_tracking_topk_kf.sh`
