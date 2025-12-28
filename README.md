# Event-Based Target Tracking (SAE → Top-K → Kalman)

A reproducible pipeline that **consumes preprocessed event frames** (e.g., SAE) and outputs:
- Top-K candidate detections (CSV)
- Kalman-filtered trajectory (CSV)
- Evaluation metrics + plots + reports

## Input Contract (raw event parsing not included)
Raw event parsing is not included because formats vary across devices/simulators.
This repo focuses on **using event data** once adapted to a simple interface.

See: `Data/README.md`

---
# High-Speed Event-Based Single-Target Tracking (DAVIS)

A fully event-domain, reproducible tracking pipeline:
**SAE time-decay framing → Top-K + NMS candidates → CV Kalman filter with score-adaptive measurement noise + pixel gating**.

## Demos (MP4)
- Dynamic_6DoF: `assets/dynamic_6dof/demo_kf_cv_topk_gate50.mp4`
- Poster_6DoF: `assets/poster_6dof/demo_kf_gate50.mp4`
- Shapes_Rotation: `assets/shape_rotation/demo_kf.mp4`

## Key Results (from the dissertation report)
Evaluated on DAVIS (240×180): `shapes_rotation`, `poster_6dof`, `dynamic_6dof`.

| Dataset | Gate (px) | Adoption | Jitter RMS (raw → KF) | Notes |
|---|---:|---:|---:|---|
| Shapes_Rotation | 45 | 0.91 | 13.97 → 0.17 | Innovation P95 < 40 px |
| Poster_6DoF | 50 | 1.00 | 27.24 → 0.168 | Max track length 3213 |
| Dynamic_6DoF | 50 | 1.00 | 29.13 → 0.154 | Stable under rapid dynamics |

## Pipeline
**Stage 1 (SAE framing)**: time-decay surface with τ=10ms and Δt=2ms  
**Stage 2 (Candidates)**: threshold + morphology + components + NMS + Top-K (K=8)  
**Stage 3 (Tracking)**: constant-velocity Kalman filter with score-adaptive measurement noise, pixel-radius gating (45–50px), nearest-neighbour association.

---

## Repository Structure
- `src/` core implementation
- `scripts/` runnable entry scripts
- `Data/` dataset placeholders (raw data not redistributed)
- `assets/` generated outputs: videos, plots, metrics JSON, trajectory CSV
- `docs/` method + metrics definitions

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Main scripts (code navigation)
- `src/sae/stage2_candidates_topk.py` : Top-K + NMS candidate extraction
- `src/sae/stage3_kalman_tracker_topk.py` : Kalman tracking driven by Top-K candidates
- `src/sae/stage3_5_eval_tracking.py` : tracking evaluation (metrics + speed buckets)
- `src/sae/tools/stage2_topk_report.py` : stage2 report
- `src/sae/tools/stage3_7_plot_report.py` : summary plots/report
- `src/sae/tools/extract_worst_innovation.py` : worst-case (innovation) frame mining

