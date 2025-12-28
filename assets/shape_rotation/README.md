# shape_rotation — Result Artifacts (SAE → Top-K → KF)

Artifacts generated from the pipeline for quick portfolio review.

## Files
- `demo_kf.mp4` : tracking demo 
- `trajectory_kf.csv` : KF trajectory output
- `metrics_stage2_topk.json` : Stage2 candidate metrics
- `metrics_stage3_tracking.json` : Stage3 tracking metrics
- `plot_*.png` : diagnostic plots (Top-K / innovation / track length)

## Plot meanings
- `plot_topk_hist.png` / `plot_topk_timeseries.png` : candidate quality + stability over time
- `plot_innovation_hist.png` : KF residual distribution (stability)
- `plot_tracklen_hist.png` : continuity / track breaks
- `plot_bestscore_hist.png` : confidence profile

## Notes
Exact parameter settings are documented in the run script:
`scripts/run_tracking_topk_kf.sh`
