#!/usr/bin/env bash
set -e

SEQ="${1:-shapes_rotation}"
BASE="$(pwd)"
DATA_DIR="$BASE/Data"
EXP_DIR="$DATA_DIR/${SEQ}_exp/kf_cv_topk"

python "$BASE/src/sae/stage2_candidates_topk.py" \
  --sae_dir "$DATA_DIR/$SEQ/sae_frames" \
  --out_csv "$DATA_DIR/$SEQ/detections_topk.csv" \
  --topk 8 --thr 0.25 --close 3 --min_area 12 --nms_radius 8

python "$BASE/src/sae/stage3_kalman_tracker_topk.py" \
  --detections_topk_csv "$DATA_DIR/$SEQ/detections_topk.csv" \
  --out_dir "$EXP_DIR" \
  --dt_fallback_ms 2.0 \
  --q_process 10.0 \
  --r_base 3.5 --r_min 0.5 --r_max 10.0 --use_score_R --r_alpha 2.0 --r_s0 0.5 \
  --prefer nearest \
  --gate_px 45 --gate_mahal 0 \
  --sae_dir "$DATA_DIR/$SEQ/sae_frames" \
  --save_every 50

python "$BASE/src/sae/stage3_5_eval_tracking.py" \
  --traj_csv "$EXP_DIR/trajectory_kf.csv" \
  --out_dir "$EXP_DIR" \
  --gt_speed_csv "$DATA_DIR/$SEQ/ground_truth_speed.csv" \
  --align_first --speed_bins "0,0.05,0.2,999"

python "$BASE/src/sae/tools/stage2_topk_report.py" \
  --detections_topk_csv "$DATA_DIR/$SEQ/detections_topk.csv" \
  --out_dir "$EXP_DIR"

python "$BASE/src/sae/tools/stage3_7_plot_report.py" \
  --traj_csv "$EXP_DIR/trajectory_kf.csv" \
  --metrics_json "$EXP_DIR/metrics_stage3_tracking.json" \
  --out_dir "$EXP_DIR"

echo "Done. Outputs in: $EXP_DIR"
