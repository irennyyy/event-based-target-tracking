# Event-Based Target Tracking (SAE → Top-K → Kalman)

A reproducible pipeline that **consumes preprocessed event frames** (e.g., SAE) and outputs:
- Top-K candidate detections (CSV)
- Kalman-filtered trajectory (CSV)
- Evaluation metrics + plots + reports

**Portfolio positioning (UK roles):** Data Engineer / Applied Data Scientist / Technology Consultant

---

## Input Contract (raw event parsing not included)
Raw event parsing is not included because formats vary across devices/simulators.
This repo focuses on **using event data** once adapted to a simple interface.

See: `Data/README.md`

---

## Repo Structure
- `src/sae/` pipeline scripts
- `src/sae/tools/` reporting / overlays / audits
- `src/sae/utils/` video helpers
- `scripts/` one-command run scripts
- `Data/` data interface (datasets not committed)
- `docs/` documentation (optional)
- `assets/` demo figures (optional)

---

## Quickstart
### 1) Install
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

