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
- `src/` pipeline source code
- `scripts/` runnable entrypoints
- `Data/` input interface (datasets not committed)
- `docs/` documentation / role lens (optional)
- `assets/` demo figures (optional)

---

## Quickstart
### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
