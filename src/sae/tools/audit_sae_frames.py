#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, json, numpy as np
from pathlib import Path

PAT = re.compile(r".*_(\d{6})_(\d{10,})\.npy$")  # idx, t_us

def main(sae_dir):
    sae_dir = Path(sae_dir)
    files = list(sae_dir.glob("sae_*_*.npy"))
    rec = {}
    for p in files:
        m = PAT.match(p.name)
        if not m: continue
        idx = int(m.group(1)); tus = int(m.group(2))
        rec.setdefault(idx, []).append(tus)

    idxs = sorted(rec.keys())
    all_ts = sorted({t for v in rec.values() for t in v})
    H, W = None, None

    # Dimension detection
    import numpy as np
    for p in files[:5]:
        try:
            a = np.load(p)
            H, W = a.shape
            break
        except Exception:
            pass

    # Δt statisti
    dt = np.diff(all_ts) if len(all_ts) > 1 else np.array([0])
    def stats(x):
        if len(x)==0: return {}
        x = np.array(x, dtype=np.float64)
        return {
            "count": int(x.size),
            "min": float(np.min(x)),
            "p50": float(np.percentile(x,50)),
            "p90": float(np.percentile(x,90)),
            "p95": float(np.percentile(x,95)),
            "mean": float(np.mean(x)),
            "max": float(np.max(x))
        }

    # Missing frame index
    missing = []
    if idxs:
        for k in range(min(idxs), max(idxs)+1):
            if k not in rec: missing.append(k)

    out = {
        "dir": str(sae_dir),
        "files": len(files),
        "frame_indices": {"min": idxs[0] if idxs else None, "max": idxs[-1] if idxs else None, "count": len(idxs)},
        "unique_t_us": len(all_ts),
        "dt_us_stats": stats(dt.tolist()),
        "shape": [H,W],
        "missing_indices": missing[:200],  # # Only list the first 200 to avoid being too long
        "prefix_coverage": {
            "fuse": len(list(sae_dir.glob("sae_fuse_*_*.npy"))),
            "pos":  len(list(sae_dir.glob("sae_pos_*_*.npy"))),
            "neg":  len(list(sae_dir.glob("sae_neg_*_*.npy"))),
        }
    }
    with open(sae_dir/"audit_sae.json","w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[INFO] 写出：", sae_dir/"audit_sae.json")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--sae_dir", required=True)
    args=ap.parse_args(); main(args.sae_dir)
