"""True (Vicon) path length and recorded duration per method; timeouts reported separately.
Usage: python -I path_length.py <phase-c alignment dir> <sweeps dir>"""
import csv, json, sys
from pathlib import Path
import numpy as np
ALIGN, SWEEPS = Path(sys.argv[1]), Path(sys.argv[2])
# Duality run 4 has no Vicon recording; its repeat from the same start (same outcome) is used in its place.
SUBSTITUTES = {("duality", 4): ("supplementary_duality_run04_B_attempt02", "supplementary_duality")}
for m, label in [("plain", "CBF"), ("rcbfqp", "R-CBF-QP"), ("duality", "Duality CBF"), ("nmrcbf", "NMR-CBF")]:
    rows = {int(r["run"]): r for r in csv.DictReader(open(SWEEPS / f"sweep_{m}" / "runs.csv"))}
    src = {i: (f"sweep_{m}", m) for i in rows}
    for (mm, i), (sweep, align) in SUBSTITUTES.items():
        if mm == m:
            rows[i], src[i] = next(csv.DictReader(open(SWEEPS / sweep / "runs.csv"))), (sweep, align)
    L, T, Lto, Tto = [], [], [], []
    for i, r in rows.items():
        rep = json.load(open(next((SWEEPS / src[i][0]).glob(f"run_{i:02d}_*/report.json"))))
        d = list((ALIGN / src[i][1]).glob(f"run_{i:02d}_*"))
        length = np.sum(np.hypot(*np.diff(np.load(d[0] / "gt.npy")[:, :2], axis=0).T)) if d else np.nan
        (L if r["outcome"] == "reached" else Lto).append(length)
        (T if r["outcome"] == "reached" else Tto).append(rep["duration_s"])
    print(f"{label:12s} reached {len(T):2d}: path median {np.nanmedian(L):.2f} m (range {np.nanmin(L):.2f}-{np.nanmax(L):.2f}), "
          f"time median {np.median(T):.1f} s, mean {np.mean(T):.1f} s"
          + (f" | timeouts {len(Tto)}: durations {[round(t, 1) for t in Tto]} s, paths {[round(x, 2) for x in Lto]} m" if Tto else ""))
