"""R-CBF-QP: share of steps with essentially no tightening (phi < 0.01, the minimum-gain fallback) vs. the
along-track error bound, over steps where the estimated control point is within 2 m of an obstacle."""
import sys
from pathlib import Path
import numpy as np
ALIGN = Path(sys.argv[1])
eps_all, zero_all = [], []
for run in sorted((ALIGN / "rcbfqp").glob("run_*")):
    d = np.load(run / "diagnostics.npz"); f = list(d["fields"]); c = lambda n: d["data"][:, f.index(n)]
    for k in range(2):
        near = np.hypot(c("xhat_xc") - c(f"obs{k}_x"), c("xhat_yc") - c(f"obs{k}_y")) < 2.0
        phi = c(f"obs{k}_phi")
        ok = near & np.isfinite(phi)
        eps_all.append(c("eps_px")[ok]); zero_all.append(phi[ok] < 0.01)
eps, zero = np.concatenate(eps_all), np.concatenate(zero_all)
for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.21)]:
    m = (eps >= lo) & (eps < hi)
    if m.any():
        print(f"eps_px in [{lo:.1f}, {hi:.1f}): {m.sum():4d} steps, minimum-gain fallback on {zero[m].mean():.0%}")
