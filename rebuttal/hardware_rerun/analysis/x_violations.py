"""Where (true distance from start) the along-track bound is first violated before obstacle 2, and by how much."""
import sys
from pathlib import Path
import numpy as np
ALIGN = Path(sys.argv[1]); OBS2_X = 1.922
for method in ["nmrcbf", "plain", "rcbfqp", "duality"]:
    out = []
    for run in sorted((ALIGN / method).glob("run_*")):
        gt, est = np.load(run / "gt.npy"), np.load(run / "est.npy")
        ex = np.abs(est[:, 0] - gt[:, 0])
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(est[:, :2], axis=0).T))])
        d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(gt[:, :2], axis=0).T))])
        bx = np.minimum(0.05 + 0.25 * s, 1.2)
        i2 = int(np.argmin(np.abs(gt[:, 0] - OBS2_X)))
        viol = np.where(ex[:i2 + 1] > bx[:i2 + 1])[0]
        if len(viol):
            out.append(f"{run.name[4:6]}: {len(viol)} steps, d {d[viol[0]]:.2f}-{d[viol[-1]]:.2f} m, max excess {np.max(ex[viol] - bx[viol]):.3f} m")
    print(method, "|", "; ".join(out) if out else "none")
