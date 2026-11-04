"""Tightening (phi) applied while approaching each obstacle, and Vicon clearance per run, for CBF / R-CBF-QP / NMR-CBF.
Usage: python -I robustness_terms.py <phase-c alignment dir>"""
import sys
from pathlib import Path
import numpy as np

ALIGN = Path(sys.argv[1])
OBS = np.array([[-0.964, 0.331], [1.922, -0.467]])


def load(method, run):
    d = np.load(ALIGN / method / run / "diagnostics.npz")
    f = list(d["fields"])
    return lambda name: d["data"][:, f.index(name)]


print("phi = tightening added to the CBF constraint; 'near' = estimated control point within 1.5 m of the obstacle centre")
for method in ["rcbfqp", "nmrcbf"]:
    print(f"== {method}")
    for run in sorted((ALIGN / method).glob("run_*")):
        c = load(method, run.name)
        cp = np.stack([c("xhat_xc"), c("xhat_yc")], 1)
        parts = []
        for k in range(2):
            ox, oy = c(f"obs{k}_x"), c(f"obs{k}_y")
            near = np.hypot(cp[:, 0] - ox, cp[:, 1] - oy) < 1.5
            phi = c(f"obs{k}_phi")[near]
            phi = phi[np.isfinite(phi)]
            parts.append(f"obs{k+1}: median phi {np.median(phi):.2f}, phi<0.05 on {np.mean(phi < 0.05):.0%}" if len(phi) else f"obs{k+1}: -")
        print(f"  {run.name}: " + " | ".join(parts))

print("== min body clearance to each obstacle (Vicon), same start index")
for i in range(1, 11):
    row = []
    for method in ["plain", "rcbfqp", "nmrcbf"]:
        run = next((ALIGN / method).glob(f"run_{i:02d}_*"))
        gt = np.load(run / "gt.npy")[:, :2]
        clr = np.linalg.norm(gt[:, None] - OBS[None], axis=2).min(0) - 0.43
        row.append(f"{method:6s} {clr[0]:+.2f}/{clr[1]:+.2f}")
    print(f"  run {i:2d}: " + "   ".join(row))
