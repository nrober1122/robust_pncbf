"""Signed estimation error (estimate incl. injected py bias minus Vicon) at each obstacle pass, per method/run.
Usage: python -I error_at_obstacles.py <phase-c alignment dir>"""
import json, sys
from pathlib import Path
import numpy as np

ALIGN = Path(sys.argv[1])
OBS = np.array([[-0.964, 0.331], [1.922, -0.467]])
BIAS_Y = -0.15
BASE_X, RATE_X, CAP_X, BOUND_Y = 0.05, 0.25, 1.2, 0.30

for method in ["nmrcbf", "plain", "rcbfqp", "duality"]:
    print(f"== {method}")
    for run in sorted((ALIGN / method).glob("run_*")):
        gt, est = np.load(run / "gt.npy"), np.load(run / "est.npy")
        ex = est[:, 0] - gt[:, 0]
        ey = (est[:, 1] + BIAS_Y) - gt[:, 1]
        eyaw = np.rad2deg(np.angle(np.exp(1j * (est[:, 2] - gt[:, 2]))))
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(est[:, :2], axis=0).T))])
        bx = np.minimum(BASE_X + RATE_X * s, CAP_X)
        clr = np.linalg.norm(gt[:, None, :2] - OBS[None], axis=2) - 0.43
        row = []
        for k in range(2):
            i = int(np.argmin(np.abs(gt[:, 0] - OBS[k, 0])))  # passing the obstacle along x
            row.append(f"obs{k+1}: ex {ex[i]:+.2f} (bound {bx[i]:.2f}) ey {ey[i]:+.2f} yaw {eyaw[i]:+.0f}°")
        i2 = int(np.argmin(np.abs(gt[:, 0] - OBS[1, 0])))
        pre = slice(0, i2 + 1)
        vx, vy = (np.abs(ex[pre]) > bx[pre]).any(), (np.abs(ey[pre]) > BOUND_Y).any()
        print(f"  {run.name}: " + " | ".join(row) + f" | before obs2: x out {vx}, y out {vy}, max|ey| {np.abs(ey[pre]).max():.2f}"
              f" | min body clearance {clr.min():+.2f} m")
