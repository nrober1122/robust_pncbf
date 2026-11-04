"""Fig. 6: ground-truth and estimated Go2 trajectories for the leftmost and rightmost start (runs 1 and 10), CBF vs NMR-CBF.

    python -I plot_selected_runs.py <phase-c alignment dir> <out.png>

The estimate is the onboard odometry with the injected p_y bias added back, i.e. what the filter used.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ALIGN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OBS = [(-0.964, 0.331), (1.922, -0.467)]
R_REAL, R_KEEPOUT = 0.43, 0.43 + 0.16
GOAL = (3.5, 0.0)
BIAS = np.array([0.0, -0.15])
PANELS = [("plain", "run_01_A", "CBF run 1"), ("plain", "run_10_B", "CBF run 10"),
          ("nmrcbf", "run_01_A", "NMR-CBF run 1"), ("nmrcbf", "run_10_B", "NMR-CBF run 10")]
COLORS = {"plain": "tab:orange", "nmrcbf": "tab:blue"}

plt.style.use("default")
with matplotlib.rc_context({"font.family": "serif", "text.usetex": True, "font.size": 10, "axes.facecolor": "white", "figure.facecolor": "white"}):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.9), sharex=True, sharey=True, layout="constrained")
    for ax, (method, run, title) in zip(axes.flat, PANELS):
        gt = np.load(ALIGN / method / run / "gt.npy")[:, :2]
        est = np.load(ALIGN / method / run / "est.npy")[:, :2] + BIAS
        c = COLORS[method]
        for o in OBS:
            ax.add_patch(Circle(o, R_REAL, color="0.6", zorder=1))
            ax.add_patch(Circle(o, R_KEEPOUT, fill=False, ls="--", lw=0.9, color="0.3", zorder=1))
        ax.plot(gt[:, 0], gt[:, 1], color=c, lw=1.6, zorder=3)
        ax.plot(est[:, 0], est[:, 1], color=c, lw=1.6, ls=":", zorder=3)
        for traj in (gt, est):
            ax.plot(*traj[0], "o", color=c, ms=3.5, zorder=4)
            ax.plot(*traj[-1], "s", color=c, ms=3.5, zorder=4)
        ax.plot(*GOAL, marker="*", ms=12, mfc="gold", mec="k", mew=0.8, zorder=5)
        ax.set_title(rf"\textbf{{{title}}}")
        ax.set_xlim(-4.4, 5.2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"$p_x$ (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_y$ (m)")
    handles = [Patch(color="0.6", label="obstacle"),
               Line2D([], [], ls="--", color="0.3", label="keep-out (+body margin)"),
               Line2D([], [], color="k", label="true state"),
               Line2D([], [], color="k", ls=":", label="state estimate"),
               Line2D([], [], marker="*", ls="", ms=10, mfc="gold", mec="k", label="goal")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved", OUT)
