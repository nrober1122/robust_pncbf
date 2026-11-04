"""All hardware runs, one panel per method: Vicon ground truth (solid) and onboard estimate incl. injected bias (dotted).
Contacts reported by the operator are marked with a red x at the closest approach; stalls (timeouts) end with a black x.

    python -I plot_all_runs.py <phase-c alignment dir> <sweeps dir> <out.png>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ALIGN, SWEEPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OBS = np.array([[-0.964, 0.331], [1.922, -0.467]])
R_REAL, R_KEEPOUT = 0.43, 0.59
GOAL = (3.5, 0.0)
BIAS = np.array([0.0, -0.15])
METHODS = [("plain", "CBF"), ("rcbfqp", "R-CBF-QP"), ("duality", "Duality CBF"), ("nmrcbf", "NMR-CBF")]
# Duality run 4 has no Vicon recording; its repeat from the same start (same outcome) is used in its place.
SUBSTITUTES = {("duality", 4): ("supplementary_duality_run04_B_attempt02", "supplementary_duality")}

plt.style.use("default")
with matplotlib.rc_context({"font.family": "serif", "text.usetex": True, "font.size": 10}):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, layout="constrained")
    cmap = plt.cm.viridis
    for ax, (method, label) in zip(axes.flat, METHODS):
        rows = {int(r["run"]): r for r in csv.DictReader(open(SWEEPS / f"sweep_{method}" / "runs.csv"))}
        for (m, i), (sweep, align) in SUBSTITUTES.items():
            if m == method:
                rows[i] = next(csv.DictReader(open(SWEEPS / sweep / "runs.csv")))
        for o in OBS:
            ax.add_patch(Circle(o, R_REAL, color="0.6", zorder=1))
            ax.add_patch(Circle(o, R_KEEPOUT, fill=False, ls="--", lw=0.8, color="0.3", zorder=1))
        n_plotted = 0
        for i in range(1, 11):
            align_dir = ALIGN / SUBSTITUTES.get((method, i), (None, method))[1]
            run_dirs = list(align_dir.glob(f"run_{i:02d}_*"))
            if not run_dirs:
                continue
            gt = np.load(run_dirs[0] / "gt.npy")[:, :2]
            est = np.load(run_dirs[0] / "est.npy")[:, :2] + BIAS
            c = cmap((i - 1) / 9)
            ax.plot(gt[:, 0], gt[:, 1], color=c, lw=1.3, zorder=3)
            ax.plot(est[:, 0], est[:, 1], color=c, lw=0.9, ls=":", alpha=0.8, zorder=2)
            r = rows[i]
            if r["physical_collision"] == "True":
                k = int(r["physical_obstacles"].strip("[]"))
                j = np.argmin(np.linalg.norm(gt - OBS[k], axis=1))
                ax.plot(*gt[j], "x", color="red", ms=9, mew=2.2, zorder=6)
            if r["outcome"] == "timeout":
                ax.plot(*gt[-1], "x", color="k", ms=8, mew=2, zorder=6)
            n_plotted += 1
        n_contact = sum(r["physical_collision"] == "True" for r in rows.values())
        n_timeout = sum(r["outcome"] == "timeout" for r in rows.values())
        ax.plot(*GOAL, marker="*", ms=13, mfc="gold", mec="k", mew=0.8, zorder=5)
        ax.set_title(rf"\textbf{{{label}}}: {n_contact} contacts, {n_timeout} timeouts ({n_plotted} runs with Vicon)")
        ax.set_xlim(-4.4, 6.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"$p_x$ (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_y$ (m)")
    handles = [Line2D([], [], color="k", label="true state (Vicon)"),
               Line2D([], [], color="k", ls=":", label="state estimate (incl. bias)"),
               Line2D([], [], marker="x", ls="", color="red", mew=2, label="contact"),
               Line2D([], [], marker="x", ls="", color="k", mew=2, label="stall (timeout)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 10))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="run (start position, left to right)")
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.03)
print("saved", OUT)
