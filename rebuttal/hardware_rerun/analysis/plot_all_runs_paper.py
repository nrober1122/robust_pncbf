"""Single-column paper figure: all ten hardware runs per method, Vicon ground truth (solid) and onboard estimate incl.
injected bias (dotted), one static color per method, or with `speed` the ground truth colored by speed (as in Fig. 3) and
the estimate in gray. Contacts are marked at the closest approach; stalls (timeouts) at the last ground-truth sample.

    python -I plot_all_runs_paper.py <phase-c alignment dir> <sweeps dir> <out.png> [speed]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ALIGN, SWEEPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SPEED = "speed" in sys.argv[4:]
CMAP, NORM = plt.cm.plasma, plt.Normalize(0.0, 1.1)
OBS = np.array([[-0.964, 0.331], [1.922, -0.467]])
R_REAL, R_KEEPOUT = 0.43, 0.43 + 0.16
GOAL = (3.5, 0.0)
BIAS = np.array([0.0, -0.15])
METHODS = [("plain", "CBF", "tab:orange"), ("rcbfqp", "R-CBF-QP", "tab:green"),
           ("duality", "Duality CBF", "tab:purple"), ("nmrcbf", "NMR-CBF", "tab:blue")]
# Duality run 4 has no Vicon recording; its repeat from the same start (same outcome) is used in its place.
SUBSTITUTES = {("duality", 4): ("supplementary_duality_run04_B_attempt02", "supplementary_duality")}

plt.style.use("default")
rc = {"font.family": "serif", "text.usetex": True, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "axes.facecolor": "white", "figure.facecolor": "white"}
with matplotlib.rc_context(rc):
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 1.95 if SPEED else 2.1), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.02, hspace=0.04)
    for ax, (method, label, c) in zip(axes.flat, METHODS):
        rows = {int(r["run"]): r for r in csv.DictReader(open(SWEEPS / f"sweep_{method}" / "runs.csv"))}
        for (m, i), (sweep, _) in SUBSTITUTES.items():
            if m == method:
                rows[i] = next(csv.DictReader(open(SWEEPS / sweep / "runs.csv")))
        for o in OBS:
            ax.add_patch(Circle(o, R_REAL, color="0.6", lw=0, zorder=1))
            ax.add_patch(Circle(o, R_KEEPOUT, fill=False, ls="--", lw=0.6, color="0.3", zorder=1))
        for i in range(1, 11):
            align_dir = ALIGN / SUBSTITUTES.get((method, i), (None, method))[1]
            run_dir = next(align_dir.glob(f"run_{i:02d}_*"))
            gt = np.load(run_dir / "gt.npy")[:, :2]
            est = np.load(run_dir / "est.npy")[:, :2] + BIAS
            if SPEED:
                v = np.linalg.norm(np.gradient(gt, np.load(run_dir / "time.npy"), axis=0), axis=1)
                v = np.convolve(v, np.ones(9) / 9, "same")  # 0.18 s moving average against Vicon jitter
                seg = np.stack([gt[:-1], gt[1:]], axis=1)
                ax.add_collection(LineCollection(seg, colors=CMAP(NORM(v[:-1])), lw=0.8, zorder=3))
                ax.plot(est[:, 0], est[:, 1], color="0.35", lw=0.6, ls=(0, (1, 1)), alpha=0.6, zorder=2)
            else:
                ax.plot(gt[:, 0], gt[:, 1], color=c, lw=0.8, zorder=3)
                ax.plot(est[:, 0], est[:, 1], color=c, lw=0.6, ls=(0, (1, 1)), alpha=0.6, zorder=2)
            r = rows[i]
            if r["physical_collision"] == "True":
                k = int(r["physical_obstacles"].strip("[]"))
                j = np.argmin(np.linalg.norm(gt - OBS[k], axis=1))
                ax.plot(*gt[j], "x", color="k" if SPEED else "red", ms=6, mew=1.6, zorder=6)
            if r["outcome"] == "timeout":
                ax.plot(*gt[-1], "o", mfc="none", mec="k", ms=5, mew=1.2, zorder=6)
        ax.plot(*GOAL, marker="*", ms=8, mfc="gold", mec="k", mew=0.5, zorder=5)
        ax.set_title(rf"\textbf{{{label}}}", pad=5)
        ax.set_xlim(-4.3, 5.7)
        ax.set_ylim(-2.0, 1.5)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-2, -1, 0, 1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(length=2, pad=1.5)
    for ax in axes[1]:
        ax.set_xlabel(r"$p_x$ (m)", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_y$ (m)", labelpad=1)
    handles = [Line2D([], [], color="k", lw=0.9, label="true state"),
               Line2D([], [], color="k", lw=0.9, ls=(0, (1, 1)), label="state estimate"),
               Line2D([], [], marker="*", ls="", ms=7, mfc="gold", mec="k", mew=0.5, label="goal"),
               Patch(color="0.6", label="obstacle"),
               Line2D([], [], marker="x", ls="", color="k" if SPEED else "red", ms=5, mew=1.4, label="collision"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=5, mew=1.1, label="stall")]
    if SPEED:
        handles[1] = Line2D([], [], color="0.35", lw=0.9, ls=(0, (1, 1)), label="state estimate")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=axes, pad=0.01, aspect=25)
        cb.set_label("speed (m/s)", labelpad=2)
        cb.ax.tick_params(length=2, pad=1.5)
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.2)
    fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT)
