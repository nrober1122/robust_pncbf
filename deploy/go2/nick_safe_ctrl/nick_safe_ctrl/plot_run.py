"""Plot a safe_ctrl_node diagnostics bag.

Reads ``/safe_ctrl_node/diagnostics`` (``std_msgs/Float64MultiArray``) from a
rosbag2 recording and writes a PNG with:

  - XY body trajectory, with every obstacle (real circle + inflated keep-out)
    and the goal (geometry is read from the bag, no CLI args needed)
  - h_true (max barrier over all obstacles) / phi vs time
  - vx / vyaw commands vs time

Record a run with::

    ros2 bag record -o ~/bags/myrun /safe_ctrl_node/diagnostics

Then plot::

    python3 plot_run.py ~/bags/myrun [--out myrun.png] [--show]

Overlay several runs (e.g. plain vs pshrcbf) on one set of axes::

    python3 plot_run.py ~/bags/plain ~/bags/pshr --labels plain pshr --out ab.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from .diagnostics import DIAG_FIELDS, column
except ImportError:
    from diagnostics import DIAG_FIELDS, column


def _read_via_rosbag2(bag_dir, topic):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import Float64MultiArray

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    rows = []
    while reader.has_next():
        tname, data, _t = reader.read_next()
        if tname == topic:
            rows.append(list(deserialize_message(data, Float64MultiArray).data))
    return rows


def _read_via_sqlite(bag_dir, topic):
    """Fallback reader: parse the rosbag2 sqlite3 .db3 directly (no rosbag2_py)."""
    import glob
    import sqlite3
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import Float64MultiArray

    db3s = sorted(glob.glob(str(Path(bag_dir) / "*.db3")))
    if not db3s:
        raise RuntimeError(f"no .db3 file found in {bag_dir}")
    rows = []
    for db3 in db3s:
        con = sqlite3.connect(db3)
        try:
            cur = con.cursor()
            cur.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            hit = cur.fetchone()
            if hit is None:
                continue
            cur.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                        (hit[0],))
            for (blob,) in cur.fetchall():
                rows.append(list(deserialize_message(bytes(blob), Float64MultiArray).data))
        finally:
            con.close()
    return rows


def read_diag_bag(bag_dir: str, topic: str = "/safe_ctrl_node/diagnostics") -> np.ndarray:
    """Return an (N, len(DIAG_FIELDS)) array of diagnostics rows from a bag.

    Uses rosbag2_py if available, else reads the sqlite3 .db3 directly.
    """
    try:
        rows = _read_via_rosbag2(bag_dir, topic)
    except ImportError:
        rows = _read_via_sqlite(bag_dir, topic)
    if not rows:
        raise RuntimeError(f"no '{topic}' messages found in {bag_dir}")
    arr = np.array(rows)
    if arr.shape[1] != len(DIAG_FIELDS):
        raise RuntimeError(
            f"bag has {arr.shape[1]} fields, expected {len(DIAG_FIELDS)} "
            f"-- DIAG_FIELDS mismatch between recording and this script")
    return arr


def _obstacles_of(run, c):
    """Return [(ox, oy, r_cbf, real_r)] for the populated obstacle slots of a run.

    Geometry is run-constant; read it from the first row. A slot counts if its
    index is < n_obs and its centre is finite.
    """
    try:
        from .diagnostics import MAX_OBS
    except ImportError:
        from diagnostics import MAX_OBS
    n_obs = int(run[0, c["n_obs"]])
    obs = []
    for k in range(min(n_obs, MAX_OBS)):
        ox, oy = run[0, c[f"obs{k}_x"]], run[0, c[f"obs{k}_y"]]
        if not (np.isfinite(ox) and np.isfinite(oy)):
            continue
        obs.append((ox, oy, run[0, c[f"obs{k}_rcbf"]], run[0, c[f"obs{k}_realr"]]))
    return obs


def plot_runs(bags, labels, out_path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    runs = [read_diag_bag(b) for b in bags]
    c = {name: column(name) for name in DIAG_FIELDS}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    ax_xy, ax_barrier, ax_cmd = axes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    # obstacle + goal geometry from the first run (constant within a run)
    r0 = runs[0]
    obstacles = _obstacles_of(r0, c)
    gx, gy = r0[0, c["goal_x"]], r0[0, c["goal_y"]]

    for j, (ox, oy, r_cbf, real_r) in enumerate(obstacles):
        ax_xy.add_patch(Circle((ox, oy), max(real_r, 1e-3), color="0.4", alpha=0.8,
                               zorder=3, label="obstacle" if j == 0 else None))
        ax_xy.add_patch(Circle((ox, oy), max(r_cbf, 1e-3), fill=False, ls="--",
                               color="0.5", zorder=3,
                               label="CBF keep-out" if j == 0 else None))
    ax_xy.scatter([gx], [gy], marker="*", s=180, color="goldenrod",
                  zorder=5, label="goal")

    for run, lab, col in zip(runs, labels, colors):
        t = run[:, c["t"]] - run[0, c["t"]]
        ax_xy.plot(run[:, c["x"]], run[:, c["y"]], color=col, lw=2, label=lab, zorder=4)
        ax_xy.scatter(run[0, c["x"]], run[0, c["y"]], color=col, marker="o", s=40, zorder=5)
        ax_barrier.plot(t, run[:, c["h_true"]], color=col, lw=1.8, label=f"{lab} h_true")
        ax_barrier.plot(t, run[:, c["phi"]], color=col, lw=1.2, ls=":", label=f"{lab} phi")
        ax_cmd.plot(t, run[:, c["vx_cmd"]], color=col, lw=1.8, label=f"{lab} vx")
        ax_cmd.plot(t, run[:, c["vyaw_cmd"]], color=col, lw=1.2, ls="--", label=f"{lab} vyaw")

    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m, task frame)"); ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("body trajectory"); ax_xy.legend(fontsize=8); ax_xy.grid(alpha=0.3)

    ax_barrier.axhline(0, color="k", lw=0.6)
    ax_barrier.set_xlabel("time (s)"); ax_barrier.set_ylabel("value")
    ax_barrier.set_title("h_true (max over obstacles; <0 = body safe)  &  phi")
    ax_barrier.legend(fontsize=8); ax_barrier.grid(alpha=0.3)

    ax_cmd.set_xlabel("time (s)"); ax_cmd.set_ylabel("command")
    ax_cmd.set_title("commanded vx (m/s) & vyaw (rad/s)")
    ax_cmd.legend(fontsize=8); ax_cmd.grid(alpha=0.3)

    # report the safety-relevant number per run: min body->surface clearance,
    # computed per obstacle from the trajectory, then the worst (min) over all.
    for run, lab in zip(runs, labels):
        obs = _obstacles_of(run, c)
        bx, by = run[:, c["x"]], run[:, c["y"]]
        worst = np.inf
        for k, (ox, oy, _r_cbf, real_r) in enumerate(obs):
            d_center = np.hypot(bx - ox, by - oy)
            clr = float(np.min(d_center) - real_r)
            worst = min(worst, clr)
            print(f"  {lab:<12} obs{k} @({ox:+.2f},{oy:+.2f}) "
                  f"min body->surface = {clr:+.3f} m (real R={real_r:.2f})")
        if np.isfinite(worst):
            print(f"  {lab:<12} -> worst-case clearance over all obstacles = {worst:+.3f} m")

    fig.suptitle("safe_ctrl_node run" + (" comparison" if len(runs) > 1 else ""))
    if show:
        plt.show()
    else:
        fig.savefig(out_path, dpi=120)
        print(f"saved {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+", help="one or more rosbag2 directories")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels (defaults to bag dir names)")
    ap.add_argument("--out", default="run_plot.png", help="output PNG path")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    args = ap.parse_args(argv)

    labels = args.labels or [Path(b).name for b in args.bags]
    if len(labels) != len(args.bags):
        ap.error("number of --labels must match number of bags")
    plot_runs(args.bags, labels, args.out, show=args.show)


if __name__ == "__main__":
    main()
