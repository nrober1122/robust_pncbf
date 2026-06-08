"""Analyze a safe_ctrl_node diagnostics bag for safety outcomes.

Reads ``/safe_ctrl_node/diagnostics`` from a rosbag2 recording and reports, for
the run as a whole and per obstacle:

  - whether the body **collided** with any real obstacle (body centre came
    within the obstacle's real radius), and where / when the closest approach
    was -- the key thing you want flagged after an experiment;
  - the minimum body-centre -> obstacle-surface clearance per obstacle;
  - whether any CBF **keep-out** was entered (``h_true`` went > 0);
  - whether the goal was reached, plus run duration and sample rate.

Collision uses the *real* obstacle radius (``realr`` in the bag), not the
inflated keep-out ``R_cbf``.  By default the body is treated as a point at the
reported pose; pass ``--body-radius R`` to inflate it (clearance is then
``min_dist - real_r - R``) if you want to account for the robot's footprint.

Usage::

    python3 run_analysis.py <bag_dir> [--body-radius R] [--json out.json]

It is also imported by ``scripts/record_experiment.sh`` to print a report right
after a recording stops.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:  # works both as a package module and as a plain script
    from .plot_run import read_diag_bag, _obstacles_of
    from .diagnostics import DIAG_FIELDS, column
except ImportError:
    from plot_run import read_diag_bag, _obstacles_of
    from diagnostics import DIAG_FIELDS, column


def analyze_array(arr: np.ndarray, body_radius: float = 0.0) -> dict:
    """Compute safety outcomes from an (N, len(DIAG_FIELDS)) diagnostics array."""
    c = {name: column(name) for name in DIAG_FIELDS}
    n = int(arr.shape[0])
    t = arr[:, c["t"]] - arr[0, c["t"]]
    duration = float(t[-1]) if n else 0.0
    rate = (n - 1) / duration if duration > 0 else float("nan")

    bx, by = arr[:, c["x"]], arr[:, c["y"]]

    obs_reports = []
    worst_clear = np.inf
    any_collision = False
    for k, (ox, oy, r_cbf, real_r) in enumerate(_obstacles_of(arr, c)):
        d_center = np.hypot(bx - ox, by - oy)
        i_min = int(np.argmin(d_center))
        eff_r = real_r + body_radius
        clearance = float(d_center[i_min] - eff_r)
        inside = d_center < eff_r
        collided = bool(inside.any())
        any_collision |= collided
        obs_reports.append(dict(
            index=k,
            x=float(ox), y=float(oy), real_r=float(real_r), r_cbf=float(r_cbf),
            min_clearance=clearance,
            collided=collided,
            closest_t=float(t[i_min]),
            closest_xy=(float(bx[i_min]), float(by[i_min])),
            frac_time_inside=float(np.mean(inside)),
            first_contact_t=(float(t[int(np.argmax(inside))]) if collided else None),
        ))
        worst_clear = min(worst_clear, clearance)

    h = arr[:, c["h_true"]]
    goal_dist = arr[:, c["goal_dist"]]
    return dict(
        n_samples=n,
        duration_s=duration,
        rate_hz=float(rate),
        n_obs=len(obs_reports),
        obstacles=obs_reports,
        any_collision=any_collision,
        worst_clearance=(float(worst_clear) if np.isfinite(worst_clear) else None),
        keepout_violated=bool(np.any(h > 0.0)),
        max_h_true=(float(np.max(h)) if n else float("nan")),
        min_goal_dist=(float(np.min(goal_dist)) if n else float("nan")),
        final_goal_dist=(float(goal_dist[-1]) if n else float("nan")),
        body_radius=float(body_radius),
    )


def analyze_bag(bag_dir: str, body_radius: float = 0.0) -> dict:
    """Read a diagnostics bag and analyze it."""
    return analyze_array(read_diag_bag(bag_dir), body_radius=body_radius)


def format_report(a: dict, bag_dir: str = "") -> str:
    title = f" run report: {bag_dir} " if bag_dir else " run report "
    bar = "=" * max(len(title), 60)
    L = [bar, title.center(len(bar)), bar]
    L.append(f"samples: {a['n_samples']}   duration: {a['duration_s']:.1f} s   "
             f"rate: ~{a['rate_hz']:.1f} Hz")
    L.append(f"goal: min dist {a['min_goal_dist']:+.3f} m, final {a['final_goal_dist']:+.3f} m")
    if a["keepout_violated"]:
        L.append(f"keep-out: ENTERED (h_true reached {a['max_h_true']:+.3f} > 0 -- "
                 f"body crossed an inflated R_cbf boundary)")
    else:
        L.append(f"keep-out: max h_true {a['max_h_true']:+.3f} (never entered)")

    br = a["body_radius"]
    L.append("")
    L.append(f"obstacles (body-centre -> surface clearance; body_radius={br:.2f} m):")
    if not a["obstacles"]:
        L.append("  (no obstacles recorded in this bag)")
    for o in a["obstacles"]:
        flag = "*** COLLISION ***" if o["collided"] else "ok"
        line = (f"  obs{o['index']} @({o['x']:+.2f},{o['y']:+.2f}) realR={o['real_r']:.2f}  "
                f"min clearance {o['min_clearance']:+.3f} m  "
                f"at t={o['closest_t']:.1f}s pos({o['closest_xy'][0]:+.2f},"
                f"{o['closest_xy'][1]:+.2f})  {flag}")
        L.append(line)
        if o["collided"]:
            L.append(f"        inside {100 * o['frac_time_inside']:.1f}% of the run, "
                     f"first contact t={o['first_contact_t']:.1f}s")

    L.append("")
    if a["any_collision"]:
        n_hit = sum(1 for o in a["obstacles"] if o["collided"])
        L.append(f"RESULT: COLLISION with {n_hit} obstacle(s)")
    elif a["obstacles"]:
        L.append(f"RESULT: no collisions (worst clearance {a['worst_clearance']:+.3f} m)")
    else:
        L.append("RESULT: no obstacles to check")
    L.append(bar)
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag2 directory containing the diagnostics topic")
    ap.add_argument("--body-radius", type=float, default=0.0,
                    help="inflate the body to this radius for collision (default 0 = point)")
    ap.add_argument("--json", default=None, help="also write the analysis as JSON to this path")
    args = ap.parse_args(argv)

    a = analyze_bag(args.bag, body_radius=args.body_radius)
    print(format_report(a, bag_dir=str(args.bag)))
    if args.json:
        Path(args.json).write_text(json.dumps(a, indent=2))
        print(f"wrote {args.json}")
    # non-zero exit on collision so scripts/CI can branch on it
    return 1 if a["any_collision"] else 0


if __name__ == "__main__":
    sys.exit(main())
