"""Post-run checks for the Go2 hardware rerun (RA-L 26-5080 revision).

Usage (no ROS needed; reads rosbag2 .db3 files directly):

    python hw_checks.py phi-frame   <sweep_dir> <phi_ckpt.pkl> [--phinet-dir DIR]
    python hw_checks.py error-bound <aligned_dir> [--rate 0.13 --base-x 0.05 --cap-x 0.7 --bound-y 0.15]
    python hw_checks.py summary     <sweep_dir> [<sweep_dir> ...]
    python hw_checks.py pilot       <sweep_dir> [<sweep_dir> ...] --ckpt <nmr_ckpt.pkl> [--phinet-dir DIR]

phi-frame    Replays each run's logged state estimate through the PhiNet in the absolute
             frame and in each obstacle's frame, and reports which one reproduces the logged
             phi. After the fix, the obstacle-relative column must match (median error ~0).
             Uses the logged eps vector if the diagnostics contain one (fields eps_px, eps_vx,
             eps_py, eps_vy); otherwise reconstructs it from the distance schedule options.
error-bound  For aligned ground-truth / estimate trajectories (<run>/gt.npy, <run>/est.npy,
             columns x, y, yaw), reports how often |xhat - x| leaves the configured bound.
             The bound is only conservative if this is 0 for every run.
summary      Collisions, trial outcomes (reached / timeout / out_of_bounds / aborted), minimum
             clearance, and duration per sweep, from runs.csv/report.json.
pilot        PASS/FAIL checks on a short pilot sweep: bags decode; the new diagnostics fields
             (eps_*, filter_ms, obs{k}_phi) exist; the logged bound follows the schedule and stays
             within the training envelope; the filter keeps up at 50 Hz; NMR-CBF phi is evaluated
             per obstacle; runs.csv and sweep_meta.json record what is needed to match Vicon files
             and reproduce the run. Exits with code 1 if anything fails.
"""

import argparse
import glob
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

DIAG_TOPIC = "/safe_ctrl_node/diagnostics"


def read_float64_multiarray(bag_dir, topic=DIAG_TOPIC):
    """Rows of a std_msgs/Float64MultiArray topic from a rosbag2 sqlite bag (no layout dims)."""
    db = glob.glob(str(Path(bag_dir) / "*.db3"))[0]
    con = sqlite3.connect(db)
    tid = con.execute("select id from topics where name=?", (topic,)).fetchone()[0]
    rows = []
    for (blob,) in con.execute(
            "select data from messages where topic_id=? order by timestamp", (tid,)):
        b = blob[4:]  # CDR encapsulation header
        assert struct.unpack_from("<I", b, 0)[0] == 0, "layout dims not supported"
        n = struct.unpack_from("<I", b, 8)[0]
        rows.append(struct.unpack_from(f"<{n}d", b, 16))
    return np.array(rows)


def phi_frame(args):
    sys.path.insert(0, args.phinet_dir)
    from diagnostics import DIAG_FIELDS, column  # the onboard package's field layout
    from phi_net import load_phi_net

    net = load_phi_net(args.ckpt)
    cfg = dict(base=np.array(args.base_eps), rate=np.array(args.rate_eps), cap=np.array(args.cap_eps))
    for bag in sorted(glob.glob(str(Path(args.sweep_dir) / "run_*" / "bag"))):
        A = read_float64_multiarray(bag)
        c = lambda name: A[:, column(name)]
        xh = np.stack([c("xhat_xc"), c("xhat_vxc"), c("xhat_yc"), c("xhat_vyc")], 1)
        if "eps_px" in DIAG_FIELDS:
            eps = np.stack([c("eps_px"), c("eps_vx"), c("eps_py"), c("eps_vy")], 1)
        else:
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xh[:, 0]), np.diff(xh[:, 2])))])
            eps = np.minimum(cfg["base"] + cfg["rate"] * d[:, None], cfg["cap"])
        n_obs = int(np.nanmax(c("n_obs")))
        obs = np.stack([np.stack([c(f"obs{k}_x"), c(f"obs{k}_y")], 1) for k in range(n_obs)], 1)
        phi_log = c("phi")
        idx = np.where(np.isfinite(phi_log))[0][::10]
        ab = np.array([net(xh[i], eps[i]) for i in idx])
        rel = np.array([[net(xh[i] - np.array([o[0], 0, o[1], 0]), eps[i]) for o in obs[i]] for i in idx])
        err_rel = np.min(np.abs(rel - phi_log[idx, None]), 1)
        print(f"{Path(bag).parent.name}: median |phi_abs - logged| {np.median(np.abs(ab - phi_log[idx])):.3f}"
              f"   median |phi_rel - logged| {np.median(err_rel):.3f}")


PILOT_RUN_COLUMNS = ["run", "cbf_method", "start_x", "start_y", "start_yaw_deg", "start_wall_time", "outcome"]
PILOT_META_KEYS = ["git_commit", "ckpt_md5", "run_config"]


def pilot(args):
    """PASS/FAIL checks that a short pilot sweep logged everything the paper needs. Exit code 1 on any FAIL."""
    import csv
    sys.path.insert(0, args.phinet_dir)
    from diagnostics import DIAG_FIELDS, column
    from phi_net import load_phi_net

    net = load_phi_net(args.ckpt)
    base, rate, cap = map(np.array, (args.base_eps, args.rate_eps, args.cap_eps))
    failures = []

    def check(ok, what, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(what)

    for sweep in args.sweep_dirs:
        sweep = Path(sweep)
        print(f"== {sweep.name}")
        meta = sweep / "sweep_meta.json"
        check(meta.exists() and all(k in json.load(open(meta)) for k in PILOT_META_KEYS),
              "sweep_meta.json records the git commit, checkpoint md5 and config")
        rows = list(csv.DictReader(open(sweep / "runs.csv"))) if (sweep / "runs.csv").exists() else []
        check(bool(rows), "runs.csv exists and has rows")
        missing = [k for k in PILOT_RUN_COLUMNS if rows and k not in rows[0]]
        check(rows and not missing, "runs.csv has run ID, method, start pose and wall-clock time", f"missing {missing}" if missing else "")
        for name in ["eps_px", "eps_vx", "eps_py", "eps_vy", "filter_ms", "obs0_phi", "obs1_phi"]:
            check(name in DIAG_FIELDS, f"diagnostics field {name} exists")
        if failures and any(f.startswith("diagnostics field") for f in failures):
            continue
        for row, bag in zip(rows, sorted(glob.glob(str(sweep / "run_*" / "bag")))):
            print(f" - {Path(bag).parent.name} ({row.get('cbf_method')})")
            try:
                A = read_float64_multiarray(bag)
            except Exception as e:  # noqa: BLE001
                check(False, "bag decodes", repr(e))
                continue
            check(len(A) > 100, "bag has diagnostics rows", f"{len(A)} rows")
            c = lambda name: A[:, column(name)]
            t = c("t")
            hz = (len(t) - 1) / (t[-1] - t[0])
            check(45 <= hz <= 55, "diagnostics rate is about 50 Hz", f"{hz:.1f} Hz")
            ms = c("filter_ms")
            check(np.nanpercentile(ms, 99) < 20, "filter keeps up at 50 Hz (99th percentile < 20 ms)",
                  f"median {np.nanmedian(ms):.2f} ms, p99 {np.nanpercentile(ms, 99):.2f} ms")
            xh = np.stack([c("xhat_xc"), c("xhat_vxc"), c("xhat_yc"), c("xhat_vyc")], 1)
            eps = np.stack([c("eps_px"), c("eps_vx"), c("eps_py"), c("eps_vy")], 1)
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xh[:, 0]), np.diff(xh[:, 2])))])
            sched = np.minimum(base + rate * d[:, None], cap)
            err = np.nanmax(np.abs(eps - sched))
            check(err < 0.03, "logged error bound follows the schedule", f"max deviation {err:.3f}")
            check(np.all(eps <= np.array(args.ebar) + 1e-6), "error bound stays within the training envelope")
            if row.get("cbf_method") == "nmrcbf":
                n_obs = int(np.nanmax(c("n_obs")))
                idx = np.where(np.isfinite(c("obs0_phi")))[0][::10]
                rel_err, n_diff = [], 0
                for k in range(n_obs):
                    o = np.stack([c(f"obs{k}_x"), c(f"obs{k}_y")], 1)
                    logged = c(f"obs{k}_phi")
                    ok = idx[np.isfinite(logged[idx])]
                    pred = np.array([net(xh[i] - np.array([o[i, 0], 0, o[i, 1], 0]), eps[i]) for i in ok])
                    rel_err.append(np.median(np.abs(pred - logged[ok])))
                n_diff = np.median(np.abs(c("obs0_phi")[idx] - c("obs1_phi")[idx])) if n_obs > 1 else 1.0
                check(max(rel_err) < 0.02, "NMR-CBF phi is evaluated in each obstacle's frame",
                      "median |phi_rel - logged| " + ", ".join(f"{e:.3f}" for e in rel_err))
                check(n_diff > 0.01, "NMR-CBF phi differs between obstacles", f"median difference {n_diff:.3f}")
            if (Path(bag).parent / "report.json").exists():
                rep = json.load(open(Path(bag).parent / "report.json"))
                check(rep.get("outcome") in ("reached", "timeout", "out_of_bounds", "aborted"),
                      "report.json records the trial outcome", f"outcome {rep.get('outcome')!r}")
                print(f"    info: collided {rep.get('any_collision')}, worst clearance {rep.get('worst_clearance', float('nan')):.2f} m, "
                      f"final goal distance {rep.get('final_goal_dist', float('nan')):.2f} m, duration {rep.get('duration_s', float('nan')):.1f} s")
            else:
                check(False, "report.json exists")
    print("\nPILOT " + ("PASSED" if not failures else f"FAILED ({len(failures)} checks)"))
    sys.exit(1 if failures else 0)


def error_bound(args):
    for run in sorted(glob.glob(str(Path(args.aligned_dir) / "*"))):
        if not Path(run, "gt.npy").exists():
            continue
        gt, est = np.load(Path(run, "gt.npy")), np.load(Path(run, "est.npy"))
        e = np.abs(est[:, :2] - gt[:, :2])
        d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(est[:, :2], axis=0).T))])
        bx = np.minimum(args.base_x + args.rate * d, args.cap_x)
        out = (e[:, 0] > bx) | (e[:, 1] > args.bound_y)
        d_true = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(gt[:, :2], axis=0).T))])
        first = f"first outside at {d_true[np.argmax(out)]:.1f} m (true distance)" if out.any() else "never outside"
        print(f"{Path(run).name}: path {d_true[-1]:.1f} m, max|ex| {e[:, 0].max():.2f}, max|ey| {e[:, 1].max():.2f}, "
              f"steps outside bound {out.mean():.0%}, {first}, worst excess x {np.max(e[:, 0] - bx):+.2f} y {np.max(e[:, 1] - args.bound_y):+.2f}")


def summary(args):
    import csv
    for sweep in args.sweep_dirs:
        rows = list(csv.DictReader(open(Path(sweep, "runs.csv"))))
        reps = [json.load(open(p)) for p in sorted(glob.glob(str(Path(sweep, "run_*", "report.json"))))]
        coll = sum(r["physical_collision"] == "True" for r in rows)
        clr = np.array([float(r["min_clearance_m"]) for r in rows])
        dur = np.array([r["duration_s"] for r in reps])
        outcomes = ""
        if "outcome" in rows[0]:
            from collections import Counter
            outcomes = ", outcomes " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["outcome"] for r in rows).items()))
        print(f"{Path(sweep).name}: method {rows[0]['cbf_method']}, n {len(rows)}, physical collisions {coll}{outcomes}, "
              f"min clearance min {clr.min():.2f} / median {np.median(clr):.2f} m, duration median {np.median(dur):.1f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("phi-frame")
    a.add_argument("sweep_dir")
    a.add_argument("ckpt")
    a.add_argument("--phinet-dir", default="deploy/go2/nick_safe_ctrl/nick_safe_ctrl")
    a.add_argument("--base-eps", type=float, nargs=4, default=[0.05, 0.01, 0.15, 0.01])
    a.add_argument("--rate-eps", type=float, nargs=4, default=[0.13, 0.0, 0.0, 0.0])
    a.add_argument("--cap-eps", type=float, nargs=4, default=[0.7, 0.15, 0.7, 0.15])
    a.set_defaults(fn=phi_frame)
    b = sub.add_parser("error-bound")
    b.add_argument("aligned_dir")
    b.add_argument("--rate", type=float, default=0.13)
    b.add_argument("--base-x", type=float, default=0.05)
    b.add_argument("--cap-x", type=float, default=0.7)
    b.add_argument("--bound-y", type=float, default=0.15)
    b.set_defaults(fn=error_bound)
    pl = sub.add_parser("pilot")
    pl.add_argument("sweep_dirs", nargs="+")
    pl.add_argument("--ckpt", required=True)
    pl.add_argument("--phinet-dir", default="deploy/go2/nick_safe_ctrl/nick_safe_ctrl")
    pl.add_argument("--base-eps", type=float, nargs=4, default=[0.05, 0.25, 0.30, 0.25])
    pl.add_argument("--rate-eps", type=float, nargs=4, default=[0.25, 0.0, 0.0, 0.0])
    pl.add_argument("--cap-eps", type=float, nargs=4, default=[1.2, 0.3, 0.5, 0.3])
    pl.add_argument("--ebar", type=float, nargs=4, default=[1.2, 0.3, 0.5, 0.3])
    pl.set_defaults(fn=pilot)
    s = sub.add_parser("summary")
    s.add_argument("sweep_dirs", nargs="+")
    s.set_defaults(fn=summary)
    args = p.parse_args()
    args.fn(args)
