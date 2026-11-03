"""Offline closed-loop simulation of the Go2 course with the onboard filter code. Never touches the robot.

The control point is a 2D double integrator. The estimate drifts like the June odometry: estimated displacement =
(1 - drift) x true displacement, plus a constant py bias. Needs only numpy and the onboard package's
double_integrator_cbf.py and phi_net.py.

    python sim_go2_course.py --pkg-dir <.../nick_safe_ctrl/nick_safe_ctrl> \
        --new-ckpt ckpts/phi_params_finetune_e1.2_cap12_unif.pkl --old-ckpt <.../ckpts/phi_params.pkl>

Collisions are scored like the June report: the body centre (0.35 m behind the control point) inside the real
obstacle radius. The robot stops when its *estimate* is within 0.45 m of the goal. This model is harsher than the
robot (with the June settings it has the CBF colliding 10/10 vs 3/10 on hardware), so compare methods within it only.
"""

import argparse
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--pkg-dir", required=True, help="folder containing double_integrator_cbf.py and phi_net.py")
p.add_argument("--new-ckpt", required=True)
p.add_argument("--old-ckpt", default=None, help="June checkpoint, for reference (optional)")
p.add_argument("--drift", type=float, default=0.195, help="fraction of true distance the odometry under-reports")
p.add_argument("--bias-y", type=float, default=-0.15)
p.add_argument("--body-margin", type=float, default=0.16)
p.add_argument("--new-eps-base", type=float, nargs=4, default=[0.05, 0.25, 0.30, 0.25])
p.add_argument("--new-eps-rate", type=float, nargs=4, default=[0.25, 0.0, 0.0, 0.0])
p.add_argument("--new-eps-cap", type=float, nargs=4, default=[1.2, 0.3, 0.5, 0.3])
args = p.parse_args()

sys.path.insert(0, args.pkg_dir)
import double_integrator_cbf as dic
from phi_net import load_phi_net

# June layout (Vicon frame) and start positions of the June NMR-CBF runs
OBS = [(-0.964, 0.331), (1.922, -0.467)]
R_REAL = 0.43
GOAL = np.array([3.5 + 0.35, 0.0])  # nominal targets goal + d * heading
STARTS = [(-3.854, 0.58), (-3.921, -0.624), (-3.873, 0.432), (-3.793, 0.283), (-3.887, 0.181),
          (-3.912, 0.041), (-3.903, -0.101), (-3.861, -0.235), (-3.862, -0.368), (-3.874, -0.498)]
DT, T_MAX, V_MAX = 0.02, 30.0, 1.0
gains = dict(alpha0=3.0, alpha_qp=2.0, u_max=2.0, qp_penalty=200.0)
params = [dic.CBFParams(obstacle_xy=o, obstacle_radius=R_REAL + args.body_margin, **gains) for o in OBS]


def schedule(base, rate, cap):
    base, rate, cap = map(np.array, (base, rate, cap))
    return lambda d: np.minimum(base + rate * d, cap)


june_eps = schedule([0.05, 0.01, 0.15, 0.01], [0.13, 0, 0, 0], [0.7, 0.15, 0.7, 0.15])
new_eps = schedule(args.new_eps_base, args.new_eps_rate, args.new_eps_cap)
rel = lambda x, q: x - np.array([q.obstacle_xy[0], 0, q.obstacle_xy[1], 0])
new = load_phi_net(args.new_ckpt)
METHODS = {"CBF": lambda x, d: [0.0] * len(params),
           "new network, per obstacle": lambda x, d: [new(rel(x, q), new_eps(d)) for q in params]}
if args.old_ckpt:
    old = load_phi_net(args.old_ckpt)
    METHODS["June network, per obstacle, June bound"] = lambda x, d: [old(rel(x, q), june_eps(d)) for q in params]

for name, phis_fn in METHODS.items():
    res = []
    for s0 in map(np.array, STARTS):
        pos, v, d_est, t = s0.copy(), np.zeros(2), 0.0, 0.0
        min_clr, y_lo, y_hi, reached = np.inf, np.inf, -np.inf, False
        while t < T_MAX:
            p_hat = s0 + (1 - args.drift) * (pos - s0) + np.array([0, args.bias_y])
            v_hat = (1 - args.drift) * v
            xh = np.array([p_hat[0], v_hat[0], p_hat[1], v_hat[1]])
            u, _, _ = dic.cbf_qp_filter_multi(xh, dic.nominal_goto(xh, GOAL, 2.0, 2.0, 2.0), params, phis_fn(xh, d_est))
            v = v + u * DT
            sp = np.linalg.norm(v)
            v = v * min(1, V_MAX / sp) if sp > 0 else v
            pos = pos + v * DT
            d_est += (1 - args.drift) * np.linalg.norm(v) * DT
            t += DT
            body = pos - 0.35 * v / max(np.linalg.norm(v), 1e-6)
            min_clr = min(min_clr, min(np.hypot(*(body - np.array(o))) - R_REAL for o in OBS))
            y_lo, y_hi = min(y_lo, pos[1]), max(y_hi, pos[1])
            if np.linalg.norm(p_hat - GOAL) < 0.45:
                reached = True
                break
        res.append((min_clr < 0, min_clr, t if reached else np.nan, y_lo, y_hi))
    r = np.array(res, dtype=float)
    med_t = np.nanmedian(r[:, 2]) if np.isfinite(r[:, 2]).any() else np.nan
    print(f"{name:40s} collisions {int(r[:, 0].sum())}/{len(STARTS)} | min clearance {r[:, 1].min():+.2f} m "
          f"| reached {int(np.isfinite(r[:, 2]).sum())}/{len(STARTS)}, median {med_t:.1f} s "
          f"| y range [{r[:, 3].min():+.1f}, {r[:, 4].max():+.1f}] m")
