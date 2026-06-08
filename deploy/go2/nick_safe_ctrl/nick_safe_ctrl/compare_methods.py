"""Closed-loop unicycle rollout comparing the four phi-source strategies.

Mirrors the subset of the notebook's ``_ALL_ROLLOUTS`` that we ported:
``plain``, ``mrcbf``, ``pshrcbf``, ``nmrcbf``.  For each method, simulates a
unicycle robot doing CBF obstacle avoidance under the same persistent
position bias and reports:

  - body's closest approach to the obstacle (smaller = riskier)
  - whether the body penetrated the real obstacle (key safety check)
  - final distance to the goal

Run directly::

    python3 compare_methods.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _run(method, phi_source, *, bias, dt=0.02, steps=3000):
    """Simulate one rollout under a given phi_source + persistent bias."""
    import double_integrator_cbf as dic
    from feedback_linearization import control_point, to_unicycle_cmd

    d = 0.5
    obstacle = (2.0, 0.35)
    real_radius, body_margin = 0.30, 0.20
    p = dic.CBFParams(
        obstacle_xy=obstacle,
        obstacle_radius=real_radius + body_margin + d,
        alpha0=2.0, alpha_qp=1.0, u_max=0.5, qp_penalty=200.0,
    )
    goal = np.array([3.6, 0.0])
    cp_speed_max = 0.4
    vx_lim, vyaw_lim = (-0.15, 0.4), 0.8

    x, y, yaw = 0.0, 0.0, 0.0
    vc = np.zeros(2)
    body_min = np.inf
    real_violation = False
    bias = np.asarray(bias, dtype=float)
    for _ in range(steps):
        if np.hypot(x - goal[0], y - goal[1]) <= 0.15:
            break
        xc, yc = control_point(x, y, yaw, d)
        true_state = np.array([xc, vc[0], yc, vc[1]])
        xhat = true_state + bias                              # noisy estimate
        goal_cp = goal + d * np.array([np.cos(yaw), np.sin(yaw)])
        u_nom = dic.nominal_goto(xhat, goal_cp, u_max=p.u_max)
        phi = phi_source(xhat, p)
        u_safe, _, _ = dic.cbf_qp_filter(xhat, u_nom, p, phi=phi)
        vc = vc + u_safe * dt
        sp = float(np.linalg.norm(vc))
        if sp > cp_speed_max:
            vc *= cp_speed_max / sp
        vx, vyaw = to_unicycle_cmd(vc[0], vc[1], yaw, d)
        vx = float(np.clip(vx, *vx_lim))
        vyaw = float(np.clip(vyaw, -vyaw_lim, vyaw_lim))
        x += vx * np.cos(yaw) * dt
        y += vx * np.sin(yaw) * dt
        yaw += vyaw * dt
        body_dist = float(np.hypot(x - obstacle[0], y - obstacle[1]))
        body_min = min(body_min, body_dist)
        if body_dist < real_radius:
            real_violation = True

    goal_dist = float(np.hypot(x - goal[0], y - goal[1]))
    safe_margin = body_min - real_radius                       # > 0 means body cleared
    return {
        "method": method,
        "body_min": body_min,
        "safe_margin": safe_margin,
        "real_violation": real_violation,
        "goal_dist": goal_dist,
    }


def main():
    import double_integrator_cbf as dic
    from phi_sources import build_phi_source

    obstacle = (2.0, 0.35)
    real_radius, body_margin, d = 0.30, 0.20, 0.5
    p = dic.CBFParams(
        obstacle_xy=obstacle,
        obstacle_radius=real_radius + body_margin + d,
        alpha0=2.0, alpha_qp=1.0, u_max=0.5, qp_penalty=200.0,
    )
    state_eps = np.array([0.10, 0.05, 0.10, 0.05])
    # Persistent position-bias noise -- worst-case direction: +px (perceives
    # the obstacle as farther away than it is). Within state_eps bound.
    bias = np.array([0.10, 0.0, 0.0, 0.0])

    ckpt = Path(__file__).resolve().parent.parent / "ckpts" / "phi_params.pkl"

    sources = {
        "plain":   build_phi_source("plain", p, state_eps),
        "mrcbf":   build_phi_source("mrcbf", p, state_eps,
                                    mr_bounds=(np.array([-3., -2., -3., -2.]),
                                               np.array([ 3.,  2.,  3.,  2.])),
                                    mr_samples=2000),
        "pshrcbf": build_phi_source("pshrcbf", p, state_eps),
        "nmrcbf":  build_phi_source("nmrcbf", p, state_eps, nmr_checkpoint=ckpt),
    }

    print(f"obstacle: real R={real_radius}, body_margin={body_margin}, "
          f"control-point d={d} -> R_cbf={p.obstacle_radius:.2f}")
    print(f"bias (constant per rollout): {bias.tolist()}, state_eps={state_eps.tolist()}")
    if isinstance(sources["mrcbf"], type(sources["mrcbf"])):
        print(f"MR-CBF phi_const = {sources['mrcbf'].phi_const:.4f}")
    print()
    print(f"  {'method':<10} {'body_min':>9} {'safe_margin':>12} {'penetrated':>11} {'goal_dist':>10}")
    print(f"  {'-'*10:<10} {'-'*9:>9} {'-'*12:>12} {'-'*11:>11} {'-'*10:>10}")
    for name, src in sources.items():
        res = _run(name, src, bias=bias)
        flag = "**YES**" if res["real_violation"] else "no"
        print(f"  {res['method']:<10} {res['body_min']:>9.3f} "
              f"{res['safe_margin']:>12.3f} {flag:>11} {res['goal_dist']:>10.3f}")


if __name__ == "__main__":
    main()
