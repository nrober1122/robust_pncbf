"""Feedback linearization: 2-D double integrator <-> unicycle commands.

Port of the ``unicycle`` branch of ``linearized.ipynb``.  A control point is
held a fixed distance ``d`` ahead of the robot::

    p_c = (x + d cos(psi),  y + d sin(psi))

Under unicycle kinematics (no lateral velocity, ``vy = 0``) the control-point
velocity relates to the robot commands ``(vx, vyaw)`` by::

    [xc_dot]   [cos psi   -d sin psi] [vx  ]
    [yc_dot] = [sin psi    d cos psi] [vyaw]

That 2x2 matrix has determinant ``d`` (always invertible for ``d > 0``), so the
linearizing map back to robot commands is its inverse::

    [vx  ]   [ cos psi        sin psi   ] [vxc]
    [vyaw] = [-sin psi / d    cos psi / d] [vyc]

The CBF runs on the control point -- it is the point that obeys holonomic
double-integrator dynamics under feedback linearization.  Because the robot
*body* trails the control point by ``d``, the CBF obstacle radius is inflated
by ``d`` (see ``safe_ctrl_node``).
"""
from __future__ import annotations

import numpy as np


def control_point(x: float, y: float, yaw: float, d: float):
    """World-frame position of the control point ``d`` ahead of the robot."""
    return x + d * np.cos(yaw), y + d * np.sin(yaw)


def linearizing_map(yaw: float, d: float) -> np.ndarray:
    """Matrix ``A(psi)`` such that ``[vx, vyaw] = A @ [vxc, vyc]``."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,       s],
                     [-s / d,  c / d]])


def to_unicycle_cmd(vxc: float, vyc: float, yaw: float, d: float):
    """Map a world-frame control-point velocity to ``(vx, vyaw)`` commands."""
    vx, vyaw = linearizing_map(yaw, d) @ np.array([vxc, vyc])
    return float(vx), float(vyaw)


def _demo_rollout() -> bool:
    """Closed-loop unicycle sim -- validates the control-point CBF + the A map.

    The control point is a double integrator filtered by the CBF-QP; its
    velocity is mapped through A(psi) to (vx, vyaw), which then drives unicycle
    kinematics.  Confirms the robot *body* clears the obstacle and reaches the
    goal.  Run directly:  ``python3 feedback_linearization.py``.
    """
    import double_integrator_cbf as dic

    d = 0.5
    obstacle = (2.0, 0.35)
    real_radius, body_margin = 0.30, 0.20
    p = dic.CBFParams(obstacle_xy=obstacle,
                      obstacle_radius=real_radius + body_margin + d,
                      alpha0=2.0, alpha_qp=1.0, u_max=0.5, qp_penalty=200.0)
    goal = np.array([3.6, 0.0])
    dt, steps, cp_speed_max = 0.02, 3000, 0.4
    vx_lim, vyaw_lim = (-0.15, 0.4), 0.8

    x, y, yaw = 0.0, 0.0, 0.0
    vc = np.zeros(2)
    body_min = np.inf
    for _ in range(steps):
        if np.hypot(x - goal[0], y - goal[1]) <= 0.15:
            break
        xc, yc = control_point(x, y, yaw, d)
        state = np.array([xc, vc[0], yc, vc[1]])
        goal_cp = goal + d * np.array([np.cos(yaw), np.sin(yaw)])
        u_nom = dic.nominal_goto(state, goal_cp, u_max=p.u_max)
        u, _, _ = dic.cbf_qp_filter(state, u_nom, p)
        vc = vc + u * dt
        speed = float(np.linalg.norm(vc))
        if speed > cp_speed_max:
            vc *= cp_speed_max / speed
        vx, vyaw = to_unicycle_cmd(vc[0], vc[1], yaw, d)
        vx = float(np.clip(vx, *vx_lim))
        vyaw = float(np.clip(vyaw, -vyaw_lim, vyaw_lim))
        x += vx * np.cos(yaw) * dt
        y += vx * np.sin(yaw) * dt
        yaw += vyaw * dt
        body_min = min(body_min, float(np.hypot(x - obstacle[0], y - obstacle[1])))

    goal_dist = float(np.hypot(x - goal[0], y - goal[1]))
    want = real_radius + body_margin
    ok = body_min >= want - 0.05 and goal_dist <= 0.2
    print(f"  body closest approach to obstacle = {body_min:.3f} m "
          f"(real R={real_radius}, want >= {want:.2f})")
    print(f"  final body distance to goal       = {goal_dist:.3f} m")
    print("UNICYCLE ROLLOUT OK" if ok else "*** UNICYCLE ROLLOUT FAILED ***")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _demo_rollout() else 1)
