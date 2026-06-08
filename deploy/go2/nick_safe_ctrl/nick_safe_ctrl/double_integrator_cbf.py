"""Standalone 2-D double-integrator obstacle-avoidance CBF (no JAX).

Numpy port of the ``_cbf_step`` baseline in ``dbint2d_cbf_compare.ipynb`` /
``dbint2d_quad.py``.  The neural NMR-CBF term ``phi`` is intentionally omitted:
in the notebook NMR-CBF only ever adds ``phi / alpha_qp`` to the barrier, and
setting ``phi = 0`` reduces NMR-CBF *exactly* to this plain CBF.  To re-enable
the network later, add ``phi / alpha_qp`` to ``psi1`` inside ``cbf_qp_filter``
(``phi`` from a numpy re-implementation of the trained PhiNet).

State ``x = (px, vx, py, vy)`` for a 2-D double integrator::

    f(x) = [vx, 0, vy, 0]      G = [[0, 0], [1, 0], [0, 0], [0, 1]]
    xdot = f(x) + G u          u = (ux, uy)   (acceleration)

Safety:  ``h(x) = R_obs - ||p - p_obs||``;  ``h < 0`` <=> safe (outside the
obstacle).  Relative degree 2  =>  HOCBF  ``psi1 = Lf h + alpha0 * h``.
CBF-QP constraint:  ``Lf psi1 + LG psi1 . u + alpha_qp * psi1 <= 0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Control-effectiveness matrix G, constant for the double integrator.
G = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 0.0],
              [0.0, 1.0]])


@dataclass
class CBFParams:
    """Obstacle geometry and CBF-QP gains."""

    obstacle_xy: tuple        # obstacle centre, metres (CBF working frame)
    obstacle_radius: float    # CBF keep-out radius -- already inflated, see node
    alpha0: float = 2.0       # HOCBF class-K gain (ALPHA_HOCBF in the notebook)
    alpha_qp: float = 1.0     # CBF-QP class-K gain (ALPHA_QP)
    u_max: float = 0.5        # per-axis acceleration bound
    qp_penalty: float = 200.0  # CBF-QP slack penalty (see cbf_qp_filter notes)
    rho_eps: float = 1e-3     # floor on ||p - p_obs|| to avoid divide-by-zero


def f(x: np.ndarray) -> np.ndarray:
    """Drift dynamics f(x) = [vx, 0, vy, 0]."""
    px, vx, py, vy = x
    return np.array([vx, 0.0, vy, 0.0])


def barrier_h(x: np.ndarray, p: CBFParams) -> float:
    """Raw barrier h = R_obs - ||p - p_obs||;  h < 0  <=>  safe (outside)."""
    px, _, py, _ = x
    ox, oy = p.obstacle_xy
    return p.obstacle_radius - float(np.hypot(px - ox, py - oy))


def lf_psi1_plus_alpha_psi1(x: np.ndarray, p: CBFParams) -> float:
    """``g(x) = Lf psi1(x) + alpha_qp * psi1(x)``.

    The CBF-QP constraint at state ``x`` is ``g(x) + LG_psi1 . u <= 0``; PSHR-CBF
    maximizes this g over an uncertainty box to get a worst-case tightening.
    """
    psi1, grad = hocbf_psi1(x, p)
    return float(grad @ f(x)) + p.alpha_qp * psi1


def hocbf_psi1(x: np.ndarray, p: CBFParams):
    """Return ``(psi1, grad_psi1)`` -- the HOCBF value and its state gradient.

    ``psi1 > 0`` means the CBF-QP must intervene.  ``grad_psi1`` has shape (4,).
    Derivation: with ``d = p - p_obs``, ``rho = ||d||``, ``s = d . v``,
    ``psi1 = -s/rho + alpha0 (R - rho)``; the partials below are that
    expression differentiated analytically (no autodiff needed).
    """
    px, vx, py, vy = x
    ox, oy = p.obstacle_xy
    dx, dy = px - ox, py - oy
    rho = max(float(np.hypot(dx, dy)), p.rho_eps)
    s = dx * vx + dy * vy                       # s = -rho * Lf h
    a0 = p.alpha0

    psi1 = -s / rho + a0 * (p.obstacle_radius - rho)

    dpsi_dpx = -vx / rho + s * dx / rho ** 3 - a0 * dx / rho
    dpsi_dpy = -vy / rho + s * dy / rho ** 3 - a0 * dy / rho
    dpsi_dvx = -dx / rho
    dpsi_dvy = -dy / rho
    grad = np.array([dpsi_dpx, dpsi_dvx, dpsi_dpy, dpsi_dvy])
    return psi1, grad


def nominal_goto(x: np.ndarray, goal_xy, kp: float = 2.0, kd: float = 2.0,
                 u_max: float = 0.5) -> np.ndarray:
    """PD acceleration driving (px, py) -> goal, clipped to the inf-norm box.

    This is ``nom_pol_goto`` from ``dbint2d_quad.py``.
    """
    px, vx, py, vy = x
    pos = np.array([px, py])
    vel = np.array([vx, vy])
    u = -kp * (pos - np.asarray(goal_xy, dtype=float)) - kd * vel
    norm_inf = float(np.max(np.abs(u)))
    if norm_inf > u_max:
        u = u / norm_inf * u_max
    return u


def cbf_qp_filter(x: np.ndarray, u_nom: np.ndarray, p: CBFParams,
                  phi: float = 0.0):
    """Min-norm CBF safety filter.  Returns ``(u_safe, psi1, slack)``.

    Solves the relaxed single-constraint QP::

        min  1/2 ||u - u_nom||^2 + 1/2 * qp_penalty * delta^2
        s.t. LG_psi1 . u + (Lf_psi1 + alpha_qp * psi1 + phi) <= delta,  delta >= 0

    which has the closed form ``u = u_nom - lambda * a`` with
    ``lambda = max(0, (a.u_nom + b) / (||a||^2 + 1/qp_penalty))``.  The
    acceleration box is then enforced by clipping (standard practical handling;
    the slack absorbs any residual constraint violation it introduces).

    ``phi >= 0`` is the optional barrier tightening (defaults to 0 -- plain CBF).
    Robust variants pass a non-zero phi from a ``phi_source`` (see
    ``phi_sources.py``); the constraint then enforces the tightened
    ``psi1 + phi/alpha_qp`` instead of psi1.

    Note: the relaxation permits a steady-state barrier softening of about
    ``1/(qp_penalty * alpha_qp * alpha0)`` (e.g. a head-on stop creeps that far
    inside h=0).  The notebook used ``relax_eps2 = 0.1`` (~ qp_penalty 10) for
    its policy-comparison study; for deployment keep ``qp_penalty`` large so
    this softening stays well below ``body_margin``.
    """
    psi1, grad = hocbf_psi1(x, p)
    a = grad @ G                                              # LG psi1, shape (2,)
    b = float(grad @ f(x)) + p.alpha_qp * psi1 + float(phi)   # Lf psi1 + alpha_qp * psi1 + phi

    aa = float(a @ a)
    violation = float(a @ u_nom + b)
    lam = max(0.0, violation / (aa + 1.0 / p.qp_penalty))

    u_safe = np.clip(u_nom - lam * a, -p.u_max, p.u_max)
    slack = lam / p.qp_penalty
    return u_safe, psi1, slack


def cbf_qp_filter_multi(x: np.ndarray, u_nom: np.ndarray, obs_params,
                        phis, pgs_sweeps: int = 50):
    """Min-norm CBF safety filter with **one barrier constraint per obstacle**.

    ``obs_params`` is a sequence of :class:`CBFParams` that share gains
    (``alpha_qp``, ``u_max``, ``qp_penalty``) but each carry a distinct
    obstacle; ``phis`` is the matching sequence of per-obstacle tightening
    terms ``phi_i >= 0`` (0 for plain CBF).  Returns ``(u_safe, psi1s, slacks)``
    where ``psi1s`` / ``slacks`` are per-obstacle arrays aligned with
    ``obs_params``.

    Solves the relaxed QP (per-constraint slack)::

        min_{u, d}  1/2 ||u - u_nom||^2 + (P/2) sum_i d_i^2
        s.t.        a_i . u + b_i <= d_i           (one row per obstacle)

    with ``a_i = LG psi1_i``, ``b_i = Lf psi1_i + alpha_qp psi1_i + phi_i``.
    Its dual is ``min 1/2 lam^T K lam - c^T lam,  lam >= 0`` with
    ``K = A A^T + (1/P) I`` (symmetric positive-definite), ``c = A u_nom + b``;
    the optimal control is ``u = u_nom - A^T lam``.  We solve the dual with
    projected Gauss-Seidel (coordinate descent with a max(0, .) projection),
    which converges for PD ``K`` and, for a **single** obstacle, collapses to
    exactly the closed form in :func:`cbf_qp_filter`.  The acceleration box is
    enforced by a final clip, as in the single-constraint version.
    """
    n = len(obs_params)
    if n == 0:
        raise ValueError("cbf_qp_filter_multi needs >= 1 obstacle")
    P = obs_params[0].qp_penalty
    u_max = obs_params[0].u_max

    A = np.zeros((n, 2))
    b = np.zeros(n)
    psi1s = np.zeros(n)
    fx = f(x)
    for i, (pi, phi_i) in enumerate(zip(obs_params, phis)):
        psi1, grad = hocbf_psi1(x, pi)
        A[i] = grad @ G                                   # LG psi1_i
        b[i] = float(grad @ fx) + pi.alpha_qp * psi1 + float(phi_i)
        psi1s[i] = psi1

    K = A @ A.T + np.eye(n) / P
    c = A @ u_nom + b
    Kdiag = np.diag(K).copy()                              # all > 0 (1/P floor)
    lam = np.zeros(n)
    for _ in range(pgs_sweeps):
        for i in range(n):
            r = c[i] - (K[i] @ lam - K[i, i] * lam[i])
            lam[i] = max(0.0, r / Kdiag[i])

    u_safe = np.clip(u_nom - A.T @ lam, -u_max, u_max)
    slacks = lam / P
    return u_safe, psi1s, slacks


def _demo_rollout() -> bool:
    """Closed-loop sanity check -- reproduces the notebook's ``_cbf_step``.

    No robot, no feedback linearization: integrate the bare double integrator
    under (nominal -> CBF-QP) and confirm the obstacle is never penetrated.
    Run directly:  ``python3 double_integrator_cbf.py``.
    """
    p = CBFParams(obstacle_xy=(0.0, 0.0), obstacle_radius=0.25,
                  alpha0=2.0, alpha_qp=1.0, u_max=1.0, qp_penalty=200.0)
    goal = np.array([2.0, 0.0])
    dt, steps = 0.05, 400
    ics = [np.array([-1.5, 0.0, 0.10, 0.0]),
           np.array([-1.5, 0.0, -0.10, 0.0]),
           np.array([-2.0, 0.0, 0.30, 0.0]),
           np.array([-1.8, 0.0, 0.00, 0.0])]   # collinear: expected to stop short

    all_safe = True
    for x0 in ics:
        x = x0.astype(float).copy()
        min_rho = np.inf
        for _ in range(steps):
            u_nom = nominal_goto(x, goal, u_max=p.u_max)
            u, _, _ = cbf_qp_filter(x, u_nom, p)
            x = x + dt * (f(x) + G @ u)
            min_rho = min(min_rho, float(np.hypot(x[0], x[2])))
        safe = min_rho >= p.obstacle_radius - 2e-2
        all_safe &= safe
        goal_dist = float(np.hypot(x[0] - goal[0], x[2] - goal[1]))
        print(f"  IC {x0.tolist()}: min obstacle dist = {min_rho:.3f} "
              f"(R = {p.obstacle_radius}) -> {'SAFE' if safe else 'VIOLATION'}"
              f";  final goal dist = {goal_dist:.3f}")
    print("ALL TRAJECTORIES SAFE" if all_safe
          else "*** SOME TRAJECTORIES VIOLATED THE OBSTACLE ***")
    return all_safe


def _demo_rollout_multi() -> bool:
    """Two-obstacle sanity check for :func:`cbf_qp_filter_multi`.

    Two keep-outs straddle the start->goal line; the nominal points straight at
    the goal, so the body must thread between / around both.  Confirms neither
    obstacle is penetrated and that for one obstacle the multi solver matches
    the single-constraint closed form.
    """
    gains = dict(alpha0=2.0, alpha_qp=1.0, u_max=1.0, qp_penalty=200.0)
    obs = [
        (CBFParams(obstacle_xy=(1.0, 0.25), obstacle_radius=0.30, **gains), 0.30),
        (CBFParams(obstacle_xy=(2.0, -0.25), obstacle_radius=0.30, **gains), 0.30),
    ]
    obs_params = [c for c, _ in obs]
    goal = np.array([3.0, 0.0])
    dt, steps = 0.05, 600

    # (a) one-obstacle agreement check: multi == single closed form.
    x_chk = np.array([0.3, 0.4, 0.1, -0.2])
    u_nom_chk = nominal_goto(x_chk, goal, u_max=1.0)
    u_single, _, _ = cbf_qp_filter(x_chk, u_nom_chk, obs_params[0])
    u_multi, _, _ = cbf_qp_filter_multi(x_chk, u_nom_chk, obs_params[:1], [0.0])
    agree = bool(np.allclose(u_single, u_multi, atol=1e-9))
    print(f"  single-obstacle agreement (multi vs closed form): "
          f"{'OK' if agree else 'MISMATCH'} (max diff {np.max(np.abs(u_single-u_multi)):.2e})")

    # (b) two-obstacle rollout.
    x = np.array([0.0, 0.0, 0.0, 0.0])
    min_d = [np.inf, np.inf]
    for _ in range(steps):
        u_nom = nominal_goto(x, goal, u_max=gains["u_max"])
        u, _, _ = cbf_qp_filter_multi(x, u_nom, obs_params, [0.0, 0.0])
        x = x + dt * (f(x) + G @ u)
        for k, (c, _) in enumerate(obs):
            ox, oy = c.obstacle_xy
            min_d[k] = min(min_d[k], float(np.hypot(x[0] - ox, x[2] - oy)))
    safe = all(min_d[k] >= obs[k][1] - 2e-2 for k in range(2))
    goal_dist = float(np.hypot(x[0] - goal[0], x[2] - goal[1]))
    for k, (c, real_r) in enumerate(obs):
        print(f"  obstacle {k} @ {c.obstacle_xy}: min dist = {min_d[k]:.3f} "
              f"(R = {real_r}) -> {'SAFE' if min_d[k] >= real_r - 2e-2 else 'VIOLATION'}")
    print(f"  final goal dist = {goal_dist:.3f}")
    ok = agree and safe
    print("MULTI-OBSTACLE OK" if ok else "*** MULTI-OBSTACLE FAILED ***")
    return ok


if __name__ == "__main__":
    print("== single-obstacle ==")
    a = _demo_rollout()
    print("== multi-obstacle ==")
    b = _demo_rollout_multi()
    raise SystemExit(0 if (a and b) else 1)
