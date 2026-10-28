"""R-CBF-QP and Duality CBF baselines (numpy ports of the notebook versions).

Both are measurement-robust baselines from ``dbint2d_cbf_compare.ipynb``:

  rcbfqp   -- R-CBF-QP (arXiv 2508.19159).  The constraint is tightened by
              ``rho = gamma1 ||LG psi1|| + gamma2^2 ||LG psi1||^2``, with
              (gamma1, gamma2) chosen each tick by grid search over sampled
              states in the eps-box (notebook ``_rcbf_step`` +
              ``compute_rcbf_gammas`` = option C in ``mrncbf.qp.min_norm_cbf``).
              It is a phi-source: ``phi = rho`` slots into the shared QP.

  duality  -- Duality CBF (notebook ``_duality_step``).  The image map
              ``z(x) = [LG psi1(x), Lf psi1(x) + alpha_qp psi1(x)]`` over the
              eps-box is outer-approximated by the polytope
              ``P = {z : D z <= d}``, with ``d_j`` = support function along
              Fibonacci direction ``D_j`` (projected-gradient ascent).  The
              robust constraint ``max_{z in P} z . [u; 1] <= 0`` is enforced.

How the Duality CBF QP is solved
--------------------------------
The notebook writes the robust constraint through its LP dual certificate
(``lambda >= 0, D^T lambda = [u; 1], d^T lambda <= r``) and hands it to
JaxProxQP.  By LP strong duality (P is bounded and non-empty: the directions
positively span R^3 and ``z(xhat)`` lies in P), that is exactly
``max_{v in vert(P)} v . [u; 1] <= r``.  We enumerate the vertices of the 3-D
polytope directly (intersect every triple of planes, keep the feasible
points) and pass one row per vertex to ``min_norm_qp_rows`` -- so this stays
numpy-only like the rest of the package.

Differences from the notebook, both shared with every other method here:
the slack is the package's per-row quadratic slack (``qp_penalty``, no
``relax_eps`` margin) and the acceleration box is enforced by clipping.
"""
from __future__ import annotations

import itertools
from typing import Callable, Optional

import numpy as np

try:  # works as a package member (used by safe_ctrl_node)
    from . import double_integrator_cbf as dic
except ImportError:  # falls back when run directly as a script
    import double_integrator_cbf as dic


# ---- R-CBF-QP --------------------------------------------------------------

class RCBFQPPhi:
    """R-CBF-QP tightening ``rho(gamma1, gamma2)`` as a phi-source.

    Needs the nominal policy at perturbed states (``nominal_fn``), since the
    gamma cost compares filtered controls across the eps-box.  Defaults match
    the notebook: 20 samples, 50x50 grid on [1e-3, 8], reg 0.1, smooth 0.05
    anchored at 0 (no temporal smoothing), feasibility filter on.
    """

    name = "rcbfqp"
    needs_nominal = True

    def __init__(self, state_eps: np.ndarray, n_samples: int = 20,
                 gamma_min: float = 1e-3, gamma_max: float = 8.0,
                 n_grid: int = 50, reg_weight: float = 0.1,
                 smooth_weight: float = 0.05, seed: int = 0):
        self.eps = np.asarray(state_eps, dtype=float)
        self.n_samples = int(n_samples)
        self.reg_weight = float(reg_weight)
        self.smooth_weight = float(smooth_weight)
        vals = np.linspace(gamma_min, gamma_max, int(n_grid))
        g1, g2 = np.meshgrid(vals, vals, indexing="ij")
        self._g1 = g1.ravel()
        self._g2 = g2.ravel()
        self._rng = np.random.default_rng(int(seed))
        self.last_gammas = (0.0, 0.0)

    def select_gammas(self, X: np.ndarray, U: np.ndarray, p: dic.CBFParams,
                      tol: float = 1e-6):
        """Grid-search (gamma1, gamma2).  ``X[0]``/``U[0]`` are the estimate.

        Vectorized ``_compute_rcbf_gammas_core`` (option C) for a single
        constraint: closed-form projection onto each tightened halfspace,
        sigma_hat = max deviation of perturbed filtered controls from the
        nominal one, cost = max(sigma_hat - g1, 0) / (2 g2) + regularizers.
        """
        LG, c = dic.cbf_ingredients_batch(X, p)            # (S, 2), (S,)
        nsq = np.sum(LG ** 2, axis=1)
        nrm = np.sqrt(nsq)
        g1, g2 = self._g1[:, None], self._g2[:, None]
        rho = g1 * nrm[None] + g2 ** 2 * nsq[None]          # (Ng, S)
        deficit = np.sum(LG * U, axis=1)[None] + c[None] + rho
        gain = np.maximum(deficit, 0.0) / (nsq[None] + 1e-12)
        u_star = U[None] - gain[..., None] * LG[None]        # (Ng, S, 2)
        feas = np.all(np.abs(u_star) <= p.u_max + tol, axis=(1, 2))
        u_filt = np.clip(u_star, -p.u_max, p.u_max)
        sigma = np.max(np.linalg.norm(u_filt[:, 1:] - u_filt[:, :1], axis=-1),
                       axis=1)

        g1, g2 = self._g1, self._g2
        cost = np.maximum(sigma - g1, 0.0) / (2.0 * g2 + 1e-12)
        cost = cost + self.reg_weight * (g1 + g2)
        cost = cost + self.smooth_weight * (g1 ** 2 + g2 ** 2)
        cost = np.where(feas, cost, 1e6)
        best = int(np.argmin(cost))
        return float(g1[best]), float(g2[best])

    def __call__(self, xhat: np.ndarray, p: dic.CBFParams,
                 nominal_fn: Optional[Callable] = None) -> float:
        if nominal_fn is None:
            raise ValueError("rcbfqp needs nominal_fn (nominal policy at perturbed states)")
        lo, hi = xhat - self.eps, xhat + self.eps
        X = np.vstack([xhat[None],
                       lo + self._rng.random((self.n_samples, 4)) * (hi - lo)])
        U = np.array([nominal_fn(x) for x in X])
        g1, g2 = self.select_gammas(X, U, p)
        self.last_gammas = (g1, g2)
        LG, _ = dic.cbf_ingredients_batch(xhat[None], p)
        nrm = float(np.linalg.norm(LG[0]))
        return g1 * nrm + g2 ** 2 * nrm ** 2


# ---- Duality CBF -----------------------------------------------------------

def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` near-uniform unit directions on S^2 (the notebook's ``_DUAL_DIRS``)."""
    golden = (1 + np.sqrt(5)) / 2
    ks = np.arange(n)
    theta = np.arccos(1 - 2 * (ks + 0.5) / n)
    phi = 2 * np.pi * ks / golden
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)


def polytope_vertices(D: np.ndarray, d: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Vertices of the bounded 3-D polytope ``{z : D z <= d}`` (brute force).

    Intersects every triple of planes and keeps the points satisfying all
    constraints.  With 16 planes that is 560 3x3 solves -- negligible.
    """
    tri = np.array(list(itertools.combinations(range(D.shape[0]), 3)))
    M = D[tri]                                             # (T, 3, 3)
    rhs = d[tri]                                           # (T, 3)
    ok = np.abs(np.linalg.det(M)) > 1e-9
    V = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
    inside = np.all(V @ D.T <= d[None] + tol * (1 + np.abs(d[None])), axis=1)
    V = V[inside]
    if V.shape[0] == 0:
        raise RuntimeError("duality polytope has no vertices (empty or unbounded)")
    return np.unique(np.round(V, 9), axis=0)


class DualityCBF:
    """Duality-based robust CBF.  Not a scalar phi: it emits several QP rows.

    ``constraint_rows(xhat, p)`` returns ``(A, b)`` with one row per polytope
    vertex ``v``: ``A = v[:2]``, ``b = v[2]`` (so each row reads
    ``v_LG . u + v_c <= slack``).  Defaults match the notebook: 16 planes,
    2 restarts x 15 steps of projected-gradient ascent, lr 0.05.
    """

    name = "duality"
    needs_nominal = False

    def __init__(self, state_eps: np.ndarray, n_planes: int = 16,
                 restarts: int = 2, steps: int = 15, lr: float = 0.05,
                 seed: int = 0, fd_h: float = 1e-5):
        self.eps = np.asarray(state_eps, dtype=float)
        self.D = fibonacci_directions(int(n_planes))
        self.restarts = int(restarts)
        self.steps = int(steps)
        self.lr = float(lr)
        self.fd_h = float(fd_h)
        self._rng = np.random.default_rng(int(seed))

    @staticmethod
    def _image(X: np.ndarray, p: dic.CBFParams) -> np.ndarray:
        """``z(x) = [LG psi1, Lf psi1 + alpha_qp psi1]`` for a batch, shape (M, 3)."""
        LG, c = dic.cbf_ingredients_batch(X, p)
        return np.concatenate([LG, c[:, None]], axis=1)

    def support(self, xhat: np.ndarray, p: dic.CBFParams) -> np.ndarray:
        """``d_j ~ max_{x in box} D_j . z(x)`` for every direction (shape (N,)).

        All ``N * restarts`` ascents run as one batch, with forward-difference
        gradients (as in ``PSHRPhi``).  ``d`` is floored at ``D z(xhat)`` so
        the polytope always contains the estimate's image (non-empty).
        """
        lo, hi = xhat - self.eps, xhat + self.eps
        N, R = self.D.shape[0], self.restarts
        dirs = np.repeat(self.D, R, axis=0)                    # (N*R, 3)
        X = lo + self._rng.random((N * R, 4)) * (hi - lo)
        H = np.eye(4) * self.fd_h

        def value(Xb, dirs_b):
            return np.sum(dirs_b * self._image(Xb, p), axis=1)

        for _ in range(self.steps):
            v0 = value(X, dirs)
            Xp = (X[:, None, :] + H[None]).reshape(-1, 4)
            vp = value(Xp, np.repeat(dirs, 4, axis=0)).reshape(-1, 4)
            grad = (vp - v0[:, None]) / self.fd_h
            X = np.clip(X + self.lr * grad, lo, hi)

        d = value(X, dirs).reshape(N, R).max(axis=1)
        return np.maximum(d, self.D @ self._image(xhat[None], p)[0])

    def constraint_rows(self, xhat: np.ndarray, p: dic.CBFParams):
        V = polytope_vertices(self.D, self.support(xhat, p))
        return V[:, :2], V[:, 2]


# ---- shared filter step ----------------------------------------------------

def solve_filter(source, xhat: np.ndarray, u_nom: np.ndarray, obs_params,
                 nominal_fn: Optional[Callable] = None):
    """One CBF-QP filter step for any method.  Returns ``(u_safe, psi1s, phis, slacks)``.

    Scalar phi-sources go through ``cbf_qp_filter_multi`` (one row per
    obstacle).  The Duality CBF stacks every obstacle's vertex rows into one
    QP; its reported ``phi`` is the effective tightening at the chosen
    control, ``max_v v . [u; 1] - z(xhat) . [u; 1] >= 0``, and its slack is
    the largest row slack of that obstacle -- so diagnostics stay comparable.
    """
    if isinstance(source, DualityCBF):
        rows = [source.constraint_rows(xhat, p) for p in obs_params]
        A = np.vstack([r[0] for r in rows])
        b = np.concatenate([r[1] for r in rows])
        owner = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(rows)])
        u_safe, row_slacks = dic.min_norm_qp_rows(
            A, b, u_nom, obs_params[0].qp_penalty, obs_params[0].u_max)
        psi1s = np.array([dic.hocbf_psi1(xhat, p)[0] for p in obs_params])
        z_hat = np.vstack([DualityCBF._image(xhat[None], p) for p in obs_params])
        nominal_val = z_hat[:, :2] @ u_safe + z_hat[:, 2]
        robust_val = A @ u_safe + b
        phis = np.array([robust_val[owner == i].max() for i in range(len(rows))])
        phis = np.maximum(phis - nominal_val, 0.0)
        slacks = np.array([row_slacks[owner == i].max() for i in range(len(rows))])
        return u_safe, psi1s, phis, slacks

    kw = {"nominal_fn": nominal_fn} if getattr(source, "needs_nominal", False) else {}
    phis = np.array([float(source(xhat, p, **kw)) for p in obs_params])
    u_safe, psi1s, slacks = dic.cbf_qp_filter_multi(xhat, u_nom, obs_params, phis)
    return u_safe, psi1s, phis, slacks
