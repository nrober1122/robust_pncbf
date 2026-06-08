"""Four barrier-tightening strategies for the CBF-QP.

All four share the same min-norm CBF-QP (see ``double_integrator_cbf.cbf_qp_filter``)
and differ only in how the non-negative tightening term ``phi`` is computed.
The QP enforces the tightened constraint::

    Lf psi1(xhat) + LG psi1(xhat) . u + alpha_qp * psi1(xhat) + phi  <=  delta

so any ``phi >= 0`` makes the filter more conservative.  See
``dbint2d_cbf_compare.ipynb`` for the underlying math.

  plain    -- phi = 0                                               (baseline)
  mrcbf    -- phi = constant Lipschitz tightening (Dean et al.)     (conservative)
  pshrcbf  -- phi = max_{x in [xhat-eps, xhat+eps]} g(x) - g(xhat)  (oracle)
  nmrcbf   -- phi = PhiNet(xhat, eps)                               (learned PSHR approx)

where ``g(x) = Lf psi1(x) + alpha_qp * psi1(x)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # works as a package member (used by safe_ctrl_node)
    from . import double_integrator_cbf as dic
    from .phi_net import load_phi_net
except ImportError:  # falls back when run directly as a script
    import double_integrator_cbf as dic
    from phi_net import load_phi_net


# ---- plain ----------------------------------------------------------------

class PlainPhi:
    """``phi = 0``.  Preserves the baseline CBF behaviour."""

    name = "plain"

    def __call__(self, xhat: np.ndarray, p: dic.CBFParams) -> float:
        return 0.0


# ---- MR-CBF (Lipschitz tightening) ---------------------------------------

def estimate_lipschitz_per_dim(
    p: dic.CBFParams,
    state_lo: np.ndarray,
    state_hi: np.ndarray,
    n_samples: int = 5000,
    delta: float = 1e-3,
    margin: float = 0.05,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-dim Lipschitz constants for ``psi1``, ``Lf psi1``, ``||LG psi1||_2``.

    Samples states inside ``[state_lo, state_hi]`` but outside the obstacle
    (with ``margin`` slack), finite-differences each ingredient along each
    state axis, and returns the per-dim max.  Mirrors the notebook's
    ``_estimate_lipschitz_per_dim``.
    """
    rng = np.random.default_rng(seed)
    n_cand = n_samples * 5
    xs_cand = state_lo[None] + rng.random((n_cand, 4)) * (state_hi - state_lo)
    ox, oy = p.obstacle_xy
    outside = np.hypot(xs_cand[:, 0] - ox, xs_cand[:, 2] - oy) > (p.obstacle_radius + margin)
    xs = xs_cand[outside][:n_samples]
    if len(xs) < 100:
        raise RuntimeError(
            f"only {len(xs)} samples outside obstacle; widen state bounds or shrink margin")

    def ingredients(xi):
        psi1, grad = dic.hocbf_psi1(xi, p)
        Lf = float(grad @ dic.f(xi))
        LG = grad @ dic.G  # shape (2,)
        return psi1, Lf, LG

    n = len(xs)
    psi1_base = np.empty(n)
    Lf_base = np.empty(n)
    LG_base = np.empty((n, 2))
    for k, xi in enumerate(xs):
        psi1_base[k], Lf_base[k], LG_base[k] = ingredients(xi)

    L_B = np.zeros(4)
    L_Lf = np.zeros(4)
    L_LG = np.zeros(4)
    for dim in range(4):
        d_psi1 = np.empty(n)
        d_Lf = np.empty(n)
        d_LG = np.empty((n, 2))
        for k, xi in enumerate(xs):
            xp = xi.copy()
            xp[dim] += delta
            psi1_p, Lf_p, LG_p = ingredients(xp)
            d_psi1[k] = (psi1_p - psi1_base[k]) / delta
            d_Lf[k] = (Lf_p - Lf_base[k]) / delta
            d_LG[k] = (LG_p - LG_base[k]) / delta
        L_B[dim] = float(np.max(np.abs(d_psi1)))
        L_Lf[dim] = float(np.max(np.abs(d_Lf)))
        L_LG[dim] = float(np.max(np.linalg.norm(d_LG, axis=1)))

    return L_B, L_Lf, L_LG


class MRCBFPhi:
    """Constant Lipschitz-based tightening (Dean et al. 2020 / notebook's MR-CBF).

    With ``||u||_2 <= sqrt(nu) * u_max`` as a static bound::

        phi_const = sum_i eps_i (L_Lf_i + alpha_qp L_B_i)
                  + sqrt(nu) * u_max * sum_i eps_i L_LG_i
    """

    name = "mrcbf"

    def __init__(self, phi_const: float):
        self.phi_const = float(phi_const)

    @classmethod
    def precompute(
        cls,
        p: dic.CBFParams,
        state_eps: np.ndarray,
        state_lo: np.ndarray,
        state_hi: np.ndarray,
        n_samples: int = 5000,
        seed: int = 123,
    ) -> "MRCBFPhi":
        L_B, L_Lf, L_LG = estimate_lipschitz_per_dim(
            p, state_lo, state_hi, n_samples=n_samples, seed=seed)
        eps = np.asarray(state_eps, dtype=float)
        tight_const = float(np.sum(eps * (L_Lf + p.alpha_qp * L_B)))
        tight_LG = float(np.sum(eps * L_LG))
        nu = 2  # double-integrator: ux, uy
        tight_LG_eff = tight_LG * p.u_max * float(np.sqrt(nu))
        return cls(tight_const + tight_LG_eff)

    def __call__(self, xhat: np.ndarray, p: dic.CBFParams) -> float:
        return self.phi_const


# ---- PSHR-CBF (phi-oracle) ------------------------------------------------

class PSHRPhi:
    """Runtime projected-gradient ascent over ``[xhat - eps, xhat + eps]``.

    Computes ``phi_pshr = max(0, max_x g(x) - g(xhat))`` where
    ``g = Lf psi1 + alpha_qp psi1``.  Mirrors the notebook's ``_lfhwc_step``
    (``proj_grad_max_box`` with finite-difference gradients in place of jax
    autodiff).
    """

    name = "pshrcbf"

    def __init__(self, state_eps: np.ndarray, restarts: int = 4, steps: int = 25,
                 lr: float = 0.05, seed: int = 0, fd_h: float = 1e-5):
        self.eps = np.asarray(state_eps, dtype=float)
        self.restarts = int(restarts)
        self.steps = int(steps)
        self.lr = float(lr)
        self.fd_h = float(fd_h)
        self._rng = np.random.default_rng(int(seed))

    def __call__(self, xhat: np.ndarray, p: dic.CBFParams) -> float:
        lo = xhat - self.eps
        hi = xhat + self.eps
        g_nom = dic.lf_psi1_plus_alpha_psi1(xhat, p)
        best = -np.inf
        for _ in range(self.restarts):
            x = lo + self._rng.random(4) * (hi - lo)
            for _ in range(self.steps):
                g0 = dic.lf_psi1_plus_alpha_psi1(x, p)
                grad = np.zeros(4)
                for i in range(4):
                    xp = x.copy()
                    xp[i] += self.fd_h
                    grad[i] = (dic.lf_psi1_plus_alpha_psi1(xp, p) - g0) / self.fd_h
                x = np.clip(x + self.lr * grad, lo, hi)
            v = dic.lf_psi1_plus_alpha_psi1(x, p)
            if v > best:
                best = v
        return max(0.0, float(best - g_nom))


# ---- NMR-CBF (learned phi) ------------------------------------------------

class NMRCBFPhi:
    """Learned phi from a numpy-loaded PhiNet checkpoint."""

    name = "nmrcbf"

    def __init__(self, checkpoint_path, state_eps: np.ndarray):
        self.eps = np.asarray(state_eps, dtype=np.float32)
        self.net = load_phi_net(checkpoint_path)

    def __call__(self, xhat: np.ndarray, p: dic.CBFParams) -> float:
        return self.net(xhat, self.eps)


# ---- dispatch -------------------------------------------------------------

def build_phi_source(
    method: str,
    cbf: dic.CBFParams,
    state_eps,
    pshr_cfg: Optional[dict] = None,
    nmr_checkpoint: Optional[Path] = None,
    mr_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    mr_samples: int = 5000,
):
    """Construct a phi-source from a method name and parameters.

    ``method`` is one of ``plain | mrcbf | pshrcbf | nmrcbf`` (case-insensitive).
    """
    method = method.lower().strip()
    state_eps = np.asarray(state_eps, dtype=float)
    if method == "plain":
        return PlainPhi()
    if method == "mrcbf":
        if mr_bounds is None:
            raise ValueError("mrcbf requires mr_bounds=(state_lo, state_hi)")
        lo, hi = mr_bounds
        return MRCBFPhi.precompute(
            cbf, state_eps,
            np.asarray(lo, dtype=float), np.asarray(hi, dtype=float),
            n_samples=mr_samples)
    if method == "pshrcbf":
        cfg = pshr_cfg or {}
        return PSHRPhi(state_eps,
                       restarts=cfg.get("restarts", 4),
                       steps=cfg.get("steps", 25),
                       lr=cfg.get("lr", 0.05),
                       seed=cfg.get("seed", 0))
    if method == "nmrcbf":
        if nmr_checkpoint is None:
            raise ValueError("nmrcbf requires nmr_checkpoint")
        return NMRCBFPhi(nmr_checkpoint, state_eps)
    raise ValueError(
        f"unknown cbf_method '{method}'; choices: plain | mrcbf | pshrcbf | nmrcbf")
