import functools as ft

import einops as ei
import jax.numpy as jnp
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP, QPSolution

from pncbf.dyn.dyn_types import Control, HFloat, HState, State
from pncbf.solvers.qp import get_relaxed_constr_Gh, jaxopt_osqp
from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.jax_utils import jax_vmap


def min_norm_cbf_qp_mats(
    alpha: FloatScalar,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    nx, nu = G.shape
    dtype = h_V.dtype
    assert u_lb.shape == u_ub.shape == (nu,)

    if h_V.ndim == 0:
        h_V = h_V[None]

        assert hx_Vx.shape == (nx,)
        hx_Vx = hx_Vx[None]

    H = np.eye(nu + 1, dtype=dtype)
    # 0.5 * penalty * ( r + relax_eps2 )^2
    # = 0.5 * penalty * ( r^2 + 2 r relax_eps2 + relax_eps2^2 )
    H[-1, -1] = penalty
    # We now have a nominal control.
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)], axis=0)
    assert g.shape == (nu + 1,)

    # Get G and h for each CBF constraint.
    if isinstance(alpha, float):
        alpha = jnp.array(alpha)

    if alpha.ndim == 0:
        h_G, h_h = jax_vmap(ft.partial(get_relaxed_constr_Gh, f, G, alpha))(h_V, hx_Vx)
    else:
        nh = len(h_V)
        assert alpha.shape == (nh,)
        h_alpha = alpha
        h_G, h_h = jax_vmap(ft.partial(get_relaxed_constr_Gh, f, G))(h_alpha, h_V, hx_Vx)

    # -eps1 <= r <= infty
    r_lb = jnp.array(-relax_eps1, dtype=dtype)
    r_ub = jnp.array(1e9, dtype=dtype)

    l_box = jnp.concatenate([u_lb, r_lb[None]], axis=0)
    u_box = jnp.concatenate([u_ub, r_ub[None]], axis=0)

    # C = jnp.stack([*h_G], axis=0)
    # u = jnp.array([*h_h])
    C = h_G
    u = h_h

    return JaxProxQP.QPModel.create(H, g, C, u, l_box, u_box)


def min_norm_cbf(
    alpha: FloatScalar,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings: JaxProxQP.Settings = None,
) -> tuple[Control, FloatScalar, QPSolution]:
    nx, nu = G.shape

    qp = min_norm_cbf_qp_mats(alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
    if settings is None:
        settings = JaxProxQP.Settings.default()
    solver = JaxProxQP(qp, settings)
    sol = solver.solve()

    assert sol.x.shape == (nu + 1,)
    u_opt, r = sol.x[:nu], sol.x[-1]
    return u_opt, r, sol


def lb_bangbang(u_lb: Control, u_ub: Control, G, Vx):
    # Minimize V.
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")
    # To minimize, if LG_V is negative, we want u to be as large as possible.
    u_opt = jnp.where(LG_V < 0, u_ub, u_lb)
    return u_opt

# ── shared helpers ────────────────────────────────────────────────────────────

def _lie_derivs(
    h_V: HFloat,       # (nh,)
    hx_Vx: HState,     # (nh, nx)
    f: State,          # (nx,)
    G,                 # (nx, nu)
    alpha,             # scalar or (nh,)
) -> tuple:
    """
    Returns per-constraint Lie derivative scalars ready for QP assembly.

    h_Lf_V  : (nh,)   Lf h along f
    h_LG_V  : (nh, nu)  LG h (one row per constraint)
    h_alphah: (nh,)   alpha * (-V)  (the CBF decay term)
    """
    nh = h_V.shape[0]
    assert hx_Vx.shape[0] == nh

    h_Lf_V  = jnp.sum(hx_Vx * f, axis=-1)          # (nh,)
    h_LG_V  = hx_Vx @ G                              # (nh, nu)

    if isinstance(alpha, float) or jnp.asarray(alpha).ndim == 0:
        h_alpha = jnp.full((nh,), alpha)
    else:
        h_alpha = jnp.asarray(alpha)
        assert h_alpha.shape == (nh,)

    h_alphah = h_alpha * (h_V)                       # (nh,)
    return h_Lf_V, h_LG_V, h_alphah


def _check_and_promote(h_V, hx_Vx):
    """Promote scalar/1-D inputs to (nh, ...) shapes."""
    if h_V.ndim == 0:
        h_V = h_V[None]
    if hx_Vx.ndim == 1:
        hx_Vx = hx_Vx[None]
    assert h_V.shape[0] == hx_Vx.shape[0]
    return h_V, hx_Vx


# ── r-cbf ────────────────────────────────────────────────────────────────────

def rcbf_qp_mats(
    alpha,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    rho_scale: float = 0.25,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    """
    Robust CBF QP with fixed margin per constraint:

        rho_i = rho_scale * |LG_i| * (|LG_i| + 1)

    where |LG_i| is the 2-norm of the i-th row of the LG matrix.
    One slack variable r shared across all nh constraints.

    Variables: [u (nu), r (1)]
    Constraints (one row per h):
        Lf_i + LG_i @ u + alphah_i - rho_i  >=  -r
    """
    nx, nu = G.shape
    dtype = h_V.dtype
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    nh = h_V.shape[0]

    h_Lf_V, h_LG_V, h_alphah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    # h_LG_V: (nh, nu) — per-constraint robustness margin uses its row-norm.
    h_LG_norm = jnp.linalg.norm(h_LG_V, axis=-1)            # (nh,)
    h_rho = rho_scale * h_LG_norm * (h_LG_norm + 1.0)       # (nh,)

    # Cost: 0.5*||u - u_nom||^2 + 0.5*penalty*(r + relax_eps2)^2
    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])

    # Constraint matrix C @ [u, r] >= b
    # Row i:  LG_i @ u + r >= -(Lf_i + alphah_i - rho_i)
    h_rhs = h_Lf_V + h_alphah + h_rho                        # (nh,)
    # C: (nh, nu+1)
    C = jnp.concatenate([h_LG_V, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs                                               # (nh,)

    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9], dtype=dtype)])

    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)


def rcbf(
    alpha,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    rho_scale: float = 0.25,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings: JaxProxQP.Settings = None,
) -> tuple[Control, FloatScalar, QPSolution]:
    nx, nu = G.shape
    qp = rcbf_qp_mats(
        alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
        rho_scale, penalty, relax_eps1, relax_eps2,
    )
    if settings is None:
        settings = JaxProxQP.Settings.default()
    sol = JaxProxQP(qp, settings).solve()
    assert sol.x.shape == (nu + 1,)
    return sol.x[:nu], sol.x[-1], sol


# ── r-cbf-qp ─────────────────────────────────────────────────────────────────

def rcbf_qp_linear_mats(
    alpha,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    gamma1: float | jnp.ndarray,   # (nh,) or scalar — from gamma optimization
    gamma2: float | jnp.ndarray,   # (nh,) or scalar — from gamma optimization
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    """
    R-CBF-QP constraint per component i:

        Lf_i + LG_i @ u + alphah_i
          - gamma1_i * ||LG_i||
          - gamma2_i^2 * ||LG_i||^2  >= -r

    gamma1, gamma2 are pre-computed by minimizing max(sigma_hat - gamma1, 0) / (2*gamma2)
    where sigma_hat = max_{x in state_bounds} ||u_nom(x) - u_nom(x_hat)||.
    """
    nx, nu = G.shape
    dtype = h_V.dtype
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    nh = h_V.shape[0]

    h_Lf_V, h_LG_V, h_alphah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    h_LG_norm  = jnp.linalg.norm(h_LG_V, axis=-1)          # (nh,)
    h_LG_norm2 = h_LG_norm ** 2                              # (nh,)

    gamma1 = jnp.broadcast_to(jnp.asarray(gamma1, dtype=dtype), (nh,))
    gamma2 = jnp.broadcast_to(jnp.asarray(gamma2, dtype=dtype), (nh,))

    # Fixed robustness margin (no u-dependent terms — pure offset)
    h_rho = gamma1 * h_LG_norm + gamma2**2 * h_LG_norm2     # (nh,)

    # Cost
    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])

    # Constraint: LG_i @ u + r >= -(Lf_i + alphah_i - rho_i)
    h_rhs = h_Lf_V + h_alphah + h_rho
    C = jnp.concatenate([h_LG_V, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs

    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9],         dtype=dtype)])

    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)


def rcbf_qp_linear(
    alpha,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    gamma1: float | jnp.ndarray,
    gamma2: float | jnp.ndarray,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings: JaxProxQP.Settings = None,
) -> tuple[Control, FloatScalar, QPSolution]:
    nx, nu = G.shape
    qp = rcbf_qp_linear_mats(
        alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
        gamma1, gamma2, penalty, relax_eps1, relax_eps2,
    )
    if settings is None:
        settings = JaxProxQP.Settings.default()
    sol = JaxProxQP(qp, settings).solve()
    assert sol.x.shape == (nu + 1,)
    return sol.x[:nu], sol.x[-1], sol


def compute_rcbf_gammas(
    sigma_hat: float,
    gamma_min: float = 1e-4,
    gamma_max: float = 4.0,
    n_grid: int = 400,
) -> tuple[float, float]:
    """
    Solve:  min_{gamma1, gamma2 > 0}  max(sigma_hat - gamma1, 0) / (2 * gamma2)

    Returns (gamma1_star, gamma2_star).

    For a fixed sigma_hat, the optimal gamma1 = sigma_hat (absorbs all the
    control deviation), and gamma2 -> 0+ minimizes the objective to 0.
    The grid search finds the best finite discretization of this.
    """
    gamma_vals = jnp.linspace(gamma_min, gamma_max, n_grid)
    gamma_x, gamma_y = jnp.meshgrid(gamma_vals, gamma_vals, indexing="ij")
    objective_grid = jnp.maximum(sigma_hat - gamma_x, 0.0) / (2.0 * gamma_y)
    argmin_idx = jnp.unravel_index(jnp.argmin(objective_grid), objective_grid.shape)
    gamma1_star = gamma_vals[argmin_idx[0]].astype(jnp.float32)
    gamma2_star = gamma_vals[argmin_idx[1]].astype(jnp.float32)
    return gamma1_star, gamma2_star