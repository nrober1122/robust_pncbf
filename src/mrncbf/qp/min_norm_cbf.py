import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP, QPSolution

from mrncbf.dyn.dyn_types import Control, HFloat, HState, State
from mrncbf.solvers.qp import get_relaxed_constr_Gh, jaxopt_osqp
from mrncbf.utils.jax_types import FloatScalar
from mrncbf.utils.jax_utils import jax_vmap


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
    alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
    rho_scale: float = 0.25,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
):
    nx, nu = G.shape
    dtype = h_V.dtype
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    nh = h_V.shape[0]
 
    h_Lf_V, h_LG_V, h_alphah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    h_LG_norm = jnp.linalg.norm(h_LG_V, axis=-1)
    h_rho = rho_scale * h_LG_norm * (h_LG_norm + 1.0)
 
    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])
 
    # Add h_rho to tighten the constraint (negative-safe convention)
    h_rhs = h_Lf_V + h_alphah + h_rho

    C = jnp.concatenate([h_LG_V, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs

    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9], dtype=dtype)])

    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)


def rcbf(
    alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
    rho_scale: float = 0.25,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings=None,
):
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
    u_lb,u_ub,
    h_V, hx_Vx, f, G, u_nom,
    gamma1, gamma2,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
):
    nx, nu = G.shape
    dtype = h_V.dtype
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    nh = h_V.shape[0]
 
    h_Lf_V, h_LG_V, h_alphah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    h_LG_norm  = jnp.linalg.norm(h_LG_V, axis=-1)
    h_LG_norm2 = h_LG_norm ** 2
 
    gamma1 = jnp.broadcast_to(jnp.asarray(gamma1, dtype=dtype), (nh,))
    gamma2 = jnp.broadcast_to(jnp.asarray(gamma2, dtype=dtype), (nh,))
 
    h_rho = gamma1 * h_LG_norm + gamma2**2 * h_LG_norm2
 
    # Add h_rho to tighten the constraint (negative-safe convention)
    h_rhs = h_Lf_V + h_alphah + h_rho
 
    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])
 
    C = jnp.concatenate([h_LG_V, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs
 
    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9],         dtype=dtype)])
 
    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)


def rcbf_qp_linear(
    alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
    gamma1, gamma2,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings=None,
):
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


# ── accurate per-gamma sigma_hat ──────────────────────────────────────────────

def _rcbf_closed_form_control(
    h_V, hx_Vx, f, G, u_nom, alpha, u_lb, u_ub, gamma1, gamma2, tol: float = 1e-6,
):
    """
    Closed-form min-norm R-CBF-filtered control for a single state (negative-safe).

    Returns (u_clipped, feasible) where `feasible` is True iff the unclipped
    projection onto the tightened halfspace already lies within [u_lb, u_ub].
    The paper rejects gamma pairs that would force the QP to leave the box;
    we expose the same flag so the gamma optimizer can filter them out.
    """
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    Lf_V, LG_V, ah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    LG_norm = jnp.linalg.norm(LG_V, axis=-1)
    LG_nsq  = LG_norm ** 2
    rho     = gamma1 * LG_norm + gamma2 ** 2 * LG_nsq
    rhs     = -(Lf_V + ah) - rho
    proj    = jnp.sum(LG_V * u_nom[None, :], axis=-1)
    deficit = proj - rhs
    gain    = jnp.maximum(deficit, 0.0) / (LG_nsq + 1e-12)
    active  = jnp.argmax(deficit)
    u_star  = u_nom - gain[active] * LG_V[active]
    feasible = jnp.all((u_star >= u_lb - tol) & (u_star <= u_ub + tol))
    return jnp.clip(u_star, u_lb, u_ub), feasible


def _compute_rcbf_gammas_core(
    alpha, u_lb, u_ub,
    h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
    h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
    *,
    use_feasibility: bool,
    reg_weight: float,
    smooth_weight: float,
    gamma1_prev,
    gamma2_prev,
    gamma_min: float,
    gamma_max: float,
    n_grid: int,
) -> tuple:
    """
    Shared core for the three gamma-selection variants A/B/C.

    All three minimize the paper's (Eq. 10) ratio cost
        max(sigma_hat(g1,g2) - g1, 0) / (2*g2)
    over a (gamma_min, gamma_max) grid of size n_grid^2; the variants differ
    in which of the reference-code extras are added on top.
    """
    h_V_nom, hx_Vx_nom = _check_and_promote(h_V_nom, hx_Vx_nom)

    gamma_vals = jnp.linspace(gamma_min, gamma_max, n_grid)
    g1, g2 = [a.ravel() for a in jnp.meshgrid(gamma_vals, gamma_vals, indexing="ij")]

    def sigma_and_feas(g1_i, g2_i):
        u_nom_filt, feas_n = _rcbf_closed_form_control(
            h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
            alpha, u_lb, u_ub, g1_i, g2_i,
        )
        u_pert_filt, feas_p = jax.vmap(
            lambda hV, hx, f_i, G_i, un: _rcbf_closed_form_control(
                hV, hx, f_i, G_i, un, alpha, u_lb, u_ub, g1_i, g2_i,
            )
        )(h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert)
        sigma = jnp.max(jnp.linalg.norm(u_pert_filt - u_nom_filt[None, :], axis=-1))
        return sigma, feas_n & jnp.all(feas_p)

    sigma_hat_grid, feas_grid = jax.vmap(sigma_and_feas)(g1, g2)

    cost = jnp.maximum(sigma_hat_grid - g1, 0.0) / (2.0 * g2 + 1e-12)
    if reg_weight != 0.0:
        cost = cost + reg_weight * (g1 + g2)
    if smooth_weight != 0.0:
        cost = cost + smooth_weight * (
            (g1 - gamma1_prev) ** 2 + (g2 - gamma2_prev) ** 2
        )
    if use_feasibility:
        cost = jnp.where(feas_grid, cost, 1e6)

    best = jnp.argmin(cost)
    return g1[best].astype(jnp.float32), g2[best].astype(jnp.float32)


def compute_rcbf_gammas_A(
    alpha, u_lb, u_ub,
    h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
    h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
    gamma_min: float = 1e-3,
    gamma_max: float = 4.0,
    n_grid: int = 100,
) -> tuple:
    """
    Option A — strict paper (arXiv 2508.19159, Eq. 10).

        argmin_{g1,g2 > 0}  max(sigma_hat(g1,g2) - g1, 0) / (2*g2)

    No feasibility filter, no regularization, no temporal smoothing. Note
    that Eq. 10 is degenerate over the cost-zero region {g1 >= sigma_hat};
    the bounded grid + argmin tie-breaking pin down a particular (g1, g2),
    but the choice is sensitive to grid resolution and ordering.
    """
    return _compute_rcbf_gammas_core(
        alpha, u_lb, u_ub,
        h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
        h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
        use_feasibility=False, reg_weight=0.0, smooth_weight=0.0,
        gamma1_prev=0.0, gamma2_prev=0.0,
        gamma_min=gamma_min, gamma_max=gamma_max, n_grid=n_grid,
    )


def compute_rcbf_gammas_B(
    alpha, u_lb, u_ub,
    h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
    h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
    gamma_min: float = 1e-3,
    gamma_max: float = 4.0,
    n_grid: int = 100,
) -> tuple:
    """
    Option B — paper Eq. 10 + feasibility filter only.

    Reject (g1, g2) pairs whose closed-form projection at the nominal or any
    perturbed state would lie outside [u_lb, u_ub]. The feasibility predicate
    is the precondition for sigma_hat being well-defined as a control-space
    distance, so this can be motivated as a sanity check on top of the paper
    rather than a new cost term.
    """
    return _compute_rcbf_gammas_core(
        alpha, u_lb, u_ub,
        h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
        h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
        use_feasibility=True, reg_weight=0.0, smooth_weight=0.0,
        gamma1_prev=0.0, gamma2_prev=0.0,
        gamma_min=gamma_min, gamma_max=gamma_max, n_grid=n_grid,
    )


def compute_rcbf_gammas_C(
    alpha, u_lb, u_ub,
    h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
    h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
    gamma1_prev: float = 0.0,
    gamma2_prev: float = 0.0,
    gamma_min: float = 1e-3,
    gamma_max: float = 8.0,
    n_grid: int = 100,
    reg_weight: float = 0.1,
    smooth_weight: float = 0.05,
) -> tuple:
    """
    Option C — reference-code-faithful (safety_filterRAL.py).

        cost = max(sigma_hat - g1, 0) / (2*g2)
             + reg_weight * (g1 + g2)
             + smooth_weight * ((g1 - g1_prev)^2 + (g2 - g2_prev)^2)
        + feasibility filter (reject (g1,g2) that exceed control bounds).

    Defaults match the paper repo: reg_weight=0.1, smooth_weight=0.05.
    Pass gamma{1,2}_prev from the previous timestep's selection to enable
    temporal smoothing; leaving them at 0.0 reduces the smoothing term to a
    plain L2 anchor at the origin.
    """
    return _compute_rcbf_gammas_core(
        alpha, u_lb, u_ub,
        h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom_base,
        h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
        use_feasibility=True,
        reg_weight=reg_weight, smooth_weight=smooth_weight,
        gamma1_prev=gamma1_prev, gamma2_prev=gamma2_prev,
        gamma_min=gamma_min, gamma_max=gamma_max, n_grid=n_grid,
    )


# Backward-compatible alias. Option B (paper + feasibility filter) is the
# closest match to the previous in-tree implementation. Switch to
# compute_rcbf_gammas_A for the strict paper formulation, or
# compute_rcbf_gammas_C for the reference-code-faithful one.
compute_rcbf_gammas = compute_rcbf_gammas_C


# ── guardian cbf (g-cbf) ──────────────────────────────────────────────────────

def gcbf_qp_mats(
    alpha,
    u_lb, u_ub,
    h_V_wc,     # (nh,)      per-constraint worst-case CBF values
    h_Lf_V_wc, # (nh,)      per-constraint worst-case Lf V  (largest drift term)
    h_LG_V_wc, # (nh, nu)   per-constraint worst-case LG V  (least control authority)
    u_nom,      # (nu,)
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    """
    CBF-QP assembled from independently chosen per-constraint worst-case terms.

    h_V, Lf V, and LG V are each selected from whichever sample is most
    conservative for that quantity, independently per constraint:
      - h_V_wc[i]    : sample with largest V_i    (smallest safety margin)
      - h_Lf_V_wc[i] : sample with largest Lf V_i (most dangerous passive drift)
      - h_LG_V_wc[i] : sample with smallest ||LG V_i|| (least control authority)
    """
    nh = h_V_wc.shape[0]
    nu = u_nom.shape[0]
    dtype = u_nom.dtype

    if isinstance(alpha, float) or jnp.asarray(alpha).ndim == 0:
        h_alphah = jnp.full((nh,), alpha) * h_V_wc
    else:
        h_alphah = jnp.asarray(alpha) * h_V_wc             # (nh,)

    h_rhs = h_Lf_V_wc + h_alphah                           # (nh,)

    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])
    C = jnp.concatenate([h_LG_V_wc, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs
    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9],         dtype=dtype)])
    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)


def gcbf_qp(
    alpha,
    u_lb, u_ub,
    h_V_wc, h_Lf_V_wc, h_LG_V_wc, u_nom,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings=None,
) -> tuple[Control, FloatScalar, QPSolution]:
    """Solve the per-constraint worst-case CBF-QP. Returns (u_opt, r, sol)."""
    nu = u_nom.shape[0]
    qp = gcbf_qp_mats(alpha, u_lb, u_ub, h_V_wc, h_Lf_V_wc, h_LG_V_wc, u_nom,
                      penalty, relax_eps1, relax_eps2)
    if settings is None:
        settings = JaxProxQP.Settings.default()
    sol = JaxProxQP(qp, settings).solve()
    assert sol.x.shape == (nu + 1,)
    return sol.x[:nu], sol.x[-1], sol


# ── learned rcbf ─────────────────────────────────────────────────────────────

def rcbf_qp_learned_mats(
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
    h_Delta: jnp.ndarray,          # (nh,) learned residual
    R_floor: float = 0.0,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    """
    Like rcbf_qp_linear_mats, but adds a learned residual per constraint.
 
    Per-constraint robustness margin:
        h_rho_i = max(R_floor, gamma1_i * ||LG_i|| + gamma2_i^2 * ||LG_i||^2 + Delta_i)
 
    When Delta_i < 0, the learned term REDUCES conservatism.
    When Delta_i > 0, it adds conservatism beyond the analytical baseline.
    R_floor prevents the total margin from going negative.
    """
    nx, nu = G.shape
    dtype = h_V.dtype
    h_V, hx_Vx = _check_and_promote(h_V, hx_Vx)
    nh = h_V.shape[0]
 
    h_Lf_V, h_LG_V, h_alphah = _lie_derivs(h_V, hx_Vx, f, G, alpha)
    h_LG_norm = jnp.linalg.norm(h_LG_V, axis=-1)            # (nh,)
    h_LG_norm2 = h_LG_norm ** 2
 
    gamma1 = jnp.broadcast_to(jnp.asarray(gamma1, dtype=dtype), (nh,))
    gamma2 = jnp.broadcast_to(jnp.asarray(gamma2, dtype=dtype), (nh,))
 
    # Analytical + learned, floored at R_floor
    h_rho_anal = gamma1 * h_LG_norm + gamma2**2 * h_LG_norm2
    h_rho = jnp.maximum(R_floor, h_rho_anal + h_Delta)       # (nh,)
 
    # Cost: 0.5 * ||u - u_nom||^2 + 0.5 * penalty * (r + eps2)^2
    H = np.eye(nu + 1, dtype=dtype)
    H[-1, -1] = penalty
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)])
 
    # Constraint: LG_i @ u + r >= -(Lf_i + alphah_i + rho_i)
    #   equivalently: [LG_i, -1] @ [u, r] >= -(Lf_i + alphah_i + rho_i)
    h_rhs = h_Lf_V + h_alphah + h_rho
    C = jnp.concatenate([h_LG_V, -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    b = -h_rhs
 
    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([1e9], dtype=dtype)])
 
    return JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
 
 
def rcbf_qp_learned(
    alpha,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    gamma1,
    gamma2,
    h_Delta: jnp.ndarray,
    R_floor: float = 0.0,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings: JaxProxQP.Settings = None,
) -> tuple[Control, FloatScalar, QPSolution]:
    """Solve the learned-robust CBF-QP. Returns (u_opt, r, sol)."""
    nx, nu = G.shape
    qp = rcbf_qp_learned_mats(
        alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
        gamma1, gamma2, h_Delta, R_floor,
        penalty, relax_eps1, relax_eps2,
    )
    if settings is None:
        settings = JaxProxQP.Settings.default()
    sol = JaxProxQP(qp, settings).solve()
    assert sol.x.shape == (nu + 1,)
    return sol.x[:nu], sol.x[-1], sol