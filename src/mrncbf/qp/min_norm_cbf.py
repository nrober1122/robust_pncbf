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
 
    # FIX: subtract h_rho
    h_rhs = h_Lf_V + h_alphah - h_rho
 
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
 
    # FIX: subtract h_rho (tightens constraint in negative-safe convention)
    h_rhs = h_Lf_V + h_alphah - h_rho
 
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


def compute_rcbf_gammas_(
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


def compute_rcbf_gammas(
    # --- Nominal CBF ingredients ---
    alpha,
    u_lb,               # (nu,)
    u_ub,               # (nu,)
    h_V_nom,            # (nh,)
    hx_Vx_nom,          # (nh, nx)
    f_nom,              # (nx,)
    G_nom,              # (nx, nu)
    u_nom_base,         # (nu,) — nominal policy at estimated state
    # --- Perturbed CBF ingredients (pre-computed at sampled states) ---
    h_V_pert,           # (N, nh)
    hx_Vx_pert,         # (N, nh, nx)
    f_pert,             # (N, nx)
    G_pert,             # (N, nx, nu)
    u_nom_pert,         # (N, nu) — nominal policy at each perturbed state
    # --- Grid parameters ---
    gamma_min: float = 1e-3,
    gamma_max: float = 4.0,
    n_grid: int = 50,
    reg_weight: float = 0.0,
) -> tuple:
    """
    Online gamma adaptation matching the paper (sans temporal smoothing).
 
    Two phases:
      Phase 1 — Compute sigma_hat once using the vanilla CBF (gamma=0):
        Solve the closed-form CBF correction at the nominal state and at each
        perturbed state *without* any robustness margin. sigma_hat is the max
        deviation of the filtered control across perturbed states.
        This measures the actual sensitivity to state uncertainty.
 
      Phase 2 — Grid search over (gamma1, gamma2):
        cost = max(sigma_hat - gamma1, 0) / (2*gamma2)
               + reg_weight * (gamma1 + gamma2)
        Reject infeasible gamma pairs (robustified QP exceeds control bounds).
 
    Returns:
        (gamma1_star, gamma2_star)
    """
    nx, nu = G_nom.shape
    h_V_nom, hx_Vx_nom = _check_and_promote(h_V_nom, hx_Vx_nom)
    nh = h_V_nom.shape[0]
    N = h_V_pert.shape[0]
 
    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Compute sigma_hat at gamma=0 (vanilla CBF correction)
    # ══════════════════════════════════════════════════════════════════════
 
    # ── Nominal vanilla CBF correction ───────────────────────────────────
    Lf_n, LG_n, ah_n = _lie_derivs(h_V_nom, hx_Vx_nom, f_nom, G_nom, alpha)
    LG_norm_n = jnp.linalg.norm(LG_n, axis=-1)        # (nh,)
    LG_nsq_n  = LG_norm_n ** 2                          # (nh,)
 
    # Vanilla constraint (rho=0): Lg@u >= -(Lf + alpha*h)
    rhs_n_vanilla = -(Lf_n + ah_n)                      # (nh,)
    proj_n = jnp.sum(LG_n * u_nom_base[None, :], axis=-1)  # (nh,)
    deficit_n_vanilla = rhs_n_vanilla - proj_n           # (nh,)
    gain_n_vanilla = jnp.maximum(deficit_n_vanilla, 0.0) / (LG_nsq_n + 1e-12)
 
    max_gain_n_vanilla = jnp.max(gain_n_vanilla)         # scalar
    active_n_vanilla = jnp.argmax(deficit_n_vanilla)      # scalar
    Lg_dir_n_vanilla = LG_n[active_n_vanilla]             # (nu,)
 
    U_nom_vanilla = u_nom_base + max_gain_n_vanilla * Lg_dir_n_vanilla  # (nu,)
    U_nom_vanilla_c = jnp.clip(U_nom_vanilla, u_lb, u_ub)
 
    # ── Perturbed vanilla CBF corrections ────────────────────────────────
    Lf_p = jnp.sum(hx_Vx_pert * f_pert[:, None, :], axis=-1)   # (N, nh)
    LG_p = jnp.einsum('Nhi,Niu->Nhu', hx_Vx_pert, G_pert)     # (N, nh, nu)
 
    if isinstance(alpha, float) or jnp.asarray(alpha).ndim == 0:
        h_alpha = jnp.full((nh,), alpha)
    else:
        h_alpha = jnp.asarray(alpha)
    ah_p = h_alpha[None, :] * h_V_pert                           # (N, nh)
 
    LG_norm_p = jnp.linalg.norm(LG_p, axis=-1)                  # (N, nh)
    LG_nsq_p  = LG_norm_p ** 2                                    # (N, nh)
    proj_p = jnp.sum(LG_p * u_nom_pert[:, None, :], axis=-1)    # (N, nh)
 
    # Vanilla deficit at each perturbed state (rho=0)
    deficit_p_vanilla = -(Lf_p + ah_p) - proj_p                  # (N, nh)
    gain_p_vanilla = jnp.maximum(deficit_p_vanilla, 0.0) / (LG_nsq_p + 1e-12)
    max_gain_p_vanilla = jnp.max(gain_p_vanilla, axis=-1)        # (N,)
    active_p_vanilla = jnp.argmax(
        jnp.maximum(deficit_p_vanilla, 0.0), axis=-1
    )                                                              # (N,)
    Lg_dir_p_vanilla = LG_p[jnp.arange(N), active_p_vanilla, :]  # (N, nu)
 
    U_pert_vanilla = (u_nom_pert
                      + max_gain_p_vanilla[:, None] * Lg_dir_p_vanilla)  # (N, nu)
    U_pert_vanilla_c = jnp.clip(U_pert_vanilla, u_lb, u_ub)
 
    # sigma_hat: max control deviation under vanilla CBF across perturbations
    sigma_hat = jnp.max(
        jnp.linalg.norm(U_pert_vanilla_c - U_nom_vanilla_c[None, :], axis=-1)
    )  # scalar
 
    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Grid search over (gamma1, gamma2) using sigma_hat
    # ══════════════════════════════════════════════════════════════════════
 
    gamma_vals = jnp.linspace(gamma_min, gamma_max, n_grid)
    g1_grid, g2_grid = jnp.meshgrid(gamma_vals, gamma_vals, indexing="ij")
    g1 = g1_grid.ravel()    # (n_gamma,)
    g2 = g2_grid.ravel()    # (n_gamma,)
    n_gamma = g1.size
 
    # ── Feasibility check: can the robustified QP be satisfied? ──────────
    # For each gamma pair, the rho margin at the nominal state is:
    rho_n = g1[:, None] * LG_norm_n[None, :] + g2[:, None]**2 * LG_nsq_n[None, :]
    rhs_n_robust = -(Lf_n[None, :] + ah_n[None, :]) + rho_n      # (n_gamma, nh)
    deficit_n_robust = rhs_n_robust - proj_n[None, :]              # (n_gamma, nh)
    gain_n_robust = jnp.maximum(deficit_n_robust, 0.0) / (LG_nsq_n[None, :] + 1e-12)
 
    max_gain_n_robust = jnp.max(gain_n_robust, axis=-1)           # (n_gamma,)
    active_n_robust = jnp.argmax(deficit_n_robust, axis=-1)       # (n_gamma,)
    Lg_dir_n_robust = LG_n[active_n_robust]                       # (n_gamma, nu)
 
    U_nom_robust = u_nom_base[None, :] + max_gain_n_robust[:, None] * Lg_dir_n_robust
    feas_nom = (jnp.all(U_nom_robust >= u_lb[None, :] - 1e-6, axis=-1) &
                jnp.all(U_nom_robust <= u_ub[None, :] + 1e-6, axis=-1))
 
    # ── Cost: sigma_hat is a constant w.r.t. the grid ────────────────────
    cost = (jnp.maximum(sigma_hat - g1, 0.0) / (2.0 * g2 + 1e-12)
            + reg_weight * (g1 + g2))
 
    cost = jnp.where(feas_nom, cost, 1e6)
 
    best = jnp.argmin(cost)
    return g1[best].astype(jnp.float32), g2[best].astype(jnp.float32)

# def compute_rcbf_gammas(
#     # --- Nominal CBF ingredients ---
#     alpha,
#     u_lb,               # (nu,)
#     u_ub,               # (nu,)
#     h_V_nom,            # (nh,)
#     hx_Vx_nom,          # (nh, nx)
#     f_nom,              # (nx,)
#     G_nom,              # (nx, nu)
#     u_nom_base,         # (nu,) — nominal policy at estimated state
#     # --- Perturbed CBF ingredients (pre-computed at sampled states) ---
#     h_V_pert,           # (N, nh)
#     hx_Vx_pert,         # (N, nh, nx)
#     f_pert,             # (N, nx)
#     G_pert,             # (N, nx, nu)
#     u_nom_pert,         # (N, nu) — nominal policy at each perturbed state
#     # --- Grid parameters ---
#     gamma_min: float = 1e-3,
#     gamma_max: float = 4.0,
#     n_grid: int = 400,
#     reg_weight: float = 0.1,
# ) -> tuple:
#     """
#     Online gamma adaptation matching the paper (sans temporal smoothing).
 
#     For each (gamma1, gamma2) on a grid:
#       1. Closed-form R-CBF-QP correction at nominal state -> U_nom
#       2. Closed-form R-CBF-QP correction at each perturbed state -> U_pert
#       3. sigma = max_i ||U_pert_i - U_nom||
#       4. cost = max(sigma - gamma1, 0) / (2*gamma2)
#                + reg_weight * (gamma1 + gamma2)
#       5. Reject infeasible gamma pairs (controls exceed bounds pre-clip)
#       6. Return argmin
 
#     Returns:
#         (gamma1_star, gamma2_star)
#     """
#     nx, nu = G_nom.shape
#     h_V_nom, hx_Vx_nom = _check_and_promote(h_V_nom, hx_Vx_nom)
#     nh = h_V_nom.shape[0]
#     N = h_V_pert.shape[0]
 
#     gamma_vals = jnp.linspace(gamma_min, gamma_max, n_grid)
#     g1_grid, g2_grid = jnp.meshgrid(gamma_vals, gamma_vals, indexing="ij")
#     g1 = g1_grid.ravel()    # (n_gamma,)
#     g2 = g2_grid.ravel()    # (n_gamma,)
#     n_gamma = g1.size
 
#     # ── Nominal Lie derivatives ──────────────────────────────────────────
#     Lf_n, LG_n, ah_n = _lie_derivs(h_V_nom, hx_Vx_nom, f_nom, G_nom, alpha)
#     LG_norm_n  = jnp.linalg.norm(LG_n, axis=-1)       # (nh,)
#     LG_nsq_n   = LG_norm_n ** 2                         # (nh,)
 
#     # ── Nominal closed-form correction per gamma pair ────────────────────
#     # rho[g,h] = g1[g]*||LG_n[h]|| + g2[g]^2*||LG_n[h]||^2
#     rho_n = (g1[:, None] * LG_norm_n[None, :]
#              + g2[:, None]**2 * LG_nsq_n[None, :])              # (n_gamma, nh)
 
#     # With sign-fixed constraint: Lg@u >= -(Lf + alpha*h) + rho
#     rhs_n = -(Lf_n[None, :] + ah_n[None, :]) + rho_n            # (n_gamma, nh)
 
#     proj_n = jnp.sum(LG_n * u_nom_base[None, :], axis=-1)       # (nh,)
#     deficit_n = rhs_n - proj_n[None, :]                           # (n_gamma, nh)
#     gain_n = jnp.maximum(deficit_n, 0.0) / (LG_nsq_n[None, :] + 1e-12)
 
#     # Most binding constraint determines correction direction
#     max_gain_n = jnp.max(gain_n, axis=-1)                        # (n_gamma,)
#     active_n = jnp.argmax(deficit_n, axis=-1)                    # (n_gamma,)
#     Lg_dir_n = LG_n[active_n]                                    # (n_gamma, nu)
 
#     U_nom = u_nom_base[None, :] + max_gain_n[:, None] * Lg_dir_n
 
#     feas_nom = (jnp.all(U_nom >= u_lb[None, :] - 1e-6, axis=-1) &
#                 jnp.all(U_nom <= u_ub[None, :] + 1e-6, axis=-1))
#     U_nom_c = jnp.clip(U_nom, u_lb, u_ub)
 
#     # ── Perturbed Lie derivatives (vectorized over N) ────────────────────
#     Lf_p = jnp.sum(hx_Vx_pert * f_pert[:, None, :], axis=-1)   # (N, nh)
#     LG_p = jnp.einsum('Nhi,Niu->Nhu', hx_Vx_pert, G_pert)     # (N, nh, nu)
 
#     if isinstance(alpha, float) or jnp.asarray(alpha).ndim == 0:
#         h_alpha = jnp.full((nh,), alpha)
#     else:
#         h_alpha = jnp.asarray(alpha)
#     ah_p = h_alpha[None, :] * h_V_pert                           # (N, nh)
 
#     LG_norm_p = jnp.linalg.norm(LG_p, axis=-1)                  # (N, nh)
#     LG_nsq_p  = LG_norm_p ** 2                                    # (N, nh)
 
#     proj_p = jnp.sum(LG_p * u_nom_pert[:, None, :], axis=-1)    # (N, nh)
 
#     # base_deficit is the gamma-independent part: -(Lf + alpha*h) - proj
#     base_def_p = -(Lf_p + ah_p) - proj_p                         # (N, nh)
 
#     # Full deficit: base + gamma-dependent rho terms
#     deficit_p = (base_def_p[None, :, :]
#                  + g1[:, None, None] * LG_norm_p[None, :, :]
#                  + g2[:, None, None]**2 * LG_nsq_p[None, :, :])  # (n_gamma, N, nh)
 
#     gain_p = jnp.maximum(deficit_p, 0.0) / (LG_nsq_p[None, :, :] + 1e-12)
#     max_gain_p = jnp.max(gain_p, axis=-1)                        # (n_gamma, N)
 
#     active_p = jnp.argmax(jnp.maximum(deficit_p, 0.0), axis=-1) # (n_gamma, N)
#     Lg_dir_p = LG_p[jnp.arange(N)[None, :], active_p, :]        # (n_gamma, N, nu)
 
#     U_pert = u_nom_pert[None, :, :] + max_gain_p[:, :, None] * Lg_dir_p
 
#     feas_p = (jnp.all(U_pert >= u_lb[None, None, :] - 1e-6, axis=-1) &
#               jnp.all(U_pert <= u_ub[None, None, :] + 1e-6, axis=-1))
#     feas_all_p = jnp.all(feas_p, axis=-1)                        # (n_gamma,)
 
#     U_pert_c = jnp.clip(U_pert, u_lb, u_ub)
 
#     # ── Sigma: max filtered-control deviation ────────────────────────────
#     sigma = jnp.max(
#         jnp.linalg.norm(U_pert_c - U_nom_c[:, None, :], axis=-1),
#         axis=-1,
#     )  # (n_gamma,)
 
#     # ── Cost ─────────────────────────────────────────────────────────────
#     cost = (jnp.maximum(sigma - g1, 0.0) / (2.0 * g2 + 1e-12)
#             + reg_weight * (g1 + g2))
 
#     cost = jnp.where(feas_nom & feas_all_p, cost, 1e6)
 
#     best = jnp.argmin(cost)
#     return g1[best].astype(jnp.float32), g2[best].astype(jnp.float32)


# def compute_rcbf_gammas_from_ingredients(
#     # --- Nominal CBF ingredients ---
#     alpha,
#     u_lb,               # (nu,)
#     u_ub,               # (nu,)
#     h_V_nom,            # (nh,)
#     hx_Vx_nom,          # (nh, nx)
#     f_nom,              # (nx,)
#     G_nom,              # (nx, nu)
#     u_nom_base,         # (nu,) — nominal policy output at estimated state
#     # --- Perturbed CBF ingredients (pre-computed at sampled states) ---
#     h_V_pert,           # (N, nh)
#     hx_Vx_pert,         # (N, nh, nx)
#     f_pert,             # (N, nx)
#     G_pert,             # (N, nx, nu)
#     u_nom_pert,         # (N, nu) — nominal policy output at each perturbed state
#     # --- Previous gamma values (for temporal smoothing) ---
#     gamma_prev: RCBFGammaState = RCBFGammaState(),
#     # --- Grid parameters ---
#     gamma_min: float = 1e-3,
#     gamma_max: float = 4.0,
#     n_grid: int = 400,
#     # --- Cost weights (from the paper's reference code) ---
#     reg_weight: float = 0.1,
#     smooth_weight: float = 0.05,
# ) -> tuple[float, float, RCBFGammaState]:
#     """
#     Online gamma adaptation from pre-computed CBF ingredients.
 
#     This matches the paper's Algorithm 1 / reference implementation:
#     for each (gamma1, gamma2) on a grid:
#       1. Compute the R-CBF-QP filtered control at the nominal state
#       2. Compute the R-CBF-QP filtered control at each perturbed state
#       3. sigma_hat = max_i ||u_pert_i - u_nom||
#       4. cost = max(sigma_hat - gamma1, 0)/(2*gamma2)
#                + reg*(gamma1 + gamma2)
#                + smooth*||(gamma1,gamma2) - prev||^2
#       5. Reject infeasible gamma pairs
#       6. Pick argmin
#     """
#     nx, nu = G_nom.shape
#     h_V_nom_p, hx_Vx_nom_p = _check_and_promote(h_V_nom, hx_Vx_nom)
#     nh = h_V_nom_p.shape[0]
#     N = h_V_pert.shape[0]
 
#     gamma_vals = jnp.linspace(gamma_min, gamma_max, n_grid)
#     gamma1_grid, gamma2_grid = jnp.meshgrid(gamma_vals, gamma_vals, indexing="ij")
#     g1_flat = gamma1_grid.ravel()
#     g2_flat = gamma2_grid.ravel()
#     n_gamma = g1_flat.size
 
#     # ── Nominal Lie derivatives ──────────────────────────────────────────────
#     Lf_nom, LG_nom, alphah_nom = _lie_derivs(h_V_nom_p, hx_Vx_nom_p,
#                                               f_nom, G_nom, alpha)
#     # Lf_nom: (nh,),  LG_nom: (nh, nu),  alphah_nom: (nh,)
#     LG_norm_nom = jnp.linalg.norm(LG_nom, axis=-1)         # (nh,)
#     LG_normsq_nom = LG_norm_nom ** 2                         # (nh,)
 
#     # ── Nominal rho for each gamma pair ──────────────────────────────────────
#     # rho = gamma1 * ||LG|| + gamma2^2 * ||LG||^2
#     rho_nom = (g1_flat[:, None] * LG_norm_nom[None, :]
#                + (g2_flat[:, None]**2) * LG_normsq_nom[None, :])  # (n_gamma, nh)
 
#     # ── Nominal RHS: with sign fix, constraint is Lg@u >= -Lf - alpha*h + rho
#     rhs_nom = -(Lf_nom[None, :] + alphah_nom[None, :]) + rho_nom  # (n_gamma, nh)
 
#     # ── Nominal closed-form correction ───────────────────────────────────────
#     # proj = Lg @ u_base
#     proj_nom = jnp.sum(LG_nom * u_nom_base[None, :], axis=-1)  # (nh,)
#     LG_normsq_safe = LG_normsq_nom + 1e-12
 
#     # Per-gamma, per-constraint deficit and gain
#     deficit_nom = rhs_nom - proj_nom[None, :]                    # (n_gamma, nh)
#     gain_per_h = jnp.maximum(deficit_nom, 0.0) / LG_normsq_safe[None, :]  # (n_gamma, nh)
 
#     # For each gamma pair, apply the correction from the most binding constraint.
#     # (For nh=1 this is exact; for nh>1 it's the paper's approximation.)
#     max_gain_nom = jnp.max(gain_per_h, axis=-1)                 # (n_gamma,)
#     most_active_nom = jnp.argmax(deficit_nom, axis=-1)          # (n_gamma,)
#     Lg_dir_nom = LG_nom[most_active_nom]                        # (n_gamma, nu)
 
#     U_nom = u_nom_base[None, :] + max_gain_nom[:, None] * Lg_dir_nom  # (n_gamma, nu)
 
#     # Feasibility check (before clipping)
#     feasible_nom = (jnp.all(U_nom >= u_lb[None, :] - 1e-6, axis=-1) &
#                     jnp.all(U_nom <= u_ub[None, :] + 1e-6, axis=-1))
 
#     U_nom_clipped = jnp.clip(U_nom, u_lb, u_ub)
 
#     # ── Perturbed Lie derivatives (vectorized over N samples) ────────────────
#     # h_V_pert: (N, nh), hx_Vx_pert: (N, nh, nx), f_pert: (N, nx), G_pert: (N, nx, nu)
 
#     # Lf_pert[i] = sum(hx_Vx_pert[i] * f_pert[i], axis=-1)  -> (N, nh)
#     Lf_pert = jnp.sum(hx_Vx_pert * f_pert[:, None, :], axis=-1)    # (N, nh)
#     # LG_pert[i] = hx_Vx_pert[i] @ G_pert[i]  -> (N, nh, nu)
#     LG_pert = jnp.einsum('Nhi,Niu->Nhu', hx_Vx_pert, G_pert)      # (N, nh, nu)
 
#     if isinstance(alpha, float) or jnp.asarray(alpha).ndim == 0:
#         h_alpha_arr = jnp.full((nh,), alpha)
#     else:
#         h_alpha_arr = jnp.asarray(alpha)
#     alphah_pert = h_alpha_arr[None, :] * h_V_pert                   # (N, nh)
 
#     LG_norm_pert = jnp.linalg.norm(LG_pert, axis=-1)               # (N, nh)
#     LG_normsq_pert = LG_norm_pert ** 2                               # (N, nh)
 
#     # ── Perturbed rho: (n_gamma, N, nh) — but this is huge. ─────────────────
#     # Instead, compute per-sample, per-gamma in a memory-friendly way.
#     # rho_pert[g, i, h] = g1[g]*||LG[i,h]|| + g2[g]^2 * ||LG[i,h]||^2
#     # rhs_pert[g, i, h] = -(Lf[i,h] + alphah[i,h]) + rho[g,i,h]
 
#     # proj_pert[i, h] = LG_pert[i, h, :] @ u_nom_pert[i, :]
#     proj_pert = jnp.sum(LG_pert * u_nom_pert[:, None, :], axis=-1)  # (N, nh)
#     LG_normsq_pert_safe = LG_normsq_pert + 1e-12                    # (N, nh)
 
#     # We need: for each gamma pair g, for each sample i:
#     #   rhs[g,i,h] = -(Lf[i,h] + alphah[i,h]) + g1[g]*||LG[i,h]|| + g2[g]^2*||LG[i,h]||^2
#     #   deficit[g,i,h] = rhs[g,i,h] - proj[i,h]
#     #   gain[g,i,h] = max(deficit, 0) / ||LG[i,h]||^2
#     #   max_gain[g,i] = max_h gain[g,i,h]
#     #
#     # Then u_corr[g,i] = u_base[i] + max_gain[g,i] * Lg_dir[i]
#     # sigma[g] = max_i ||u_corr[g,i] - U_nom_clipped[g]||
 
#     # For memory, compute in chunks or use einsum tricks.
#     # With n_grid=400, n_gamma=160000. N=100. This is manageable if nh is small.
 
#     # Base terms (gamma-independent): -(Lf + alphah) - proj
#     base_deficit = -(Lf_pert + alphah_pert) - proj_pert  # (N, nh)
 
#     # Gamma-dependent addition to deficit:
#     # g1[g]*||LG[i,h]|| + g2[g]^2*||LG[i,h]||^2
#     # Shape: (n_gamma,) x (N, nh) -> we need (n_gamma, N, nh)
#     # deficit[g,i,h] = base_deficit[i,h] + g1[g]*LG_norm[i,h] + g2[g]^2*LG_normsq[i,h]
 
#     # To avoid (160000, 100, nh) arrays, process in batches over gamma.
#     # But for n_grid=400, 160k * 100 * 1 floats = 64MB, manageable.
#     # For larger grids or nh, consider chunking.
 
#     deficit_pert = (base_deficit[None, :, :]
#                     + g1_flat[:, None, None] * LG_norm_pert[None, :, :]
#                     + (g2_flat[:, None, None]**2) * LG_normsq_pert[None, :, :])  # (n_gamma, N, nh)
 
#     gain_pert = jnp.maximum(deficit_pert, 0.0) / LG_normsq_pert_safe[None, :, :]  # (n_gamma, N, nh)
#     max_gain_pert = jnp.max(gain_pert, axis=-1)                                    # (n_gamma, N)
 
#     # Correction direction: most active constraint per sample
#     most_active_pert = jnp.argmax(
#         jnp.maximum(deficit_pert, 0.0), axis=-1
#     )  # (n_gamma, N)
 
#     # LG_pert: (N, nh, nu). We need Lg_dir[i] = LG_pert[i, most_active[g,i], :]
#     # For simplicity with single constraint (nh=1), this is just LG_pert[:, 0, :]
#     # For multi-constraint, index properly:
#     Lg_dir_pert = LG_pert[
#         jnp.arange(N)[None, :],
#         most_active_pert,
#         :
#     ]  # (n_gamma, N, nu)
 
#     U_pert = (u_nom_pert[None, :, :]
#               + max_gain_pert[:, :, None] * Lg_dir_pert)  # (n_gamma, N, nu)
 
#     # Feasibility of perturbed solutions
#     feasible_pert = (jnp.all(U_pert >= u_lb[None, None, :] - 1e-6, axis=-1) &
#                      jnp.all(U_pert <= u_ub[None, None, :] + 1e-6, axis=-1))  # (n_gamma, N)
#     feasible_all_pert = jnp.all(feasible_pert, axis=-1)  # (n_gamma,)
 
#     # Clip perturbed controls
#     U_pert_clipped = jnp.clip(U_pert, u_lb, u_ub)  # (n_gamma, N, nu)
 
#     # ── Sigma: max control deviation across perturbed samples ────────────────
#     control_diff = U_pert_clipped - U_nom_clipped[:, None, :]  # (n_gamma, N, nu)
#     sigma_vals = jnp.max(
#         jnp.linalg.norm(control_diff, axis=-1),   # (n_gamma, N)
#         axis=-1                                     # (n_gamma,)
#     )
 
#     # ── Cost (matching paper's reference code) ───────────────────────────────
#     # Term 1: Residual error bound
#     cost = jnp.maximum(sigma_vals - g1_flat, 0.0) / (2.0 * g2_flat + 1e-12)
 
#     # Term 2: Regularization (discourages large gammas = less conservatism)
#     cost = cost + reg_weight * (g1_flat + g2_flat)
 
#     # Term 3: Temporal smoothing (discourages jumps from previous values)
#     cost = cost + smooth_weight * (
#         (g1_flat - gamma_prev.gamma1)**2 + (g2_flat - gamma_prev.gamma2)**2
#     )
 
#     # ── Feasibility filter ───────────────────────────────────────────────────
#     overall_feasible = feasible_nom & feasible_all_pert
#     cost = jnp.where(overall_feasible, cost, 1e6)
 
#     # ── Select best ──────────────────────────────────────────────────────────
#     best_idx = jnp.argmin(cost)
#     gamma1_star = g1_flat[best_idx].astype(jnp.float32)
#     gamma2_star = g2_flat[best_idx].astype(jnp.float32)
 
#     new_state = RCBFGammaState(
#         gamma1=float(gamma1_star),
#         gamma2=float(gamma2_star),
#     )
 
#     return gamma1_star, gamma2_star, new_state

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