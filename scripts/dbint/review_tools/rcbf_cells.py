RCBF_MD = r'''## R-CBF-QP gain selection when no gain pair is feasible

The compare notebook's R-CBF-QP chooses $(\gamma_1, \gamma_2)$ with `compute_rcbf_gammas` (option C: the paper's cost plus
regularization and a feasibility filter). A pair is rejected if the filtered control at any sampled state leaves the
input box. When every pair is rejected, all costs are $10^6$ and `argmin` returns the first grid entry, i.e. the
minimum gains, so the tightening drops to about zero.

`R-CBF-QP (orig)` reproduces the compare notebook exactly (same sampling keys). `R-CBF-QP (no-feas fallback)` differs
only when no pair is feasible: it then minimizes the same cost without the filter. Both log, per step, whether the
fallback fired and the chosen gains.'''

RCBF_CODE = r'''from mrncbf.qp.min_norm_cbf import _rcbf_closed_form_control, _check_and_promote, rcbf_qp_linear

N_RCBF_SAMPLES, RCBF_GRID, RCBF_GMIN, RCBF_GMAX, RCBF_REG, RCBF_SMOOTH = 20, 50, 1e-3, 8.0, 0.1, 0.05

def _rcbf_gammas(h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom, h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert,
                 unfiltered_fallback):
    """compute_rcbf_gammas_C (prev gammas 0, as in the compare notebook), optionally without the minimum-gain fallback."""
    h_V_nom, hx_Vx_nom = _check_and_promote(h_V_nom, hx_Vx_nom)
    vals = jnp.linspace(RCBF_GMIN, RCBF_GMAX, RCBF_GRID)
    g1, g2 = [a.ravel() for a in jnp.meshgrid(vals, vals, indexing='ij')]

    def sigma_and_feas(g1_i, g2_i):
        u_n, feas_n = _rcbf_closed_form_control(h_V_nom, hx_Vx_nom, f_nom, G_nom, u_nom,
                                                ALPHA_QP, u_lb, u_ub, g1_i, g2_i)
        u_p, feas_p = jax.vmap(lambda hV, hx, fi, Gi, un: _rcbf_closed_form_control(
            hV, hx, fi, Gi, un, ALPHA_QP, u_lb, u_ub, g1_i, g2_i))(h_V_pert, hx_Vx_pert, f_pert, G_pert, u_nom_pert)
        return jnp.max(jnp.linalg.norm(u_p - u_n[None], axis=-1)), feas_n & jnp.all(feas_p)

    sigma, feas = jax.vmap(sigma_and_feas)(g1, g2)
    cost = (jnp.maximum(sigma - g1, 0.0) / (2.0 * g2 + 1e-12)
            + RCBF_REG * (g1 + g2) + RCBF_SMOOTH * (g1 ** 2 + g2 ** 2))
    any_feas = jnp.any(feas)
    filtered = jnp.where(feas, cost, 1e6)
    cost = jnp.where(any_feas, filtered, cost) if unfiltered_fallback else filtered
    best = jnp.argmin(cost)
    return g1[best].astype(jnp.float32), g2[best].astype(jnp.float32), ~any_feas

def _make_rcbf_step(unfiltered_fallback):
    def _rcbf_step(carry, step_idx, goal):
        x, bias    = carry
        xhat       = x + bias
        samp_key   = jr.fold_in(BASE_KEY, step_idx)
        lo, hi     = _box_bounds(xhat)
        h_V_nom    = B_hocbf(xhat)
        hx_Vx_nom  = jax.jacobian(B_hocbf)(xhat)
        u_nom      = nom_pol_goto(xhat, goal)
        x_pert     = lo[None] + jr.uniform(samp_key, (N_RCBF_SAMPLES, 4)) * (hi - lo)
        g1, g2, no_feas = _rcbf_gammas(
            h_V_nom, hx_Vx_nom, f(xhat), G(xhat), u_nom,
            jax.vmap(B_hocbf)(x_pert), jax.vmap(jax.jacobian(B_hocbf))(x_pert), jax.vmap(f)(x_pert),
            jax.vmap(G)(x_pert), jax.vmap(lambda xp: nom_pol_goto(xp, goal))(x_pert), unfiltered_fallback)
        u, _, _ = rcbf_qp_linear(ALPHA_QP, u_lb, u_ub, h_V_nom, hx_Vx_nom, f(xhat), G(xhat), u_nom,
                                 gamma1=g1, gamma2=g2, relax_eps2=0.1)
        return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, jnp.stack([no_feas.astype(jnp.float32), g1, g2]))
    return _rcbf_step

STEP_FNS['R-CBF-QP (orig)'] = _make_rcbf_step(False)
STEP_FNS['R-CBF-QP (no-feas fallback)'] = _make_rcbf_step(True)
'''
