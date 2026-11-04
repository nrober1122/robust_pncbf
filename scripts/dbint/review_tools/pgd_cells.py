PGD_MD = r'''## Certified DMR-CBF with projected gradient (the paper's inner solver)

Same constraint as above, $a + b(\hat x)u \ge \bar G(\hat x)$, but both worst cases come from the paper's projected
gradient ascent with random restarts (`proj_grad_max_box` settings: 4 restarts, 25 steps, lr 0.05) instead of the grid:
- $a_i = \min_{s \in \bar X \cap C} a_i(s)$,
- $\bar G_i = \max_{s \in \bar X \cap C} \sum_j \bar u_j\,|b_{ij}(\hat x) - b_{ij}(s)|$.

$\bar X \cap C$ is handled by keeping the best iterate along each restart that satisfies $h_i(s) \ge 0$. If no iterate
does, the unrestricted value over the box is used, which is conservative. Nothing here is specific to the double
integrator. PGD is not a global method, so Prop. 1 is checked against the exact terms afterwards rather than assumed.'''

PGD_CODE = r'''PGD_RESTARTS, PGD_STEPS, PGD_LR = 4, 25, 0.05

def _pgd_max_in_C(fn, cfn, lo, hi, key):
    """max of fn over the box, over iterates with cfn >= 0 (all iterates if none qualifies). Projected gradient ascent."""
    x0s = lo[None] + jr.uniform(key, (PGD_RESTARTS, lo.shape[0])) * (hi - lo)
    def run_one(x0):
        def step(carry, _):
            x, best_c, best_all = carry
            v = fn(x)
            best_c   = jnp.maximum(best_c, jnp.where(cfn(x) >= 0, v, -jnp.inf))
            best_all = jnp.maximum(best_all, v)
            return (jnp.clip(x + PGD_LR * jax.grad(fn)(x), lo, hi), best_c, best_all), None
        (x, best_c, best_all), _ = jax.lax.scan(step, (x0, -jnp.inf, -jnp.inf), None, length=PGD_STEPS)
        v = fn(x)
        return jnp.maximum(best_c, jnp.where(cfn(x) >= 0, v, -jnp.inf)), jnp.maximum(best_all, v)
    best_c, best_all = jax.vmap(run_one)(x0s)
    return jnp.where(jnp.isfinite(best_c.max()), best_c.max(), best_all.max())

def pgd_terms(xhat, key):
    """a (nh,) and Gbar (nh,) by projected gradient over box ∩ {h_i >= 0}."""
    lo, hi = _box_bounds(xhat)
    b_hat  = _b_mat(xhat)
    u_abs  = jnp.maximum(jnp.abs(u_lb), jnp.abs(u_ub))
    nh     = b_hat.shape[0]
    keys   = jr.split(key, 2 * nh).reshape(nh, 2, -1)
    def per_i(i, k):
        cfn  = lambda s: _h_vec(s)[i]
        a_i  = -_pgd_max_in_C(lambda s: -_a_vec(s)[i], cfn, lo, hi, k[0])
        G_i  = _pgd_max_in_C(lambda s: jnp.sum(u_abs * jnp.abs(b_hat[i] - _b_mat(s)[i])), cfn, lo, hi, k[1])
        return a_i, G_i
    return jax.vmap(per_i)(jnp.arange(nh), keys)

def _pgd_cert_step(carry, step_idx, goal):
    x, bias    = carry
    xhat       = x + bias
    u_nom      = nom_pol_goto(xhat, goal)
    a_lo, Gbar = pgd_terms(xhat, jr.fold_in(BASE_KEY, step_idx))
    J_hat      = jax.jacobian(B_hocbf)(xhat)
    h_V        = ((Gbar - a_lo) - J_hat @ f(xhat)) / ALPHA_QP
    u, r, _    = min_norm_cbf(ALPHA_QP, u_lb, u_ub, h_V, J_hat, f(xhat), G(xhat), u_nom,
                              penalty=CERT_PENALTY, relax_eps1=CERT_EPS1, relax_eps2=CERT_EPS2)
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, jnp.stack([a_lo[0], Gbar[0], r]))

STEP_FNS['PGD DMR-CBF'] = _pgd_cert_step
'''
