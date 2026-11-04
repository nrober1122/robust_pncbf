TAYLOR_MD = r'''## General worst-case terms: second-order Taylor bound

The grid search above uses this barrier's structure. The Taylor bound below needs only autodiff of $a$, $b$ and $h$, so it
applies unchanged to other systems (e.g. the quadrotor). With $\delta = s - \hat x$, $|\delta_k| \le \varepsilon_k$:

$$a_i(s) \;\ge\; a_i(\hat x) + \nabla a_i(\hat x)^\top\delta - \tfrac12\,\mu_i, \qquad
\mu_i \ge \max_{s \in \bar X}\, \max_{\delta}\, \bigl(-\delta^\top \nabla^2 a_i(s)\,\delta\bigr)_+ ,$$

so only negative curvature costs margin. The curvature term is bounded two ways and the smaller is used:
$\|\varepsilon\|^2 \max(0, -\lambda_{\min}(\nabla^2 a_i))$ and $n \max(0, -\lambda_{\min}(E\nabla^2 a_i E))$ with
$E = \mathrm{diag}(\varepsilon)$. The eigenvalue bound is taken over sample points of $\bar X$ (centre, vertices, random
points): it is **not** a rigorous bound, but an error in it enters only at second order in $\varepsilon$.

**Restriction to $\bar X \cap C$.** $C \subseteq \{h_i \ge 0\}$ and $h_i(s) \le h_i(\hat x) + \nabla h_i(\hat x)^\top\delta + \tfrac12\nu_i$,
so every $s \in \bar X \cap C$ satisfies one linear inequality in $\delta$. The worst case of the linear term over the box
with that half-space is a fractional-knapsack LP with a closed-form (greedy) solution (`_lp_min`).

**$\bar G$.** The same bounds on $b_{ij}(s) - b_{ij}(\hat x)$ in both directions give
$\bar G_i = \sum_j \bar u_j \max_{s} |b_{ij}(\hat x) - b_{ij}(s)|$.'''

TAYLOR_CODE = r'''TAY_N_RAND = 32          # random Hessian sample points in the box (plus centre and vertices)
TAY_MAX_VERT = 64        # vertices used (all of them when 2^n is smaller)

def _a_vec(s):           # paper-sign a_i(s), (nh,)
    return -(jax.jacobian(B_hocbf)(s) @ f(s) + ALPHA_QP * B_hocbf(s))

def _b_mat(s):           # paper-sign b_i(s), (nh, nu)
    return -(jax.jacobian(B_hocbf)(s) @ G(s))

def _h_vec(s):           # paper-sign h_i(s) >= 0 safe, (nh,)
    return -h_raw(s)

def _lp_min(c, eps, g, beta):
    """min c.d  s.t. |d_k| <= eps_k,  g.d >= -beta.  Greedy fractional knapsack; returns (value, feasible)."""
    d0   = -eps * jnp.sign(c)                      # unconstrained minimiser (a vertex)
    need = -beta - g @ d0                          # extra g.d still required
    sg   = jnp.sign(g)
    room = eps - sg * d0                           # how far each coordinate can move in direction sign(g)
    gain = jnp.abs(g) * room
    rate = jnp.where(jnp.abs(g) > 1e-12, c * sg / jnp.maximum(jnp.abs(g), 1e-12), jnp.inf)
    o    = jnp.argsort(rate)
    gs, rs = gain[o], rate[o]
    take = jnp.clip(need - (jnp.cumsum(gs) - gs), 0.0, gs)
    extra = jnp.sum(jnp.where(take > 0, take * rs, 0.0))
    return c @ d0 + jnp.where(need > 0, extra, 0.0), jnp.sum(gs) >= need

def _sample_box(c, eps, key):
    # Hessian sample points in the box c +- eps: centre, vertices (or a random subset), uniform points
    n = c.shape[0]
    if 2 ** n <= TAY_MAX_VERT:
        signs = jnp.array(np.array(np.meshgrid(*[[-1.0, 1.0]] * n, indexing='ij')).reshape(n, -1).T)
    else:
        k1, key = jr.split(key)
        signs = jr.rademacher(k1, (TAY_MAX_VERT, n)).astype(jnp.float32)
    rand = jr.uniform(key, (TAY_N_RAND, n), minval=-1.0, maxval=1.0)
    return c + jnp.concatenate([jnp.zeros((1, n)), signs, rand]) * eps

def _curv(Hs, eps, sign):
    """Second-order margin from sampled Hessians Hs (K, n, n): bound on max_d (sign * -d^T H d)_+ over the box."""
    E    = jnp.diag(eps)
    lam  = jnp.linalg.eigvalsh(sign * Hs)[:, 0].min()                 # most negative curvature (after sign)
    lamE = jnp.linalg.eigvalsh(sign * (E @ Hs @ E))[:, 0].min()
    return jnp.minimum(jnp.sum(eps ** 2) * jnp.maximum(0.0, -lam), eps.shape[0] * jnp.maximum(0.0, -lamE))

def _taylor_box(c, eps, key, restrict_C):
    """Bounds over one box c +- eps (∩ C if restrict_C): a_lo (nh,), box ∩ C nonempty (nh,), b_hi and b_lo (nh, nu)."""
    S  = _sample_box(c, eps, key)
    a0, Ja = _a_vec(c), jax.jacobian(_a_vec)(c)                       # (nh,), (nh, n)
    b0, Jb = _b_mat(c), jax.jacobian(_b_mat)(c)                       # (nh, nu), (nh, nu, n)
    h0, Jh = _h_vec(c), jax.jacobian(_h_vec)(c)
    Ha = jax.vmap(jax.hessian(_a_vec))(S)                             # (K, nh, n, n)
    Hb = jax.vmap(jax.hessian(_b_mat))(S)                             # (K, nh, nu, n, n)
    Hh = jax.vmap(jax.hessian(_h_vec))(S)
    nu = b0.shape[1]

    def per_i(i):
        # half-space containing (box ∩ C) - c:  Jh_i.d >= -(h_i + nu_i/2), nu_i from positive curvature of h_i
        beta = h0[i] + 0.5 * _curv(Hh[:, i], eps, -1.0)
        def lin_min(v):
            if not restrict_C:
                return -jnp.sum(eps * jnp.abs(v)), True
            val, ok = _lp_min(v, eps, Jh[i], beta)
            return jnp.where(ok, val, -jnp.sum(eps * jnp.abs(v))), ok
        va, ok = lin_min(Ja[i])
        a_lo = a0[i] + va - 0.5 * _curv(Ha[:, i], eps, 1.0)
        def b_range(j):
            hi = b0[i, j] - lin_min(-Jb[i, j])[0] + 0.5 * _curv(Hb[:, i, j], eps, -1.0)
            lo = b0[i, j] + lin_min(Jb[i, j])[0]  - 0.5 * _curv(Hb[:, i, j], eps, 1.0)
            return hi, lo
        b_hi, b_lo = jax.vmap(b_range)(jnp.arange(nu))
        return a_lo, jnp.asarray(ok), b_hi, b_lo
    return jax.vmap(per_i)(jnp.arange(a0.shape[0]))

def _partition(parts):
    # Sub-box centre offsets (in units of eps) and half-width scale for `parts` pieces per dimension
    axes = [(-1.0 + (2 * np.arange(m) + 1) / m) for m in parts]
    offs = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, len(parts))
    return jnp.asarray(offs, jnp.float32), jnp.asarray(1.0 / np.asarray(parts), jnp.float32)

TAY_PARTS = (1, 1, 1, 1)   # pieces per state dimension; (1, ..., 1) is the single-box bound

def taylor_terms(xhat, key, restrict_C=True, parts=None):
    """a_lo (nh,) and Gbar (nh,) from second-order Taylor bounds on a partition of the box (∩ C if restrict_C).

    Sub-boxes whose outer approximation misses C are dropped (sound, since the true state is in C)."""
    offs, scale = _partition(parts or TAY_PARTS)
    eps_sub = STATE_EPS * scale
    keys    = jr.split(key, offs.shape[0])
    a_lo, ok, b_hi, b_lo = jax.vmap(lambda o, k: _taylor_box(xhat + o * STATE_EPS, eps_sub, k, restrict_C))(offs, keys)
    ok   = ok | ~ok.any(0, keepdims=True)                             # keep everything if no piece survives
    a_lo = jnp.where(ok, a_lo, jnp.inf).min(0)
    b0   = _b_mat(xhat)
    okb  = ok[:, :, None]
    dev  = jnp.maximum(jnp.where(okb, b_hi, -jnp.inf).max(0) - b0, b0 - jnp.where(okb, b_lo, jnp.inf).min(0))
    Gbar = (jnp.maximum(jnp.abs(u_lb), jnp.abs(u_ub)) * dev).sum(-1)
    return a_lo, Gbar

def _taylor_step(carry, step_idx, goal):
    x, bias    = carry
    xhat       = x + bias
    u_nom      = nom_pol_goto(xhat, goal)
    a_lo, Gbar = taylor_terms(xhat, jr.fold_in(BASE_KEY, step_idx))
    J_hat      = jax.jacobian(B_hocbf)(xhat)
    h_V        = ((Gbar - a_lo) - J_hat @ f(xhat)) / ALPHA_QP
    u, r, _    = min_norm_cbf(ALPHA_QP, u_lb, u_ub, h_V, J_hat, f(xhat), G(xhat), u_nom,
                              penalty=CERT_PENALTY, relax_eps1=CERT_EPS1, relax_eps2=CERT_EPS2)
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, jnp.stack([a_lo[0], Gbar[0], r]))

STEP_FNS['Taylor DMR-CBF'] = _taylor_step
'''
