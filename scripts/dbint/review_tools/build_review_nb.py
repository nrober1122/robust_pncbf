import json, sys

CELLS = []
def md(s):   CELLS.append(('markdown', s))
def code(s): CELLS.append(('code', s))

md(r'''# 2D Double Integrator — certified DMR-CBF (review response)

Supports the response to Reviewer 1, Comment 2 (Prop. 1 is not enforced by the DMR-CBF) and Comment 3
(the inner minimisation uses projected gradient with random restarts).

Paper sign convention throughout: $h(x) = \|p\| - R \ge 0$ is safe, $a(s) = L_f\psi_1(s) + \alpha\psi_1(s)$,
$b(s) = L_g\psi_1(s)$, and the CBF condition is $a(x) + b(x)u \ge 0$. (`B_hocbf` in `dbint2d_quad` is $-\psi_1$.)

**Certified DMR-CBF.** Replace the DMR-CBF constraint (7), $a + b(\hat x)u \ge 0$, by
$$a + b(\hat x)\,u \;\ge\; \bar G(\hat x), \qquad \bar G(\hat x) \ge \max_{u \in U} G(u),$$
so Prop. 1's condition $a + b(\hat x)u \ge G(u)$ holds for every feasible $u$. It is still a single half-space in
$u$ (point-based in the control term). The constraint is hard: slack is allowed only when it is infeasible, and is logged.

**Worst-case terms without random restarts.** For this barrier, $b(s) = p/\|p\| = n(p)$ depends on position only, and
$$a(p, v) = \frac{(t\cdot v)^2}{\|p\|} + (\alpha+\alpha_0)\, n\cdot v + \alpha\alpha_0(\|p\| - R),$$
with $t \perp n$. For fixed $p$ this is convex in $v$, and its minimum over the velocity box lies on the box edges, where it
has a closed form. So $a = \min_s a(s)$, $G(u)$ and $M(u)$ reduce to a 2D grid over the position box.
- $\bar G(\hat x) = \max_{p} \|n(\hat p) - n(p)\|_1$ on the grid, plus a Lipschitz remainder for the grid spacing, is a sound upper bound.
- $a$ is minimised on the grid, then on a finer local grid around the best few points. It has no remainder term;
  its accuracy is checked against a denser independent evaluation below.

All worst cases are taken over $\bar X \cap C$ (grid points within one half-cell of $C$). This suffices for Prop. 1's
argument, which needs the condition only at the true state, and the true state is in $C$ while safety holds.
The certificate is continuous-time; collisions are measured on the Euler-discretised rollout ($\Delta t$ = `DT`).
''')

code(r'''import functools as ft
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from mrncbf.utils.jax_utils import jax_default_x32
from mrncbf.qp.min_norm_cbf import min_norm_cbf

from dbint2d_quad import (
    UMAX, R_OBS, DT, T, ALPHA_HOCBF, ALPHA_QP, H_SHIFT,
    STATE_EPS, NMR_WEIGHTS_DIR, u_lb, u_ub,
    f, G, h_raw, B_hocbf, GOAL_MC, nom_pol_goto,
)

jax_default_x32()

BASE_KEY        = jr.PRNGKey(0)
_mc_results_dir = Path(NMR_WEIGHTS_DIR) / 'mc_results/full_scaling2x1_combined'
EPS             = float(STATE_EPS[0])
assert H_SHIFT == 0.0, 'closed forms below assume no barrier shift'
print(f'STATE_EPS = {np.asarray(STATE_EPS)}  (eps = {EPS:.2f}),  R_OBS = {R_OBS},  '
      f'alpha = {ALPHA_QP},  alpha0 = {ALPHA_HOCBF},  DT = {DT},  T = {T}')
''')

md('## Shared helpers (copied from `dbint2d_cbf_compare.ipynb`)')

code(r'''def _box_bounds(xhat):
    return xhat - STATE_EPS, xhat + STATE_EPS

def proj_grad_max_box(scalar_fn, lo, hi, n_restarts, n_steps, lr, key):
    x0s = lo[None] + jr.uniform(key, (n_restarts, lo.shape[0])) * (hi - lo)
    def run_one(x0):
        def step(x, _):
            g = jax.grad(scalar_fn)(x)
            return jnp.clip(x + lr * g, lo, hi), None
        x_star, _ = jax.lax.scan(step, x0, None, length=n_steps)
        return scalar_fn(x_star), x_star
    vals, xs = jax.vmap(run_one)(x0s)
    best = jnp.argmax(vals)
    return vals[best], xs[best]

def _make_rollout_logged(step_fn, goal=GOAL_MC):
    """rollout(x0, bias) -> (xs, xhats, logs). xs[0] = x0 and xs[t+1] = pre-update x_t, as in the compare notebook."""
    @jax.jit
    def rollout(x0, bias):
        _, (xs, xhats, logs) = jax.lax.scan(lambda c, k: step_fn(c, k, goal), (x0, bias), jnp.arange(T))
        return jnp.concatenate([x0[None], xs], axis=0), xhats, logs
    return rollout

P_IDX, V_IDX = jnp.array([0, 2]), jnp.array([1, 3])
''')

md('## Paper-sign closed forms, checked against autodiff of `B_hocbf`')

code(r'''C_LIN = ALPHA_QP + ALPHA_HOCBF   # coefficient of n.v in a(s)
C_POS = ALPHA_QP * ALPHA_HOCBF   # coefficient of (||p|| - R) in a(s)

def _a_b(s):
    """Paper-sign a(s), b(s) by autodiff (B_hocbf = -psi_1)."""
    J = jax.jacobian(B_hocbf)(s)[0]
    return -(J @ f(s) + ALPHA_QP * B_hocbf(s)[0]), -(J @ G(s))

def _a_closed(p, v):
    rho = jnp.linalg.norm(p)
    n   = p / rho
    t   = jnp.array([-n[1], n[0]])
    return (t @ v) ** 2 / rho + C_LIN * (n @ v) + C_POS * (rho - R_OBS)

_rng = np.random.default_rng(0)
_n_checked = 0
while _n_checked < 50:
    _s = _rng.uniform([-2.5, -1.5, -1.5, -1.5], [2.5, 1.5, 1.5, 1.5]).astype(np.float32)
    if np.hypot(_s[0], _s[2]) < 0.1:
        continue
    _a_ad, _b_ad = _a_b(jnp.asarray(_s))
    _p, _v = jnp.asarray(_s[[0, 2]]), jnp.asarray(_s[[1, 3]])
    assert np.isclose(float(_a_ad), float(_a_closed(_p, _v)), rtol=1e-4, atol=1e-4), (_s, _a_ad, _a_closed(_p, _v))
    assert np.allclose(np.asarray(_b_ad), np.asarray(_p / jnp.linalg.norm(_p)), atol=1e-5), (_s, _b_ad)
    _n_checked += 1
print(f'closed-form a(s) and b(s) = p/||p|| match autodiff at {_n_checked} random states')
''')

md(r'''## Worst-case terms on a position grid

`_vmin_q` is the exact minimum over the velocity box of the convex part $(t\cdot v)^2/\|p\| + (\alpha+\alpha_0)\,n\cdot v$.
The objective's gradient in the $n$ direction is the nonzero constant $\alpha+\alpha_0$, so the minimum lies on the box
boundary. On each edge the objective is a 1D convex quadratic, which has a closed-form minimiser.''')

code(r'''GRID_K, K_REF, N_REF       = 41, 8, 11    # controller: coarse grid, refined points, local grid size
K_FINE, K_REF_F, N_REF_F   = 81, 16, 21   # independent evaluation (denser)

def _grid(K):
    # Offsets of a K x K grid over the position box, its half cell diagonal and its cell size
    e_p  = np.asarray(STATE_EPS)[[0, 2]]
    g    = np.linspace(-1.0, 1.0, K)
    offs = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2) * e_p
    cell = 2 * e_p / (K - 1)
    return jnp.asarray(offs, dtype=jnp.float32), 0.5 * float(np.linalg.norm(cell)), jnp.asarray(cell, dtype=jnp.float32)

def _vmin_q(p, vlo, vhi):
    # Exact min over the velocity box of (t.v)^2/||p|| + C_LIN n.v (convex; attained on the boundary)
    rho = jnp.linalg.norm(p)
    n   = p / rho
    t   = jnp.array([-n[1], n[0]])
    q   = lambda v: (t @ v) ** 2 / rho + C_LIN * (n @ v)
    corners = jnp.array([[vlo[0], vlo[1]], [vhi[0], vlo[1]], [vhi[0], vhi[1]], [vlo[0], vhi[1]]])
    def edge_min(v0, v1):
        d, al = v1 - v0, t @ v0
        be    = t @ d
        safe  = jnp.abs(be) > 1e-9
        tau   = jnp.where(safe, (-C_LIN * (n @ d) * rho / (2 * jnp.where(safe, be, 1.0)) - al)
                          / jnp.where(safe, be, 1.0), 0.0)
        tau   = jnp.clip(tau, 0.0, 1.0)
        return jnp.minimum(q(v0 + tau * d), jnp.minimum(q(v0), q(v1)))
    return jnp.min(jax.vmap(edge_min)(corners, jnp.roll(corners, -1, axis=0)))

def _a_p(p, vlo, vhi):
    # min over the velocity box of a(p, v)
    return _vmin_q(p, vlo, vhi) + C_POS * (jnp.linalg.norm(p) - R_OBS)

def _unit(p):
    return p / jnp.linalg.norm(p)

def _masked(obj_p, ps, r_c):
    # obj_p at grid points within half a cell (r_c) of C, inf elsewhere
    keep = jnp.linalg.norm(ps, axis=-1) >= R_OBS - r_c
    ps   = jnp.where(keep[:, None], ps, jnp.array([R_OBS, 0.0]))   # dummy for dropped points
    return jnp.where(keep, jax.vmap(obj_p)(ps), jnp.inf)

def _grid_min(obj_p, xhat, grid, k_ref, n_ref):
    # min over box ∩ C of obj_p(p): coarse grid, then an n_ref x n_ref grid spanning +-1 cell
    # around each of the k_ref best coarse points
    offs, r_c, cell = grid
    lo, hi = _box_bounds(xhat)
    ps0    = xhat[P_IDX] + offs
    v0     = _masked(obj_p, ps0, r_c)
    _, idx = jax.lax.top_k(-v0, k_ref)
    loc    = jnp.linspace(-1.0, 1.0, n_ref)
    loc    = jnp.stack(jnp.meshgrid(loc, loc, indexing='ij'), -1).reshape(-1, 2) * cell
    ps1    = jnp.clip((ps0[idx][:, None, :] + loc[None]).reshape(-1, 2), lo[P_IDX], hi[P_IDX])
    return jnp.minimum(jnp.min(v0), jnp.min(_masked(obj_p, ps1, r_c * 2 / (n_ref - 1))))

def cert_terms(xhat, grid, k_ref, n_ref):
    # a_lo = min over box ∩ C of a(s) (refined grid); Gbar >= max_{u in U} G(u) (grid + Lipschitz remainder)
    offs, r_c, _ = grid
    lo, hi = _box_bounds(xhat)
    a_lo   = _grid_min(lambda p: _a_p(p, lo[V_IDX], hi[V_IDX]), xhat, grid, k_ref, n_ref)
    n_hat  = _unit(xhat[P_IDX])
    d1     = _masked(lambda p: -jnp.abs(n_hat - _unit(p)).sum(), xhat[P_IDX] + offs, r_c)
    # |u_j| <= UMAX, and n(p) is (1/||p||)-Lipschitz, with ||p|| >= R_OBS - r_c on kept cells
    Gbar   = UMAX * (-jnp.min(d1) + jnp.sqrt(2.0) * r_c / (R_OBS - r_c))
    return a_lo, Gbar

_GRID, _GRID_F = _grid(GRID_K), _grid(K_FINE)
print(f'controller grid {GRID_K}x{GRID_K} (+{K_REF} x {N_REF}x{N_REF} local): half cell {_GRID[1]:.4f} m, '
      f'Gbar remainder {UMAX * np.sqrt(2) * _GRID[1] / (R_OBS - _GRID[1]):.4f}')

# Check _vmin_q against brute force on a dense velocity grid
_vg = np.stack(np.meshgrid(*[np.linspace(0, 1, 201)] * 2, indexing='ij'), -1).reshape(-1, 2)
for _ in range(20):
    _p   = _rng.uniform(-1.5, 1.5, 2).astype(np.float32)
    if np.linalg.norm(_p) < 0.2:
        continue
    _vlo = _rng.uniform(-1.2, 0.8, 2).astype(np.float32)
    _vhi = _vlo + 2 * np.asarray(STATE_EPS)[[1, 3]]
    _vs  = _vlo + _vg * (_vhi - _vlo)
    _brute = float(jnp.min(jax.vmap(lambda v: _a_closed(jnp.asarray(_p), v))(jnp.asarray(_vs))))
    _exact = float(_a_p(jnp.asarray(_p), jnp.asarray(_vlo), jnp.asarray(_vhi)))
    assert _exact <= _brute + 1e-4 and _brute - _exact < 1e-2, (_p, _vlo, _exact, _brute)
print('exact velocity-box minimum matches brute force')
''')

md(r'''## Controllers

- **DMR-CBF**: a copy of `_lfhwc_step` (`PSHR-CBF` in the compare notebook), with the same oracle, key and soft QP
  (`relax_eps2=0.1`). It also logs its own $a$ and the QP slack $r$.
- **Certified DMR-CBF**: $a + b(\hat x)u \ge \bar G(\hat x)$, with $a$ and $\bar G$ from the position grid. The QP is hard:
  $r \ge 0$ with an exact-penalty slope of `CERT_PENALTY * CERT_EPS2`, so $r > 0$ only when the constraint is infeasible.''')

code(r'''CERT_PENALTY, CERT_EPS1, CERT_EPS2 = 10.0, 0.0, 100.0   # r >= 0; penalty slope at r = 0 is 1000

def _dmr_step(carry, step_idx, goal):
    x, bias   = carry
    xhat      = x + bias
    lo, hi    = _box_bounds(xhat)
    u_nom     = nom_pol_goto(xhat, goal)
    opt_key   = jr.fold_in(BASE_KEY, step_idx)

    h_nom  = B_hocbf(xhat)
    J_nom  = jax.jacobian(B_hocbf)(xhat)
    Lf_nom = J_nom @ f(xhat)

    nh   = h_nom.shape[0]
    keys = jr.split(opt_key, nh)

    def worst_i(i, k):
        _, x_wc = proj_grad_max_box(
            lambda xi: (jax.jacobian(B_hocbf)(xi) @ f(xi) + ALPHA_QP * B_hocbf(xi))[i],
            lo, hi, n_restarts=4, n_steps=25, lr=0.05, key=k)
        h_wc_i  = B_hocbf(x_wc)[i]
        Lf_wc_i = (jax.jacobian(B_hocbf)(x_wc) @ f(x_wc))[i]
        return h_wc_i, Lf_wc_i

    h_wc, Lf_wc = jax.vmap(worst_i)(jnp.arange(nh), keys)
    phi_oracle  = (Lf_wc + ALPHA_QP * h_wc) - (Lf_nom + ALPHA_QP * h_nom)

    u, r, _ = min_norm_cbf(ALPHA_QP, u_lb, u_ub, h_nom + phi_oracle / ALPHA_QP, J_nom,
                           f(xhat), G(xhat), u_nom, relax_eps2=0.1)
    a_ctrl = -(Lf_wc + ALPHA_QP * h_wc)[0]
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, jnp.stack([a_ctrl, jnp.nan, r]))

def _cert_step(carry, step_idx, goal):
    x, bias    = carry
    xhat       = x + bias
    u_nom      = nom_pol_goto(xhat, goal)
    a_lo, Gbar = cert_terms(xhat, _GRID, K_REF, N_REF)
    J_hat      = jax.jacobian(B_hocbf)(xhat)                     # (1, 4), notebook sign
    # paper a_lo + b(xhat) u >= Gbar  <=>  J f(xhat) + alpha h_V + J G u <= r with
    h_V        = ((Gbar - a_lo) - J_hat @ f(xhat)) / ALPHA_QP     # (1,)
    u, r, _    = min_norm_cbf(ALPHA_QP, u_lb, u_ub, h_V, J_hat, f(xhat), G(xhat), u_nom,
                              penalty=CERT_PENALTY, relax_eps1=CERT_EPS1, relax_eps2=CERT_EPS2)
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, jnp.stack([a_lo, Gbar, r]))

''')

md(r'''## Initial conditions

These are the compare notebook's MC samples (seed 42). Biases are drawn from the same key for each ε, scaled by that
ε's `STATE_EPS`, exactly as the compare notebook's MC cell does. `N_EVAL = 200` selects the same 200-trajectory subset
(seed 42) as the paper's spaghetti figure.''')

code(r'''N_MC       = 1_000
VEL_TOL    = 0.1
MC_BOUNDS  = np.array([[-2.5, -0.25], [-1.5, 0.25]])
ROLL_CHUNK = 25
EPS_SHAPE  = jnp.array([1.0, 0.5, 1.0, 0.5])   # STATE_EPS = eps * EPS_SHAPE (paper: e = [eps, eps, eps/2, eps/2])
assert np.allclose(np.asarray(STATE_EPS), EPS * np.asarray(EPS_SHAPE)), 'dbint2d_quad STATE_EPS has a different shape'

rng_mc = jr.PRNGKey(42)
rng_mc, rng_ic, rng_bias = jr.split(rng_mc, 3)
x0pos  = jr.uniform(rng_ic, (N_MC, 2), minval=MC_BOUNDS[0], maxval=MC_BOUNDS[1])
x0s    = jnp.stack([x0pos[:, 0], jnp.zeros(N_MC), x0pos[:, 1], jnp.zeros(N_MC)], axis=1)
''')

md(r'''## Evaluation functions

The certificate quantities are recomputed independently with the denser search (`K_FINE`, `K_REF_F`, `N_REF_F`), with
$u^\star_t$ recovered from the rollout ($u_t = (v_{t+1} - v_t)/\Delta t$, exact for Euler):
- (7) margin $a + b(\hat x)u^\star$,
- $G(u^\star) = b(\hat x)u^\star - \min_s b(s)u^\star$,
- Prop. 1 certificate $a + b(\hat x)u^\star \ge G(u^\star)$,
- direct $M(u^\star) = \min_s [a(s) + b(s)u^\star]$, with the inner minimisation over velocity done exactly.

It also reports the error in each controller's own $a$, i.e. controller minus evaluation (positive means the
controller's $a$ is optimistic). For the DMR-CBF that is the projected-gradient oracle (Comment 3).''')

code(r'''def _cert_eval_step(x_t, u_t, bias):
    xhat     = x_t + bias
    lo, hi   = _box_bounds(xhat)
    vlo, vhi = lo[V_IDX], hi[V_IDX]
    gmin     = lambda obj: _grid_min(obj, xhat, _GRID_F, K_REF_F, N_REF_F)
    a_lo     = gmin(lambda p: _a_p(p, vlo, vhi))
    bu_hat   = _unit(xhat[P_IDX]) @ u_t
    G_u      = bu_hat - gmin(lambda p: _unit(p) @ u_t)
    M_u      = gmin(lambda p: _a_p(p, vlo, vhi) + _unit(p) @ u_t)
    return jnp.stack([a_lo + bu_hat, G_u, M_u, a_lo])

def _cert_eval_traj(xs_traj, bias):
    x_t = xs_traj[1:-1]
    u_t = (xs_traj[2:, V_IDX] - xs_traj[1:-1, V_IDX]) / DT
    return jax.lax.map(lambda xu: _cert_eval_step(xu[0], xu[1], bias), (x_t, u_t))

def _outcomes(xs, xhats):
    h_all   = np.asarray(jax.vmap(jax.vmap(h_raw))(jnp.asarray(xs)))[..., 0]   # notebook sign: > 0 inside
    coll    = h_all.max(1) > 0
    spd_f   = np.hypot(xs[:, -1, 1], xs[:, -1, 3])
    reached = (xhats[:, -1, 0] >= 0) & (spd_f <= VEL_TOL) & ~coll
    in_goal = (xs[:, :, 0] >= 0) & (np.hypot(xs[:, :, 1], xs[:, :, 3]) <= VEL_TOL)
    t_reach = np.where(in_goal.any(1), np.argmax(in_goal, 1) * DT, np.nan)
    return coll, reached, t_reach

def _chunked(fn, *arrays):
    out = [jax.device_get(fn(*(jnp.asarray(a[i:i + ROLL_CHUNK]) for a in arrays)))
           for i in range(0, len(arrays[0]), ROLL_CHUNK)]
    if isinstance(out[0], (tuple, list)):
        return tuple(np.concatenate([np.asarray(o[k]) for o in out]) for k in range(len(out[0])))
    return np.concatenate([np.asarray(o) for o in out])
''')

md(r'''## Run

`run_eps` sets `STATE_EPS`, the biases and both grids for one ε, rebuilds the jitted controllers (they read these globals
when traced), runs the rollouts, evaluates them and saves `review_cert_eps_{eps}_N{N_EVAL}.pkl` to the MC results folder.
- Default: the current ε and the paper's 200-trajectory subset.
- Full Monte Carlo: `EPS_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`, `N_EVAL = N_MC`. `RUN_POLICIES = ['Certified DMR-CBF']`
  is enough there, since the DMR-CBF's 1000-sample outcomes are already cached (`mc_eps_*.pkl`, as `PSHR-CBF`).''')

code(r'''EPS_LIST     = [EPS]                                  # e.g. [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_EVAL       = 200                                    # 200 = paper's spaghetti subset; N_MC = full Monte Carlo
RUN_POLICIES = ['DMR-CBF', 'Certified DMR-CBF']
SKIP_SAVED   = True                                   # skip an (eps, N_EVAL) that is already saved

STEP_FNS = {'DMR-CBF': _dmr_step, 'Certified DMR-CBF': _cert_step}

def _save_path(eps, n_eval):
    return _mc_results_dir / f'review_cert_eps_{eps:.3f}_N{n_eval}.pkl'

def run_eps(eps):
    global STATE_EPS, EPS, biases, _GRID, _GRID_F
    STATE_EPS      = EPS_SHAPE * eps
    EPS            = float(eps)
    biases         = jr.uniform(rng_bias, (N_MC, 4), minval=-STATE_EPS, maxval=STATE_EPS)
    _GRID, _GRID_F = _grid(GRID_K), _grid(K_FINE)
    eval_idx = (np.random.default_rng(42).choice(N_MC, size=N_EVAL, replace=False)
                if N_EVAL < N_MC else np.arange(N_MC))
    bias_eval = np.asarray(biases)[eval_idx]
    cert_eval = jax.jit(jax.vmap(_cert_eval_traj))

    results = {}
    for name in RUN_POLICIES:
        t0 = time.time()
        rollout = jax.jit(jax.vmap(_make_rollout_logged(STEP_FNS[name])))
        xs, xhats, logs = _chunked(rollout, np.asarray(x0s)[eval_idx], bias_eval)
        t1 = time.time()
        ev = _chunked(cert_eval, xs, bias_eval)                     # (N, T-1, 4)
        r = {'xs': xs, 'xhats': xhats, 'logs': logs}
        r['m7'], r['G_u'], r['M'], r['a_fine'] = (ev[..., k] for k in range(4))
        r['coll'], r['reached'], r['t_reach'] = _outcomes(xs, xhats)
        results[name] = r
        print(f'  eps={EPS:.2f}  {name:<18} rollout {t1 - t0:6.1f} s, eval {time.time() - t1:6.1f} s', flush=True)

    with open(_save_path(EPS, N_EVAL), 'wb') as f:
        pickle.dump({
            'STATE_EPS': np.asarray(STATE_EPS), 'N_EVAL': N_EVAL, 'eval_idx': eval_idx,
            'grid': {'controller': (GRID_K, K_REF, N_REF), 'eval': (K_FINE, K_REF_F, N_REF_F)},
            'cert_qp': {'penalty': CERT_PENALTY, 'relax_eps1': CERT_EPS1, 'relax_eps2': CERT_EPS2},
            'results': results,
        }, f)
    print(f'  saved {_save_path(EPS, N_EVAL)}', flush=True)
    return results, eval_idx

for _eps in EPS_LIST:
    if SKIP_SAVED and _save_path(_eps, N_EVAL).exists():
        print(f'  eps={_eps:.2f}: already saved, skipping ({_save_path(_eps, N_EVAL).name})')
        continue
    results, _eval_idx = run_eps(_eps)
''')

md(r'''## Summary

Loads every saved `(eps, N_EVAL)` result. For context it adds the DMR-CBF's outcomes from the paper's 1000-sample MC cache
(`PSHR-CBF` in `mc_eps_*.pkl`), on the same trajectories.''')

code(r'''SLACK_TOL = 1e-5

def _pct(mask):
    return f'{100 * np.mean(mask):5.1f}%'

def summarize(path):
    with open(path, 'rb') as f:
        R = pickle.load(f)
    eps, n_eval, idx = float(R['STATE_EPS'][0]), R['N_EVAL'], R['eval_idx']
    print(f'=== eps = {eps:.2f}, N = {n_eval}  (certificate quantities from the denser evaluation search; paper sign)')
    mc_file = _mc_results_dir / f'mc_eps_{eps:.3f}.pkl'
    if mc_file.exists():
        with open(mc_file, 'rb') as f:
            mc = pickle.load(f)
        v  = np.asarray(mc['violation_array']['PSHR-CBF'], bool)[idx]
        g  = np.asarray(mc['reached_array']['PSHR-CBF'], bool)[idx]
        tr = np.asarray(mc['reach_time_array']['PSHR-CBF'])[idx]
        print(f'  [paper MC cache] DMR-CBF: collisions {_pct(v)} ({v.sum()}/{len(v)}), reached safely '
              f'{_pct(g & ~v)}, mean time to goal {np.nanmean(tr[g & ~v]):.2f} s')
    for name, r in R['results'].items():
        a_ctrl, r_slack = r['logs'][:, :-1, 0], r['logs'][:, :-1, 2]   # steps 0..T-2 align with m7 (xs[t+1] = x_t)
        prop1 = (r['m7'] - r['G_u']) >= 0
        ok    = r['reached'] & ~np.isnan(r['t_reach'])
        print(f'  {name}')
        print(f'    collisions                     : {_pct(r["coll"])}   ({r["coll"].sum()}/{n_eval})')
        print(f'    reached safely / timeout       : {_pct(r["reached"])} / {_pct(~r["reached"] & ~r["coll"])}'
              f'   mean time to goal {np.nanmean(r["t_reach"][ok]):.2f} s')
        print(f'    QP slack used (r > {SLACK_TOL:g})     : {_pct(r_slack > SLACK_TOL)} of steps, '
              f'{_pct((r_slack > SLACK_TOL).any(1))} of trajs, max r {np.nanmax(r_slack):.4f}')
        print(f'    Prop. 1 certificate holds      : {_pct(prop1)} of steps, '
              f'{_pct(prop1.all(1))} of trajs certified on every step, worst {np.min(r["m7"] - r["G_u"]):+.4f}')
        print(f'    direct M(u*) >= 0              : {_pct(r["M"] >= 0)} of steps, worst M {r["M"].min():+.4f}')
        print(f'    controller a minus eval a      : max {np.nanmax(a_ctrl - r["a_fine"]):+.4f}, '
              f'p99 {np.nanpercentile(a_ctrl - r["a_fine"], 99):+.4f}   (> 0: controller a optimistic)')
        if name.startswith('Certified'):
            Gbar = r['logs'][:, :-1, 1]
            print(f'    Gbar                           : median {np.median(Gbar):.3f}  p95 '
                  f'{np.percentile(Gbar, 95):.3f}  max {Gbar.max():.3f}')
    print()
    return R

SUMMARY_N = N_EVAL
_saved = sorted(_mc_results_dir.glob(f'review_cert_eps_*_N{SUMMARY_N}.pkl'))
_loaded = {float(R['STATE_EPS'][0]): R for R in (summarize(p) for p in _saved)}
''')

md('## Plots (one saved result)')

code(r'''PLOT_EPS = EPS
R_plot   = _loaded[min(_loaded, key=lambda e: abs(e - PLOT_EPS))]
_n_plot  = min(R_plot['N_EVAL'], 200)
fig, axes = plt.subplots(1, len(R_plot['results']) + 1, figsize=(5.7 * (len(R_plot['results']) + 1), 4.8))
for ax, (_name, _r) in zip(axes[:-1], R_plot['results'].items()):
    for i in range(_n_plot):
        ax.plot(_r['xs'][i, :, 0], _r['xs'][i, :, 2], lw=0.6,
                color='k' if _r['coll'][i] else 'tab:blue', alpha=0.6 if _r['coll'][i] else 0.35)
    ax.add_patch(Circle((0, 0), R_OBS, facecolor='0.45', edgecolor='0.4', alpha=0.55, zorder=5))
    ax.scatter(*np.asarray(GOAL_MC)[[0, 2]], s=150, color='gold', edgecolors='k', marker='*', zorder=6)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
    ax.set_title(f'{_name}  (collisions {_pct(_r["coll"]).strip()})')
    ax.set_xlabel('$p_x$'); ax.set_ylabel('$p_y$')

_t_ax = np.arange(T - 1) * DT
for (_name, _r), _c in zip(R_plot['results'].items(), ['tab:orange', 'tab:green']):
    axes[-1].plot(_t_ax, _r['M'].min(0), color=_c, lw=1.4, label=f'{_name}: $\\min_i M_i(t)$')
axes[-1].axhline(0, color='k', lw=0.8, ls='--')
axes[-1].set_xlabel('t (s)'); axes[-1].set_ylabel(r'$M(u^\star_t)$ (paper sign, $\geq 0$ certified)')
axes[-1].set_title('Worst-case certificate margin over trajectories')
axes[-1].grid(True, alpha=0.3); axes[-1].legend(fontsize=8)
fig.suptitle(f"eps = {float(R_plot['STATE_EPS'][0]):.2f}, {R_plot['N_EVAL']} trajectories "
             f"(first {_n_plot} drawn)")
fig.tight_layout()
plt.show()
''')

sys.path.insert(0, __import__('os').path.dirname(__file__))
from taylor_cells import TAYLOR_MD, TAYLOR_CODE
from pgd_cells import PGD_MD, PGD_CODE
from spaghetti_cells import VAR_MD, VAR_CODE, SPAG_MD, SPAG_CODE, CMP_MD, CMP_CODE
md('# Inner solvers for the worst-case terms')
for _m, _c in [(TAYLOR_MD, TAYLOR_CODE), (PGD_MD, PGD_CODE), (VAR_MD, VAR_CODE), (SPAG_MD, SPAG_CODE), (CMP_MD, CMP_CODE)]:
    md(_m)
    code(_c)
from response_cells import RESP_MD, RESP_CODE, RESP_BAR_MD, RESP_BAR_CODE
md(RESP_MD)
code(RESP_CODE)
md(RESP_BAR_MD)
code(RESP_BAR_CODE)
from paper_cells import PAPER_MD, PAPER_CODE
md(PAPER_MD)
code(PAPER_CODE)

meta = json.load(open('scripts/dbint/dbint2d_cbf_compare.ipynb'))['metadata']
nb = {'cells': [], 'metadata': meta, 'nbformat': 4, 'nbformat_minor': 5}
for i, (ct, s) in enumerate(CELLS):
    if ct == 'code':
        compile(s, f'<cell {i}>', 'exec')
    c = {'cell_type': ct, 'id': f'review-{i:02d}', 'metadata': {}, 'source': s.splitlines(keepends=True)}
    if ct == 'code':
        c.update({'execution_count': None, 'outputs': []})
    nb['cells'].append(c)
out = sys.argv[1]
json.dump(nb, open(out, 'w'), indent=1)
print('wrote', out, len(nb['cells']), 'cells')
