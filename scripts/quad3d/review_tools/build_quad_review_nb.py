"""Builds scripts/quad3d/quad3d_review.ipynb (run from the repo root):

    python scripts/quad3d/review_tools/build_quad_review_nb.py scripts/quad3d/quad3d_review.ipynb

This overwrites the target notebook and its outputs.
"""
import json
import sys

CELLS = []
def md(s):   CELLS.append(('markdown', s))
def code(s): CELLS.append(('code', s))

md(r'''# 12D Quadrotor — certified DMR-CBF (review response)

Quadrotor counterpart of `scripts/dbint/dbint2d_review.ipynb` (R1 Comments 2–3). Paper sign throughout: $h_i \ge 0$ safe,
$a_i(s) = L_f\psi_{1,i}(s) + \alpha_i\psi_{1,i}(s)$, $b_i(s) = L_g\psi_{1,i}(s)$, one condition per constraint
(6 floor/ceiling/attitude constraints + the obstacle).

- **DMR-CBF (orig.)**: the compare notebook's `_lfhwc_step` (`PSHR-CBF`): $a_i + b_i(\hat x)u \ge 0$, soft QP.
- **PGD DMR-CBF**: $a_i + b_i(\hat x)u \ge \bar G_i(\hat x)$ for every $i$, with
  $a_i = \min_{s\in\bar X\cap C_i} a_i(s)$ and $\bar G_i = \max_{s\in\bar X\cap C_i}\sum_j \bar u_j|b_{ij}(\hat x) - b_{ij}(s)|$ from the same
  projected gradient as the original (4 restarts, 25 steps, lr 0.03), $C_i = \{h_i \ge 0\}$. Hard QP (slack only when infeasible).

There are no exact worst-case terms here, so Prop. 1 is checked on a subset of steps with a much stronger projected
gradient (the reference), and bracketed from below by the second-order Taylor bound.''')

code(r'''import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from mrncbf.utils.jax_utils import jax_default_x32
from mrncbf.qp.min_norm_cbf import min_norm_cbf

import quad3d_quad as Q
from quad3d_quad import (
    DT, T, NX, NU, NH, Z_HOVER, OBSTACLE, ALPHA_QP_VEC, NMR_WEIGHTS_DIR,
    u_lb, u_ub, f, G, h_raw, B_hocbf, GOAL_MC, nom_pol_goto, task,
)

jax_default_x32()

BASE_KEY        = jr.PRNGKey(0)
EPS_SHAPE       = Q.STATE_EPS / Q.STATE_EPS[task.PX]   # eps = position uncertainty; other dims scale with it
STATE_EPS       = Q.STATE_EPS
EPS             = float(STATE_EPS[task.PX])
_mc_results_dir = Path(NMR_WEIGHTS_DIR) / 'mc_results'   # compare notebook caches (mc_eps_*.pkl), read only
_review_dir     = _mc_results_dir / 'review'           # this notebook's results; never overwritten
print(f'nx={NX} nu={NU} nh={NH}  DT={DT}  T={T}  eps={EPS:.2f}')
print(f'EPS_SHAPE = {np.round(np.asarray(EPS_SHAPE), 3)}')
print(f'compare-notebook MC caches present: {sorted(p.name for p in _mc_results_dir.glob("mc_eps_*.pkl"))}')
''')

md('## Helpers and paper-sign terms')

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
    """rollout(x0, bias) -> (xs, xhats, logs); xs[0] = x0, xs[t+1] = pre-update x_t."""
    @jax.jit
    def rollout(x0, bias):
        _, (xs, xhats, logs) = jax.lax.scan(lambda c, k: step_fn(c, k, goal), (x0, bias), jnp.arange(T))
        return jnp.concatenate([x0[None], xs], axis=0), xhats, logs
    return rollout

def _a_vec(s):   # paper-sign a_i(s), (NH,)
    return -(jax.jacobian(B_hocbf)(s) @ f(s) + ALPHA_QP_VEC * B_hocbf(s))

def _b_mat(s):   # paper-sign b_i(s), (NH, NU)
    return -(jax.jacobian(B_hocbf)(s) @ G(s))

def _h_vec(s):   # paper-sign h_i(s) >= 0 safe, (NH,)
    return -h_raw(s)

U_ABS = jnp.maximum(jnp.abs(u_lb), jnp.abs(u_ub))
# logs layout per step: [u (NU) | a (NH) | Gbar (NH) | slack r]
L_U, L_A, L_G, L_R = slice(0, NU), slice(NU, NU + NH), slice(NU + NH, NU + 2 * NH), NU + 2 * NH
''')

md(r'''## Controllers

`_pgd_max_in_C` keeps the best iterate along each restart that lies in $C_i$; if none does, it uses the unrestricted box
value, which is conservative.''')

code(r'''PGD_RESTARTS, PGD_STEPS, PGD_LR    = 4, 25, 0.03       # compare notebook's DMR-CBF settings
CERT_PENALTY, CERT_EPS1, CERT_EPS2 = 10.0, 0.0, 100.0   # r >= 0, exact penalty (slope 1000): slack only if infeasible

def _dmr_step(carry, step_idx, goal):
    # Copy of the compare notebook's _lfhwc_step (PSHR-CBF), with logging
    x, bias   = carry
    xhat      = x + bias
    lo, hi    = _box_bounds(xhat)
    u_nom     = nom_pol_goto(xhat, goal)
    keys      = jr.split(jr.fold_in(BASE_KEY, step_idx), NH)
    h_nom     = B_hocbf(xhat)
    J_nom     = jax.jacobian(B_hocbf)(xhat)
    Lf_nom    = J_nom @ f(xhat)
    def worst_i(i, k):
        _, x_wc = proj_grad_max_box(
            lambda xi: (jax.jacobian(B_hocbf)(xi) @ f(xi) + ALPHA_QP_VEC * B_hocbf(xi))[i],
            lo, hi, n_restarts=PGD_RESTARTS, n_steps=PGD_STEPS, lr=PGD_LR, key=k)
        return B_hocbf(x_wc)[i], (jax.jacobian(B_hocbf)(x_wc) @ f(x_wc))[i]
    h_wc, Lf_wc = jax.vmap(worst_i)(jnp.arange(NH), keys)
    phi_oracle  = (Lf_wc + ALPHA_QP_VEC * h_wc) - (Lf_nom + ALPHA_QP_VEC * h_nom)
    u, r, _ = min_norm_cbf(ALPHA_QP_VEC, u_lb, u_ub, h_nom + phi_oracle / ALPHA_QP_VEC, J_nom,
                           f(xhat), G(xhat), u_nom, relax_eps2=0.1)
    a_ctrl = -(Lf_wc + ALPHA_QP_VEC * h_wc)
    log = jnp.concatenate([u, a_ctrl, jnp.full(NH, jnp.nan), r[None]])
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, log)

def _pgd_max_in_C(fn, cfn, lo, hi, key, n_restarts=PGD_RESTARTS, n_steps=PGD_STEPS, lr=PGD_LR):
    """max of fn over the box, over projected-gradient iterates with cfn >= 0 (all iterates if none qualifies)."""
    x0s = lo[None] + jr.uniform(key, (n_restarts, lo.shape[0])) * (hi - lo)
    def run_one(x0):
        def step(carry, _):
            x, best_c, best_all = carry
            v = fn(x)
            best_c   = jnp.maximum(best_c, jnp.where(cfn(x) >= 0, v, -jnp.inf))
            best_all = jnp.maximum(best_all, v)
            return (jnp.clip(x + lr * jax.grad(fn)(x), lo, hi), best_c, best_all), None
        (x, best_c, best_all), _ = jax.lax.scan(step, (x0, -jnp.inf, -jnp.inf), None, length=n_steps)
        v = fn(x)
        return jnp.maximum(best_c, jnp.where(cfn(x) >= 0, v, -jnp.inf)), jnp.maximum(best_all, v)
    best_c, best_all = jax.vmap(run_one)(x0s)
    return jnp.where(jnp.isfinite(best_c.max()), best_c.max(), best_all.max())

def pgd_terms(xhat, key, **pgd_kw):
    """a (NH,) and Gbar (NH,) by projected gradient over box ∩ {h_i >= 0}."""
    lo, hi = _box_bounds(xhat)
    b_hat  = _b_mat(xhat)
    keys   = jr.split(key, 2 * NH).reshape(NH, 2, -1)
    def per_i(i, k):
        cfn = lambda s: _h_vec(s)[i]
        a_i = -_pgd_max_in_C(lambda s: -_a_vec(s)[i], cfn, lo, hi, k[0], **pgd_kw)
        G_i = _pgd_max_in_C(lambda s: jnp.sum(U_ABS * jnp.abs(b_hat[i] - _b_mat(s)[i])), cfn, lo, hi, k[1], **pgd_kw)
        return a_i, G_i
    return jax.vmap(per_i)(jnp.arange(NH), keys)

def _pgd_cert_step(carry, step_idx, goal):
    x, bias    = carry
    xhat       = x + bias
    u_nom      = nom_pol_goto(xhat, goal)
    a_lo, Gbar = pgd_terms(xhat, jr.fold_in(BASE_KEY, step_idx))
    J_hat      = jax.jacobian(B_hocbf)(xhat)
    # paper a_i + b_i(xhat) u >= Gbar_i  <=>  J f + alpha h_V + J G u <= r  with:
    h_V        = ((Gbar - a_lo) - J_hat @ f(xhat)) / ALPHA_QP_VEC
    u, r, _    = min_norm_cbf(ALPHA_QP_VEC, u_lb, u_ub, h_V, J_hat, f(xhat), G(xhat), u_nom,
                              penalty=CERT_PENALTY, relax_eps1=CERT_EPS1, relax_eps2=CERT_EPS2)
    log = jnp.concatenate([u, a_lo, Gbar, r[None]])
    return (x + DT * (f(x) + G(x) @ u), bias), (x, xhat, log)

STEP_FNS = {'DMR-CBF (orig.)': _dmr_step, 'PGD DMR-CBF': _pgd_cert_step}
''')

md(r'''## Initial conditions

The compare notebook's MC samples (seed 42): hover at a random $(p_x, p_y, p_z)$ in `MC_BOUNDS`, biases uniform in the box.
Biases use the same key for every ε and scale with `STATE_EPS`. `EVAL_IDX` is the compare notebook's 200-sample spaghetti
subset.''')

code(r'''N_MC      = 1_000
GOAL_TOL  = 0.3
VEL_TOL   = 0.2
MC_BOUNDS = np.array([[-1.5, -0.25, Z_HOVER - 0.25], [-2.0, 0.25, Z_HOVER + 0.25]])

rng_mc = jr.PRNGKey(42)
rng_mc, rng_ic, rng_bias = jr.split(rng_mc, 3)
x0pos  = jr.uniform(rng_ic, (N_MC, 3), minval=MC_BOUNDS[0], maxval=MC_BOUNDS[1])
x0s    = jnp.broadcast_to(jnp.zeros(NX).at[task.PZ].set(Z_HOVER), (N_MC, NX))
x0s    = x0s.at[:, task.PX].set(x0pos[:, 0]).at[:, task.PY].set(x0pos[:, 1]).at[:, task.PZ].set(x0pos[:, 2])
EVAL_IDX = np.random.default_rng(42).choice(N_MC, size=200, replace=False)

def set_eps(eps):
    global STATE_EPS, EPS, biases
    STATE_EPS = EPS_SHAPE * eps
    EPS       = float(eps)
    biases    = jr.uniform(rng_bias, (N_MC, NX), minval=-STATE_EPS, maxval=STATE_EPS)

set_eps(EPS)
''')

md(r'''## Rollouts

`run_eps(eps, n)` rolls out `RUN_POLICIES` on the first `n` samples of the MC set (all 1000 by default) and saves
`quad_review_eps_{eps}_N{n}.pkl` in `_review_dir`. Existing files are never overwritten. Outcomes follow the compare
notebook: unsafe if any $h_i < 0$ along the true trajectory; reached when $p_x \ge c_x + 0.5$ and speed $\le$ `VEL_TOL`.''')

code(r'''EPS_LIST     = [0.1, 0.3]
N_RUN        = N_MC
RUN_POLICIES = ['DMR-CBF (orig.)', 'PGD DMR-CBF']
ROLL_CHUNK   = 50

def _rollout_path(eps, n):
    return _review_dir / f'quad_review_eps_{eps:.3f}_N{n}.pkl'

def _outcomes(xs):
    h_all   = np.asarray(jax.vmap(jax.vmap(h_raw))(jnp.asarray(xs)))
    coll    = h_all.max(axis=(1, 2)) > 0
    spd     = np.linalg.norm(xs[:, :, [task.VX, task.VY, task.VZ]], axis=-1)
    in_goal = (xs[:, :, task.PX] >= float(OBSTACLE[0]) + 0.5) & (spd <= VEL_TOL)
    reached = in_goal[:, -1]
    t_reach = np.where(in_goal.any(1), np.argmax(in_goal, 1) * DT, np.nan)
    return coll, reached, t_reach

def _chunked(fn, *arrays):
    out = [jax.device_get(fn(*(jnp.asarray(a[i:i + ROLL_CHUNK]) for a in arrays)))
           for i in range(0, len(arrays[0]), ROLL_CHUNK)]
    return tuple(np.concatenate([np.asarray(o[k]) for o in out]) for k in range(len(out[0])))

def run_eps(eps, n=None):
    n = n or N_RUN
    path = _rollout_path(eps, n)
    if path.exists():
        print(f'  eps={eps:.2f}: {path.name} exists, not overwriting')
        return
    set_eps(eps)
    results = {}
    for name in RUN_POLICIES:
        t0 = time.time()
        rollout = jax.jit(jax.vmap(_make_rollout_logged(STEP_FNS[name])))
        xs, xhats, logs = _chunked(rollout, np.asarray(x0s)[:n], np.asarray(biases)[:n])
        coll, reached, t_reach = _outcomes(xs)
        results[name] = {'xs': xs, 'xhats': xhats, 'logs': logs, 'coll': coll, 'reached': reached, 't_reach': t_reach}
        print(f'  eps={eps:.2f}  {name:<16} {time.time() - t0:7.1f} s   unsafe {coll.mean():6.1%}  '
              f'reached {reached.mean():6.1%}  t_goal {np.nanmean(t_reach[~coll]):.2f} s', flush=True)
    _review_dir.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'STATE_EPS': np.asarray(STATE_EPS), 'N': n, 'results': results,
                     'pgd': (PGD_RESTARTS, PGD_STEPS, PGD_LR),
                     'cert_qp': (CERT_PENALTY, CERT_EPS1, CERT_EPS2)}, f)
    print(f'  saved {path}', flush=True)

for _eps in EPS_LIST:
    run_eps(_eps)
''')

md(r'''## Certificate check on a subset of steps

For trajectories in `EVAL_IDX` (every `EVAL_STRIDE`-th step), with $u^\star$ from the controller log:
- **reference** ($\approx$ exact): projected gradient with `REF_RESTARTS` restarts × `REF_STEPS` steps over $\bar X \cap C_i$ for
  $a_i$, $G_i(u^\star) = \max_s (b_i(\hat x) - b_i(s))u^\star$ and $M_i(u^\star) = \min_s a_i(s) + b_i(s)u^\star$;
- **lower bound** $\underline M_i(u^\star)$: second-order Taylor bound on $a_i(s) + b_i(s)u^\star$ over the box (∩ the
  linearised $C_i$), with Hessians sampled at the centre, random vertices and random points.

Prop. 1 holds (reference) when $a_i + b_i(\hat x)u^\star \ge G_i(u^\star)$. $\underline M \ge 0$ certifies $M \ge 0$ up to the
Hessian sampling. The controller's PGD error is its own $a_i$ minus the reference $a_i$ (> 0: optimistic).''')

code(r'''REF_RESTARTS, REF_STEPS = 16, 100
EVAL_STRIDE  = 20
EVAL_POLICY  = 'PGD DMR-CBF'
TAY_N_VERT, TAY_N_RAND = 32, 16

def _lp_min(c, eps, g, beta):
    """min c.d  s.t. |d_k| <= eps_k,  g.d >= -beta.  Greedy fractional knapsack; returns (value, feasible)."""
    d0   = -eps * jnp.sign(c)
    need = -beta - g @ d0
    sg   = jnp.sign(g)
    gain = jnp.abs(g) * (eps - sg * d0)
    rate = jnp.where(jnp.abs(g) > 1e-12, c * sg / jnp.maximum(jnp.abs(g), 1e-12), jnp.inf)
    o    = jnp.argsort(rate)
    gs, rs = gain[o], rate[o]
    take = jnp.clip(need - (jnp.cumsum(gs) - gs), 0.0, gs)
    extra = jnp.sum(jnp.where(take > 0, take * rs, 0.0))
    return c @ d0 + jnp.where(need > 0, extra, 0.0), jnp.sum(gs) >= need

def _curv(Hs, eps, sign):
    """Bound on max_d (sign * -d^T H d)_+ over the box from sampled Hessians Hs (K, n, n)."""
    E    = jnp.diag(eps)
    lam  = jnp.linalg.eigvalsh(sign * Hs)[:, 0].min()
    lamE = jnp.linalg.eigvalsh(sign * (E @ Hs @ E))[:, 0].min()
    return jnp.minimum(jnp.sum(eps ** 2) * jnp.maximum(0.0, -lam), eps.shape[0] * jnp.maximum(0.0, -lamE))

def _taylor_min(fn, xhat, i, key):
    """Lower bound on min of scalar fn over box ∩ {linearised h_i >= 0}."""
    eps = STATE_EPS
    k1, k2 = jr.split(key)
    S  = xhat + jnp.concatenate([jnp.zeros((1, NX)), jr.rademacher(k1, (TAY_N_VERT, NX)).astype(jnp.float32),
                                 jr.uniform(k2, (TAY_N_RAND, NX), minval=-1.0, maxval=1.0)]) * eps
    hi_fn = lambda s: _h_vec(s)[i]
    beta  = hi_fn(xhat) + 0.5 * _curv(jax.vmap(jax.hessian(hi_fn))(S), eps, -1.0)
    v, ok = _lp_min(jax.grad(fn)(xhat), eps, jax.grad(hi_fn)(xhat), beta)
    v     = jnp.where(ok, v, -jnp.sum(eps * jnp.abs(jax.grad(fn)(xhat))))
    return fn(xhat) + v - 0.5 * _curv(jax.vmap(jax.hessian(fn))(S), eps, 1.0)

def _eval_state(xhat, u, key):
    lo, hi = _box_bounds(xhat)
    b_hat  = _b_mat(xhat)
    ref    = dict(n_restarts=REF_RESTARTS, n_steps=REF_STEPS, lr=PGD_LR)
    def per_i(i, k):
        k = jr.split(k, 4)
        cfn   = lambda s: _h_vec(s)[i]
        a_ref = -_pgd_max_in_C(lambda s: -_a_vec(s)[i], cfn, lo, hi, k[0], **ref)
        G_u   = _pgd_max_in_C(lambda s: (b_hat[i] - _b_mat(s)[i]) @ u, cfn, lo, hi, k[1], **ref)
        m     = lambda s: _a_vec(s)[i] + _b_mat(s)[i] @ u
        M_ref = -_pgd_max_in_C(lambda s: -m(s), cfn, lo, hi, k[2], **ref)
        M_lo  = _taylor_min(m, xhat, i, k[3])
        return jnp.stack([a_ref + b_hat[i] @ u, G_u, M_ref, M_lo, a_ref])
    return jax.vmap(per_i)(jnp.arange(NH), jr.split(key, NH))   # (NH, 5)

def _eval_path(eps, n):
    return _review_dir / f'quad_review_eval_eps_{eps:.3f}_N{n}.pkl'

def run_eval(eps, n=None, policy=None):
    n, policy = n or N_RUN, policy or EVAL_POLICY
    path = _eval_path(eps, n)
    if path.exists():
        print(f'  eps={eps:.2f}: {path.name} exists, not overwriting')
        return
    set_eps(eps)
    with open(_rollout_path(eps, n), 'rb') as f:
        r = pickle.load(f)['results'][policy]
    idx   = EVAL_IDX[EVAL_IDX < n]
    steps = np.arange(0, T, EVAL_STRIDE)
    xh    = r['xhats'][idx][:, steps].reshape(-1, NX)
    u     = r['logs'][idx][:, steps, L_U].reshape(-1, NU)
    keys  = jr.split(jr.PRNGKey(7), len(xh))
    fn    = jax.jit(jax.vmap(_eval_state))
    t0 = time.time()
    ev = np.concatenate([np.asarray(fn(jnp.asarray(xh[i:i + 200]), jnp.asarray(u[i:i + 200]), keys[i:i + 200]))
                         for i in range(0, len(xh), 200)]).reshape(len(idx), len(steps), NH, 5)
    print(f'  eps={eps:.2f}: evaluated {len(xh)} states in {time.time() - t0:.0f} s', flush=True)
    with open(path, 'wb') as f:
        pickle.dump({'policy': policy, 'idx': idx, 'steps': steps, 'eval': ev,
                     'ref': (REF_RESTARTS, REF_STEPS), 'taylor': (TAY_N_VERT, TAY_N_RAND)}, f)
    print(f'  saved {path}', flush=True)

for _eps in EPS_LIST:
    run_eval(_eps)
''')

md(r'''## Summary

Outcomes for all saved rollouts, the original DMR-CBF and the baselines from the compare notebook's MC caches (when present),
and the certificate check. Constraint names: floor, ceiling, roll−, roll+, pitch−, pitch+, obstacle.''')

code(r'''SLACK_TOL = 1e-5
H_NAMES   = ['floor', 'ceil', 'roll-', 'roll+', 'pitch-', 'pitch+', 'obstacle']

for _p in sorted(_review_dir.glob('quad_review_eps_*_N*.pkl')):
    with open(_p, 'rb') as _f:
        _R = pickle.load(_f)
    _e = float(_R['STATE_EPS'][task.PX])
    print(f'=== eps = {_e:.2f}, N = {_R["N"]}')
    _mc = _mc_results_dir / f'mc_eps_{_e:.3f}.pkl'
    if _mc.exists():
        with open(_mc, 'rb') as _f:
            _m = pickle.load(_f)
        for _k in _m['violation_array']:
            _v = np.asarray(_m['violation_array'][_k], bool)
            _t = np.asarray(_m['reach_time_array'][_k])
            print(f'  [cache] {_k:<16} unsafe {_v.mean():6.1%}  reached {np.mean(_m["reached_array"][_k]):6.1%}  '
                  f't_goal {np.nanmean(_t[~_v]):.2f} s')
    for _name, _r in _R['results'].items():
        _sl = _r['logs'][:, :, L_R] > SLACK_TOL
        print(f'  {_name:<16}         unsafe {_r["coll"].mean():6.1%}  reached {_r["reached"].mean():6.1%}  '
              f't_goal {np.nanmean(_r["t_reach"][~_r["coll"]]):.2f} s   slack on {_sl.mean():.2%} of steps '
              f'({_sl.any(1).sum()} trajs)')
    if 'DMR-CBF (orig.)' in _R['results'] and 'PGD DMR-CBF' in _R['results']:
        _a, _b = _R['results']['DMR-CBF (orig.)']['t_reach'], _R['results']['PGD DMR-CBF']['t_reach']
        _ok = ~np.isnan(_a) & ~np.isnan(_b)
        print(f'  PGD DMR-CBF vs DMR-CBF (orig.), paired time to goal: {np.mean(_b[_ok] / _a[_ok]) - 1:+.1%}')
    _ep = _eval_path(_e, _R['N'])
    if _ep.exists():
        with open(_ep, 'rb') as _f:
            _E = pickle.load(_f)
        ev = _E['eval']                                      # (n_traj, n_steps, NH, 5)
        r  = _R['results'][_E['policy']]
        a_ctrl = r['logs'][_E['idx']][:, _E['steps'], L_A]
        prop1  = ev[..., 0] >= ev[..., 1]
        print(f'  certificate check ({_E["policy"]}, {ev.shape[0]} trajs x {ev.shape[1]} steps, reference PGD '
              f'{_E["ref"][0]}x{_E["ref"][1]}):')
        for i, nm in enumerate(H_NAMES):
            d = a_ctrl[..., i] - ev[..., i, 4]
            print(f'    {nm:<8} Prop. 1 {prop1[..., i].mean():8.3%}   min M_ref {ev[..., i, 2].min():+8.3f}   '
                  f'M_lo >= 0 {np.mean(ev[..., i, 3] >= 0):7.2%}   PGD a error p99 {np.percentile(d, 99):+.3f} '
                  f'max {d.max():+.3f}')
        print(f'    all constraints: Prop. 1 on {prop1.all(-1).mean():.3%} of steps, {prop1.all((1, 2)).mean():.1%} of trajs')
''')

md('## Per-step compute time (single state, after JIT)')

code(r'''_x0, _b0 = jnp.asarray(x0s[0]), jnp.asarray(biases[0])
for _name, _fn in STEP_FNS.items():
    _step = jax.jit(lambda c, k, fn=_fn: fn(c, k, GOAL_MC))
    jax.block_until_ready(_step((_x0, _b0), 0))
    _t0 = time.perf_counter()
    for _k in range(200):
        jax.block_until_ready(_step((_x0, _b0), _k))
    print(f'  {_name:<16} {1e3 * (time.perf_counter() - _t0) / 200:6.2f} ms/step')
''')

md('## Trajectories (x–y)')

code(r'''PLOT_EPS = 0.3
_p = _rollout_path(PLOT_EPS, N_RUN)
if _p.exists():
    with open(_p, 'rb') as _f:
        _R = pickle.load(_f)
    fig, axes = plt.subplots(1, len(_R['results']), figsize=(6 * len(_R['results']), 4), sharey=True)
    for ax, (_name, _r) in zip(np.atleast_1d(axes), _R['results'].items()):
        _sel = EVAL_IDX[EVAL_IDX < _R['N']]
        for i in _sel:
            ax.plot(_r['xs'][i, :, task.PX], _r['xs'][i, :, task.PY], lw=0.6,
                    color='k' if _r['coll'][i] else 'tab:blue', alpha=0.6 if _r['coll'][i] else 0.3)
        ax.add_patch(plt.Circle(OBSTACLE[:2], OBSTACLE[3], color='0.5', alpha=0.6, zorder=5))
        ax.set_aspect('equal'); ax.set_xlabel('$p_x$ (m)'); ax.set_title(f'{_name} (eps = {PLOT_EPS})')
    np.atleast_1d(axes)[0].set_ylabel('$p_y$ (m)')
    plt.show()
''')

if __name__ == '__main__':
    meta = json.load(open('scripts/quad3d/quad3d_cbf_compare.ipynb'))['metadata']
    nb = {'cells': [], 'metadata': meta, 'nbformat': 4, 'nbformat_minor': 5}
    for i, (ct, s) in enumerate(CELLS):
        if ct == 'code':
            compile(s, f'<cell {i}>', 'exec')
        c = {'cell_type': ct, 'id': f'qreview-{i:02d}', 'metadata': {}, 'source': s.splitlines(keepends=True)}
        if ct == 'code':
            c.update({'execution_count': None, 'outputs': []})
        nb['cells'].append(c)
    json.dump(nb, open(sys.argv[1], 'w'), indent=1)
    print('wrote', sys.argv[1], len(nb['cells']), 'cells')
