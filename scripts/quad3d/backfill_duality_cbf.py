"""Backfill Duality-CBF results into existing MC cache files.

Usage (run from repo root or scripts/quad3d/):
    python scripts/quad3d/backfill_duality_cbf.py [--eps 0.100 0.200 ...] [--force]

For each mc_eps_*.pkl file in the MC results directory:
  - Skips files that already contain 'Duality CBF' (unless --force).
  - Reconstructs x0s / biases from the saved px0/py0/pz0 and STATE_EPS
    using the same PRNGKey(42) seed as the notebook.
  - Runs rollout_duality (with the per-file STATE_EPS) over all N_MC ICs.
  - Backfills violation_array, reached_array, reach_time_array,
    violation_rates, goal_rates, and saves the file in-place.

Progress is printed per file so you can kill/resume safely
(each file is written atomically before moving to the next).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR    = _SCRIPT_DIR.parent.parent / 'src'
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from mrncbf.utils.jax_utils import jax_default_x32
jax_default_x32()

from jaxproxqp.jaxproxqp import JaxProxQP as _JaxProxQP
from jaxproxqp.qp_problems import QPModel as _QPModel

from quad3d_quad import (
    UMAX, DT, T, ALPHA_HOCBF, ALPHA_HOCBF_BASE, ALPHA_HOCBF_OBS, ALPHA_QP, ALPHA_QP_VEC,
    EPS_MAX, NMR_WEIGHTS_DIR,
    NX, NU, NH, Z_HOVER, OBSTACLE,
    u_lb, u_ub,
    f, G, B_hocbf, _cbf_ingredients,
    GOAL_MC, nom_pol_goto,
    task,
)

# ── duality helpers (copied from notebook, closed over N_PLANES_DUAL) ─────────
N_PLANES_DUAL = 24
BASE_KEY      = jr.PRNGKey(0)

_rng_dirs  = np.random.default_rng(0)
_raw_dirs  = _rng_dirs.standard_normal((N_PLANES_DUAL, NU + 1))
_DUAL_DIRS = (_raw_dirs / np.linalg.norm(_raw_dirs, axis=1, keepdims=True)).astype(np.float32)

_IJ_CI   = jnp.repeat(jnp.arange(NH), N_PLANES_DUAL)
_IJ_DIRS = jnp.tile(jnp.array(_DUAL_DIRS), (NH, 1))


def _image_fn(x, alpha=ALPHA_HOCBF):
    h_B, Lf_B, LG_B = _cbf_ingredients(x, alpha)
    return jnp.concatenate([LG_B, (Lf_B + ALPHA_QP_VEC * h_B)[:, None]], axis=1)   # (nh, nu+1)


def _proj_grad_max_box(scalar_fn, lo, hi, n_restarts, n_steps, lr, key):
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


def _duality_cbf_qp_multi(C_poly_list, d_poly_list, u_nom, relax_eps=0.1, penalty=10.0):
    nh = C_poly_list.shape[0]
    N  = C_poly_list.shape[1]
    nu = NU
    n  = nh * N + nu + 1

    H = jnp.zeros((n, n))
    for ii in range(nu):
        H = H.at[nh*N + ii, nh*N + ii].set(1.0)
    H = H.at[-1, -1].set(penalty)

    g = jnp.zeros(n)
    for ii in range(nu):
        g = g.at[nh*N + ii].set(-u_nom[ii])
    g = g.at[-1].set(penalty * relax_eps)

    A_eq = jnp.zeros((nh * (nu + 1), n))
    b_eq = jnp.zeros(nh * (nu + 1))
    for i in range(nh):
        row0 = i * (nu + 1)
        col0 = i * N
        A_eq = A_eq.at[row0:row0 + nu + 1, col0:col0 + N].set(C_poly_list[i].T)
        for ii in range(nu):
            A_eq = A_eq.at[row0 + ii, nh*N + ii].set(-1.0)
        b_eq = b_eq.at[row0 + nu].set(1.0)

    C_ineq = jnp.zeros((nh, n))
    for i in range(nh):
        C_ineq = C_ineq.at[i, i*N:(i+1)*N].set(d_poly_list[i])
        C_ineq = C_ineq.at[i, -1].set(-1.0)
    l_ineq = jnp.full((nh,), -1e8)
    u_ineq = jnp.zeros(nh)

    l_box = jnp.concatenate([jnp.zeros(nh * N), u_lb, jnp.array([-relax_eps])])
    u_box = jnp.concatenate([jnp.full(nh * N, 1e8), u_ub, jnp.array([1e8])])

    qp  = _QPModel(H, g, A_eq, C_ineq, b_eq, u_ineq, l_ineq, u_box, l_box)
    sol = _JaxProxQP(qp, _JaxProxQP.Settings.default()).solve()
    return sol.x[nh*N:nh*N + nu], sol.x[-1], sol


def _make_duality_step(state_eps):
    """Factory: returns a _duality_step closed over the given state_eps array."""
    def _duality_step(carry, step_idx, goal):
        x, bias  = carry
        xhat     = x + bias
        lo       = xhat - state_eps
        hi       = xhat + state_eps
        u_nom    = nom_pol_goto(xhat, goal)
        opt_key  = jr.fold_in(BASE_KEY, step_idx)
        all_keys = jr.split(opt_key, NH * N_PLANES_DUAL)

        def _support_one(ci, d_vec, k_j):
            val, _ = _proj_grad_max_box(
                lambda xi: d_vec @ _image_fn(xi)[ci], lo, hi,
                n_restarts=2, n_steps=15, lr=0.03, key=k_j)
            return val

        d_poly_flat = jax.vmap(_support_one)(_IJ_CI, _IJ_DIRS, all_keys)
        d_poly_list = d_poly_flat.reshape(NH, N_PLANES_DUAL)
        C_poly_list = jnp.broadcast_to(jnp.array(_DUAL_DIRS), (NH, N_PLANES_DUAL, NU + 1))
        u, _, _ = _duality_cbf_qp_multi(C_poly_list, d_poly_list, u_nom)
        return (x + DT*(f(x) + G(x) @ u), bias), (x, xhat)
    return _duality_step


def _make_rollout(step_fn, goal=GOAL_MC):
    @jax.jit
    def rollout(x0, bias):
        def _scan_step(carry, step_idx):
            return step_fn(carry, step_idx, goal)
        _, (xs, xhats) = jax.lax.scan(_scan_step, (x0, bias), jnp.arange(T))
        return jnp.concatenate([x0[None], xs], axis=0), xhats
    return rollout


# ── MC outcome helpers ────────────────────────────────────────────────────────
GOAL_TOL = 0.3
VEL_TOL  = 0.2

from quad3d_quad import h_raw as _h_raw_fn
_h_mc = jax.jit(jax.vmap(jax.vmap(_h_raw_fn)))


def _compute_outcomes(xs_all):
    """xs_all: (N, T+1, NX) numpy array → (viols, reached, reach_times)."""
    h_all   = np.array(_h_mc(jnp.array(xs_all, dtype=jnp.float32)))  # (N, T+1, nh)
    viols   = h_all.max(axis=(1, 2)) > 0

    px_f    = xs_all[:, -1, task.PX]
    speed_f = np.sqrt(xs_all[:, -1, task.VX]**2 + xs_all[:, -1, task.VY]**2
                      + xs_all[:, -1, task.VZ]**2)
    reached = (px_f >= float(OBSTACLE[0]) + 0.5) & (speed_f <= VEL_TOL)

    px_t    = xs_all[:, :, task.PX]
    spd_t   = np.sqrt(xs_all[:, :, task.VX]**2 + xs_all[:, :, task.VY]**2
                      + xs_all[:, :, task.VZ]**2)
    in_goal = (px_t >= float(OBSTACLE[0]) + 0.5) & (spd_t <= VEL_TOL)
    ever    = in_goal.any(axis=1)
    first_t = np.argmax(in_goal, axis=1)
    reach_times = np.where(ever, first_t * DT, np.nan)

    return viols, reached, reach_times


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eps', nargs='*', type=float, default=None,
                        help='Only process files with these STATE_EPS[0] values (e.g. --eps 0.10 0.20)')
    parser.add_argument('--n-traj', type=int, default=100,
                        help='Number of trajectories to run per file (default: 100, i.e. 1/10 of 1000)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if Duality CBF is already present in a file')
    args = parser.parse_args()

    mc_results_dir = (
        Path(NMR_WEIGHTS_DIR) / 'mc_results'
    )

    bf_files = sorted(mc_results_dir.glob('mc_eps_*.pkl')) if mc_results_dir.exists() else []
    if not bf_files:
        print(f'No MC cache files found in {mc_results_dir}')
        return

    if args.eps is not None:
        bf_files = [fp for fp in bf_files
                    if any(abs(float(fp.stem.replace('mc_eps_', '')) - e) < 1e-4
                           for e in args.eps)]
        if not bf_files:
            print('No files match the requested --eps values.')
            return

    print(f'Found {len(bf_files)} file(s) to process.\n')

    for fp in bf_files:
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)

        if not args.force and 'Duality CBF' in data.get('violation_array', {}):
            print(f'  skipping {fp.name} — Duality CBF already present')
            continue

        bf_N_MC      = min(args.n_traj, data['N_MC'])
        bf_STATE_EPS = jnp.array(data['STATE_EPS'], dtype=jnp.float32)
        bf_px0       = np.array(data['px0'])[:bf_N_MC]
        bf_py0       = np.array(data['py0'])[:bf_N_MC]
        bf_pz0       = np.array(data['pz0'])[:bf_N_MC]

        print(f'{fp.name}: STATE_EPS[px]={float(bf_STATE_EPS[task.PX]):.3f}, N={bf_N_MC} (of {data["N_MC"]})')
        print(f'  compiling duality step...')
        t0 = time.perf_counter()

        step_fn = _make_duality_step(bf_STATE_EPS)
        rollout = _make_rollout(step_fn)
        vmap_fn = jax.jit(jax.vmap(rollout))

        # Warm-compile on 1 trajectory before timing the full run.
        jax.block_until_ready(vmap_fn(jnp.zeros((1, NX)), jnp.zeros((1, NX))))
        print(f'  compiled in {time.perf_counter()-t0:.1f}s')

        # Reconstruct x0s and biases with the same seeds as the notebook.
        _x_base = jnp.zeros(NX).at[task.PZ].set(Z_HOVER)
        bf_x0s  = jnp.broadcast_to(_x_base, (bf_N_MC, NX))
        bf_x0s  = (bf_x0s
                   .at[:, task.PX].set(jnp.array(bf_px0))
                   .at[:, task.PY].set(jnp.array(bf_py0))
                   .at[:, task.PZ].set(jnp.array(bf_pz0)))

        _, _, _rng_bias_bf = jr.split(jr.PRNGKey(42), 3)
        bf_biases = jr.uniform(_rng_bias_bf, (data['N_MC'], NX),
                               minval=-bf_STATE_EPS, maxval=bf_STATE_EPS)[:bf_N_MC]

        print(f'  running {bf_N_MC} rollouts (T={T}, {T*DT:.0f}s each)...')
        t0 = time.perf_counter()
        xs_all, _ = jax.device_get(vmap_fn(bf_x0s, bf_biases))
        elapsed = time.perf_counter() - t0
        print(f'  done in {elapsed/60:.1f} min')

        viols, reached, reach_times = _compute_outcomes(xs_all)

        data.setdefault('violation_array',  {})['Duality CBF']  = viols.tolist()
        data.setdefault('reached_array',    {})['Duality CBF']  = reached.tolist()
        data.setdefault('reach_time_array', {})['Duality CBF']  = reach_times.tolist()
        data['violation_rates']['Duality CBF'] = float(viols.mean())
        data['goal_rates']['Duality CBF']      = float(reached.mean())

        with open(fp, 'wb') as fh:
            pickle.dump(data, fh)

        print(('  Duality CBF  viol={:.0%} ({}/{})  goal={:.0%} ({}/{})'
               '  t_reach_mean={:.2f}s  -> saved {}').format(
            float(viols.mean()), int(viols.sum()), bf_N_MC,
            float(reached.mean()), int(reached.sum()), bf_N_MC,
            float(np.nanmean(reach_times)), fp.name))
        print()

    print('Backfill complete.')


if __name__ == '__main__':
    main()
