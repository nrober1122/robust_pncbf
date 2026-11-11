"""Command-line runs for the quadrotor review notebook (run from scripts/quad3d).

  python review_tools/quad_review_runner.py mc   <eps,..> [N]   # rollouts  -> mc_results/review/quad_review_eps_*.pkl
  python review_tools/quad_review_runner.py eval <eps,..> [N]   # certificate check -> quad_review_eval_eps_*.pkl
  python review_tools/quad_review_runner.py summary
  python review_tools/quad_review_runner.py timing              # per-step time, original vs PGD DMR-CBF (Table II)

Executes the notebook's cells from build_quad_review_nb.py (without their run loops). Never overwrites results.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '.')
from build_quad_review_nb import CELLS

G_ = {'__name__': 'quad_review'}
summary_src = None
for t, s in CELLS:
    if t != 'code':
        continue
    if 'SLACK_TOL' in s:
        summary_src = s
        break
    exec(s.split('for _eps in EPS_LIST')[0], G_)

import numpy as np, jax, jax.numpy as jnp
def time_policies(names, x0, b0, n_repeats=5):
    """Per-step time as in the compare notebooks' Table II cell: JIT warm-up, then mean over full rollouts / T."""
    import time
    print(f'  devices: {jax.devices()}', flush=True)
    for name in names:
        rollout = G_['_make_rollout_logged'](G_['STEP_FNS'][name])
        jax.block_until_ready(rollout(x0, b0))
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(rollout(x0, b0))
            times.append(time.perf_counter() - t0)
        T = G_['T']
        print(f'  {name:<18} {np.mean(times)*1e3:8.1f} ms +- {np.std(times)*1e3:.1f} ms  '
              f'({np.mean(times)/T*1e3:.3f} ms/step, T={T})', flush=True)

mode = sys.argv[1]
if mode in ('mc', 'eval'):
    eps_list = [float(e) for e in sys.argv[2].split(',')]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else G_['N_MC']
    for eps in eps_list:
        (G_['run_eps'] if mode == 'mc' else G_['run_eval'])(eps, n)
if mode == 'summary':
    exec(summary_src, G_)
if mode == 'timing':   # python review_tools/quad_review_runner.py timing
    Q, E = G_['task'], G_['STATE_EPS']   # X0_DEMO / BIAS_DEMO of quad3d_cbf_compare.ipynb's timing cell
    x0 = jnp.zeros(G_['NX']).at[Q.PX].set(-1.8).at[Q.PZ].set(G_['Z_HOVER'] - 0.1).at[Q.PY].set(0.2)
    b0 = jnp.zeros(G_['NX']).at[Q.PX].set(-E[Q.PX]).at[Q.PZ].set(-E[Q.PZ])
    time_policies(['DMR-CBF (orig.)', 'PGD DMR-CBF'], x0, b0)
