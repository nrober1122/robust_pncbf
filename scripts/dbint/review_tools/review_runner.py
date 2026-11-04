"""Review experiments for the certified DMR-CBF (run from scripts/dbint).

  python review_tools/review_runner.py accuracy                 # Taylor bound vs exact terms on visited states
  python review_tools/review_runner.py parts                    # Taylor partition sizes: tightness and time
  python review_tools/review_runner.py mc <eps,..> <N> [parts]  # Taylor DMR-CBF  -> review_taylor[_p....]_eps_*.pkl
  python review_tools/review_runner.py mcpgd <eps,..> <N>       # PGD DMR-CBF     -> review_pgdcert_eps_*.pkl
  python review_tools/review_runner.py mcdmr <eps,..> <N>       # original DMR-CBF (paper's (7)) -> review_dmr_eps_*.pkl
  python review_tools/review_runner.py mcrcbf orig|fix <eps,..> <N>  # R-CBF-QP as in the paper, or without the minimum-gain fallback

Runs never overwrite an existing results file.
"""
import sys, pickle, time
SP = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, '.')

src = open(f'{SP}/build_review_nb.py').read().split('meta = json.load(')[0]
src = src.split('sys.path.insert(0, __import__')[0]
ns = {}
exec(src, ns)
cells = [s for t, s in ns['CELLS'] if t == 'code']
G_ = {'__name__': 'proto'}
for s in cells:
    if 'for _eps in EPS_LIST' in s:
        s = s.split('for _eps in EPS_LIST')[0]
    if 'def summarize' in s:
        break
    exec(s, G_)
from taylor_cells import TAYLOR_CODE
exec(TAYLOR_CODE, G_)
from pgd_cells import PGD_CODE
exec(PGD_CODE, G_)
from rcbf_cells import RCBF_CODE
exec(RCBF_CODE, G_)
exec(r'''
def set_eps(eps):
    global STATE_EPS, EPS, biases, _GRID, _GRID_F
    STATE_EPS      = EPS_SHAPE * eps
    EPS            = float(eps)
    biases         = jr.uniform(rng_bias, (N_MC, 4), minval=-STATE_EPS, maxval=STATE_EPS)
    _GRID, _GRID_F = _grid(GRID_K), _grid(K_FINE)
''', G_)

import numpy as np, jax, jax.numpy as jnp, jax.random as jr
mode = sys.argv[1]

def run_new(eps_list, n_eval, policy, prefix):
    """Run `policy` for each eps and save to <prefix>_eps_{eps}_N{n}.pkl, skipping files that already exist."""
    G_['RUN_POLICIES'] = [policy]
    G_['N_EVAL'] = n_eval
    G_['_save_path'] = lambda e, n: G_['_mc_results_dir'] / f'{prefix}_eps_{e:.3f}_N{n}.pkl'
    for eps in eps_list:
        if G_['_save_path'](eps, n_eval).exists():
            print(f'  eps={eps:.2f}: {G_["_save_path"](eps, n_eval).name} exists, not overwriting', flush=True)
            continue
        G_['run_eps'](eps)

if mode == 'mcpgd':
    run_new([float(e) for e in sys.argv[2].split(',')], int(sys.argv[3]), 'PGD DMR-CBF', 'review_pgdcert')

if mode == 'mcdmr':
    run_new([float(e) for e in sys.argv[2].split(',')], int(sys.argv[3]), 'DMR-CBF', 'review_dmr')

if mode == 'mcrcbf':
    variant = {'orig': ('R-CBF-QP (orig)', 'review_rcbf_orig'),
               'fix': ('R-CBF-QP (no-feas fallback)', 'review_rcbf_nofeasfb')}[sys.argv[2]]
    run_new([float(e) for e in sys.argv[3].split(',')], int(sys.argv[4]), *variant)

if mode == 'accuracy':
    for eps in [0.1, 0.3, 0.5]:
        G_['set_eps'](eps)
        with open(G_['_save_path'](eps, 1000), 'rb') as f:
            R = pickle.load(f)
        r = R['results']['Certified DMR-CBF']
        xh = r['xhats'].reshape(-1, 4)
        idx = np.random.default_rng(0).choice(len(xh), 3000, replace=False)
        xh = jnp.asarray(xh[idx])
        keys = jr.split(jr.PRNGKey(1), len(xh))
        ex = jax.jit(jax.vmap(lambda x: G_['cert_terms'](x, G_['_GRID_F'], G_['K_REF_F'], G_['N_REF_F'])))
        ty = jax.jit(jax.vmap(lambda x, k: G_['taylor_terms'](x, k)))
        tyu = jax.jit(jax.vmap(lambda x, k: G_['taylor_terms'](x, k, restrict_C=False)))
        def pgd(x, k):
            lo, hi = G_['_box_bounds'](x)
            v, _ = G_['proj_grad_max_box'](lambda s: (jax.jacobian(G_['B_hocbf'])(s) @ G_['f'](s)
                                                     + G_['ALPHA_QP'] * G_['B_hocbf'](s))[0],
                                           lo, hi, n_restarts=4, n_steps=25, lr=0.05, key=k)
            return -v
        pg = jax.jit(jax.vmap(pgd))
        a_ex, G_ex = map(np.asarray, ex(xh))
        (a_ty, G_ty), (a_tu, G_tu) = [tuple(np.asarray(z)[:, 0] for z in fn(xh, keys)) for fn in (ty, tyu)]
        a_pg = np.asarray(pg(xh, keys))
        t0 = time.time(); jax.block_until_ready(ty(xh, keys)); dt = time.time() - t0
        q = lambda z: f'med {np.median(z):+.3f} p95 {np.percentile(z, 95):+.3f} max {z.max():+.3f}'
        print(f'=== eps={eps}: {len(xh)} visited states (certified MC), taylor {1e6 * dt / len(xh):.0f} us/state batched')
        print(f'  a:    exact - taylor(C)   {q(a_ex - a_ty)}   unsound frac {np.mean(a_ty > a_ex + 1e-4):.4f}')
        print(f'        exact - taylor(box) {q(a_ex - a_tu)}   unsound frac {np.mean(a_tu > a_ex + 1e-4):.4f}')
        print(f'        pgd - exact         {q(a_pg - a_ex)}   (> 0: PGD optimistic)')
        print(f'  Gbar: taylor(C) - grid    {q(G_ty - G_ex)}   below-grid frac {np.mean(G_ty < G_ex - 1e-4):.4f}')
        print(f'        taylor(box) - grid  {q(G_tu - G_ex)}')
        print(f'  total margin (Gbar - a_lo): grid {np.median(G_ex - a_ex):+.3f}, taylor(C) {np.median(G_ty - a_ty):+.3f}')

if mode == 'parts':
    PARTS = [(1, 1, 1, 1), (2, 1, 2, 1), (3, 1, 3, 1), (2, 2, 2, 2), (4, 1, 4, 1)]
    for eps in [0.3, 0.5]:
        G_['set_eps'](eps)
        with open(G_['_save_path'](eps, 1000), 'rb') as f:
            R = pickle.load(f)
        xh = R['results']['Certified DMR-CBF']['xhats'].reshape(-1, 4)
        xh = jnp.asarray(xh[np.random.default_rng(0).choice(len(xh), 2000, replace=False)])
        keys = jr.split(jr.PRNGKey(1), len(xh))
        ex = jax.jit(jax.vmap(lambda x: G_['cert_terms'](x, G_['_GRID_F'], G_['K_REF_F'], G_['N_REF_F'])))
        a_ex, G_ex = map(np.asarray, ex(xh))
        print(f'=== eps={eps}: {len(xh)} visited states')
        for parts in PARTS:
            ty = jax.jit(jax.vmap(lambda x, k: G_['taylor_terms'](x, k, parts=parts)))
            a_ty, G_ty = (np.asarray(z)[:, 0] for z in ty(xh, keys))
            one = jax.jit(lambda x, k: G_['taylor_terms'](x, k, parts=parts))
            jax.block_until_ready(one(xh[0], keys[0]))
            t0 = time.time()
            for i in range(50):
                jax.block_until_ready(one(xh[i], keys[i]))
            dt = (time.time() - t0) / 50
            q = lambda z: f'med {np.median(z):+.3f} p95 {np.percentile(z, 95):+.3f}'
            print(f'  parts {parts} ({int(np.prod(parts)):2d} boxes, {1e3 * dt:5.1f} ms/step): '
                  f'a gap {q(a_ex - a_ty)} unsound {np.mean(a_ty > a_ex + 1e-4):.4f} | '
                  f'Gbar gap {q(G_ty - G_ex)} | margin Gbar-a_lo {np.median(G_ty - a_ty):+.3f} '
                  f'(grid {np.median(G_ex - a_ex):+.3f})')

if mode == 'mc':
    if len(sys.argv) > 4:
        G_['TAY_PARTS'] = tuple(int(c) for c in sys.argv[4].split(','))
    tag = '' if G_['TAY_PARTS'] == (1, 1, 1, 1) else '_p' + ''.join(map(str, G_['TAY_PARTS']))
    eps_list = [float(e) for e in sys.argv[2].split(',')]
    n_eval = int(sys.argv[3])
    run_new(eps_list, n_eval, 'Taylor DMR-CBF', f'review_taylor{tag}')
