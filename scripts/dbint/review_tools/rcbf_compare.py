"""Compare review R-CBF-QP runs with the paper's cached R-CBF-QP outcomes on the same trajectories.
Usage (from scripts/dbint): python -I review_tools/rcbf_compare.py <N> <eps,..>"""
import pickle, sys
from pathlib import Path
import numpy as np
D = Path('../../runs/nmr_cbf_dbint2d/policies/obs0.25/mc_results/full_scaling2x1_combined')
N = int(sys.argv[1])
for eps in [float(e) for e in sys.argv[2].split(',')]:
    cache = pickle.load(open(D / f'mc_eps_{eps:.3f}.pkl', 'rb'))
    line = f'eps {eps:.1f}'
    for tag, name in [('orig', 'R-CBF-QP (orig)'), ('nofeasfb', 'R-CBF-QP (no-feas fallback)')]:
        p = D / f'review_rcbf_{tag}_eps_{eps:.3f}_N{N}.pkl'
        if not p.exists():
            continue
        r = pickle.load(open(p, 'rb')); idx = r['eval_idx']; res = r['results'][name]
        coll, reached, t = res['coll'], res['reached'], res['t_reach']
        fb = np.asarray(res['logs'])[..., 0]
        ok = reached & ~coll
        line += (f' | {tag}: unsafe {coll.mean():5.1%} timeout {(~reached & ~coll).mean():5.1%} '
                 f'time {np.nanmean(np.asarray(t)[ok]):.2f}s fallback on {fb.mean():5.1%} of steps')
    c_coll = np.asarray(cache['violation_array']['R-CBF-QP'], bool)[idx]
    line += f' | paper cache (same {N}): unsafe {c_coll.mean():5.1%}'
    if (D / f'review_rcbf_orig_eps_{eps:.3f}_N{N}.pkl').exists():
        o = pickle.load(open(D / f'review_rcbf_orig_eps_{eps:.3f}_N{N}.pkl', 'rb'))['results']['R-CBF-QP (orig)']['coll']
        line += f', identical outcomes {np.mean(o == c_coll):.0%}'
    print(line)
