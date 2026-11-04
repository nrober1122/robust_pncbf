"""Summarise review_<TAG>_eps_*_N200.pkl (TAG: taylor, taylor_p4141, pgdcert, ...) vs the grid-certified DMR-CBF on the same trajectories."""
import pickle
from pathlib import Path
import sys
import numpy as np
TAG = sys.argv[1] if len(sys.argv) > 1 else 'taylor'
POL = 'PGD DMR-CBF' if TAG.startswith('pgd') else 'Taylor DMR-CBF'

D = Path('/Users/nrober1122/projects/nmr-cbfs/robust_pncbf/runs/nmr_cbf_dbint2d/policies/obs0.25/mc_results/full_scaling2x1_combined')
TOL = 1e-5
print(f'{"eps":>4} | {"coll":>5} {"timeout":>7} {"t_goal":>6} {"grid t":>6} {"paired":>7} | {"slack trajs":>11} '
      f'{"Prop1 steps":>11} {"Prop1 trajs":>11} {"worst M":>8} | {"min clear":>9} {"grid clr":>8}')
for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    p = D / f'review_{TAG}_eps_{eps:.3f}_N200.pkl'
    if not p.exists():
        continue
    R = pickle.load(open(p, 'rb'))
    r, idx = R['results'][POL], R['eval_idx']
    C = pickle.load(open(D / f'review_cert_eps_{eps:.3f}_N1000.pkl', 'rb'))['results']['Certified DMR-CBF']
    tg, tc = r['t_reach'], C['t_reach'][idx]
    ok = ~np.isnan(tg) & ~np.isnan(tc) & ~r['coll']
    prop1 = (r['m7'] - r['G_u']) >= 0
    slack = (r['logs'][:, :-1, 2] > TOL)
    clr = np.hypot(r['xs'][:, :, 0], r['xs'][:, :, 2]).min() - 0.25
    clr_c = np.hypot(C['xs'][idx][:, :, 0], C['xs'][idx][:, :, 2]).min() - 0.25
    print(f'{eps:4.1f} | {r["coll"].sum():5d} {np.mean(~r["reached"] & ~r["coll"]):7.1%} {np.nanmean(tg[~r["coll"]]):6.2f} '
          f'{np.nanmean(tc):6.2f} {np.mean(tg[ok] / tc[ok]) - 1:+7.1%} | {slack.any(1).sum():11d} '
          f'{prop1.mean():11.2%} {prop1.all(1).mean():11.1%} {r["M"].min():+8.3f} | {clr:9.3f} {clr_c:8.3f}')
