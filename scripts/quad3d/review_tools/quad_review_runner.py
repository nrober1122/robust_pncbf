"""Command-line runs for the quadrotor review notebook (run from scripts/quad3d).

  python review_tools/quad_review_runner.py mc   <eps,..> [N]   # rollouts  -> mc_results/review/quad_review_eps_*.pkl
  python review_tools/quad_review_runner.py eval <eps,..> [N]   # certificate check -> quad_review_eval_eps_*.pkl
  python review_tools/quad_review_runner.py summary

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

mode = sys.argv[1]
if mode in ('mc', 'eval'):
    eps_list = [float(e) for e in sys.argv[2].split(',')]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else G_['N_MC']
    for eps in eps_list:
        (G_['run_eps'] if mode == 'mc' else G_['run_eval'])(eps, n)
if mode == 'summary':
    exec(summary_src, G_)
