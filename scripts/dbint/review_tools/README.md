# Review experiments: certified DMR-CBF (L-CSS 26-5080, R1 Comments 2–3)

All variants enforce $a + b(\hat x)u \ge \bar G(\hat x)$ over $\bar X \cap C$ with a hard QP (slack only when infeasible).
They differ only in how the worst-case terms $a$ and $\bar G$ are computed.

| Policy name | Inner solver | Results files (in `runs/nmr_cbf_dbint2d/policies/obs0.25/mc_results/full_scaling2x1_combined/`) |
|---|---|---|
| `Certified DMR-CBF` | exact (velocity closed form + position grid; dbint-specific) | `review_cert_eps_*_N1000.pkl`, `review_cert_eps_0.300_N200.pkl` (also has the original DMR-CBF) |
| `Taylor DMR-CBF` | 2nd-order Taylor bound, single box | `review_taylor_eps_*_N200.pkl` |
| `Taylor DMR-CBF` | Taylor, position split 2×2 | `review_taylor_p2121_eps_*_N200.pkl` |
| `Taylor DMR-CBF` | Taylor, position split 4×4 | `review_taylor_p4141_eps_*_N200.pkl` |
| `PGD DMR-CBF` | paper's projected gradient (4 restarts × 25 steps) | `review_pgdcert_eps_*_N{200,1000}.pkl` |

Each file holds trajectories, controller logs (`a`, `Gbar`, slack) and the exact-term evaluation
(`m7`, `G_u`, `M`, `a_fine`) used to check Prop. 1.

## Files

- `build_review_nb.py` — regenerates `../dbint2d_review.ipynb` (`python build_review_nb.py ../dbint2d_review.ipynb`, from the repo root).
  This overwrites the notebook and its outputs.
- `spaghetti_cells.py`, `taylor_cells.py`, `pgd_cells.py` — cell sources used by the builder and the runner.
- `review_runner.py` — command-line runs (see its docstring). Run from `scripts/dbint`. It never overwrites existing results.
- `review_summary.py <tag>` — table versus the exact (grid) controller on the same trajectories, for example `taylor_p4141` or `pgdcert`.
