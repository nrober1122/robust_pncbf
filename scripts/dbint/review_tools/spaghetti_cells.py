VAR_MD = r'''## Variant comparison

Every variant enforces $a + b(\hat x)u \ge \bar G$ over $\bar X \cap C$ with a hard QP; they differ only in the inner solver.
The runs are made with `review_tools/review_runner.py` (see `review_tools/README.md`) and loaded here. Prop. 1 and $M$ are
always evaluated with the exact terms. `VARIANTS` maps each policy name to its results-file prefix; for each ε the
N = 1000 file is used when it exists, otherwise N = 200. The original (uncertified) DMR-CBF comes from the compare
notebook's MC cache (`PSHR-CBF`).'''

VAR_CODE = r'''VARIANTS = {   # policy name -> results-file prefix
    'Certified DMR-CBF': 'review_cert',           # exact: velocity closed form + position grid (dbint only)
    'Taylor DMR-CBF':    'review_taylor_p4141',   # second-order Taylor bound, 4 x 4 position partition
    'PGD DMR-CBF':       'review_pgdcert',        # the paper's projected gradient (4 restarts x 25 steps)
}

def _variant_file(policy, eps):
    for n in (N_MC, 200):
        p = _mc_results_dir / f'{VARIANTS[policy]}_eps_{eps:.3f}_N{n}.pkl'
        if p.exists():
            return p
    return None

def _load_variant(policy, eps, idx=None):
    # Results of `policy` at eps, restricted to the MC sample indices idx (all evaluated samples if None)
    with open(_variant_file(policy, eps), 'rb') as f:
        R = pickle.load(f)
    r = R['results'][policy]
    if idx is None:
        sel = np.arange(len(R['eval_idx']))
    else:
        pos = {int(j): k for k, j in enumerate(R['eval_idx'])}
        sel = np.array([pos[int(j)] for j in idx])
    out = {k: r[k][sel] for k in ('xs', 'coll', 't_reach', 'reached', 'm7', 'G_u', 'M', 'a_fine')}
    out['logs'] = r['logs'][sel]
    return out

print(f'{"eps":>4}  {"policy":<18} {"N":>5} {"coll":>5} {"t_goal":>7} {"Prop. 1":>8} {"worst M":>8} {"slack":>6}')
for _e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    _mc = _mc_results_dir / f'mc_eps_{_e:.3f}.pkl'
    if _mc.exists():
        with open(_mc, 'rb') as _f:
            _m = pickle.load(_f)
        _v = np.asarray(_m['violation_array']['PSHR-CBF'], bool)
        _t = np.asarray(_m['reach_time_array']['PSHR-CBF'])
        print(f'{_e:4.1f}  {"DMR-CBF (orig.)":<18} {len(_v):5d} {_v.sum():5d} {np.nanmean(_t[~_v]):7.2f}')
    for _pol in VARIANTS:
        if _variant_file(_pol, _e) is None:
            continue
        _r = _load_variant(_pol, _e)
        _p1 = (_r['m7'] - _r['G_u']) >= 0
        print(f'{"":4}  {_pol:<18} {len(_r["coll"]):5d} {_r["coll"].sum():5d} {np.nanmean(_r["t_reach"][~_r["coll"]]):7.2f} '
              f'{_p1.mean():8.3%} {_r["M"].min():+8.3f} {(_r["logs"][:, :-1, 2] > SLACK_TOL).any(1).sum():6d}')

# Inner-solver error of each variant's own a against the exact a (> 0: optimistic)
for _pol in VARIANTS:
    _errs = {e: _load_variant(_pol, e) for e in [0.1, 0.3, 0.5] if _variant_file(_pol, e) is not None}
    for _e, _r in _errs.items():
        _d = _r['logs'][:, :-1, 0] - _r['a_fine']
        print(f'  {_pol:<18} eps={_e:.1f}: a error p99 {np.percentile(_d, 99):+.3f}, max {_d.max():+.3f}, '
              f'optimistic on {np.mean(_d > 1e-3):.1%} of steps')
'''

SPAG_MD = r'''## Spaghetti plots across ε (one variant)

One panel per ε for `SPAG_POLICY` (results files from `VARIANTS`), styled like the paper's Fig. 2: safe traces
coloured by speed (shared scale), colliding traces in black. `SPAG_N = 200` draws the paper's spaghetti subset (seed 42);
`SPAG_N = N_MC` draws all 1000.'''

SPAG_CODE = r'''import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

SPAG_EPS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SPAG_N      = 200                     # 200 = paper's spaghetti subset; N_MC = all
SPAG_POLICY = 'PGD DMR-CBF'          # any key of VARIANTS
SPAG_YLIM   = None                    # None: fit all drawn trajectories (at least +-2 m, the paper's range)

_spag_idx = (np.random.default_rng(42).choice(N_MC, size=SPAG_N, replace=False)
             if SPAG_N < N_MC else np.arange(N_MC))

def _load_cert(eps, idx, policy=None):
    d = _load_variant(policy or SPAG_POLICY, eps, idx)
    return {'xs': d['xs'], 'coll': d['coll'], 't': d['t_reach']}

def _draw_spaghetti(ax, xs, coll, speed_norm, cmap):
    spd     = np.hypot(xs[:, :, 1], xs[:, :, 3])
    seg_spd = 0.5 * (spd[:, :-1] + spd[:, 1:])
    if (~coll).any():
        pts  = np.stack([xs[~coll, :, 0], xs[~coll, :, 2]], axis=2)
        segs = np.concatenate([pts[:, :-1, None, :], pts[:, 1:, None, :]], axis=2).reshape(-1, 2, 2)
        rgba = cmap(speed_norm(seg_spd[~coll].ravel()))
        rgba[:, 3] = 0.8
        ax.add_collection(LineCollection(segs, colors=rgba, lw=0.7, zorder=3))
    if coll.any():
        ax.plot(xs[coll, :, 0].T, xs[coll, :, 2].T, color='k', lw=0.6, alpha=0.4, zorder=4)
    ax.add_patch(Circle((0, 0), R_OBS, facecolor='0.45', edgecolor='0.4', alpha=0.55, zorder=5))
    ax.scatter(*np.asarray(GOAL_MC)[[0, 2]], s=160, color='gold', edgecolors='k', marker='*', zorder=6)

_LEGEND = [mpatches.Patch(color='k', label='unsafe'),
           Line2D([0], [0], ls='None', marker='*', color='gold', mec='k', ms=11, label='goal')]

_spag = {e: _load_cert(e, _spag_idx) for e in SPAG_EPS if _variant_file(SPAG_POLICY, e) is not None}
_safe_spd = np.concatenate([np.hypot(d['xs'][~d['coll'], :, 1], d['xs'][~d['coll'], :, 3]).ravel()
                            for d in _spag.values() if (~d['coll']).any()])
_speed_norm = plt.Normalize(0, float(np.percentile(_safe_spd, 99)))
_cmap = plt.cm.plasma
_ylim = SPAG_YLIM or max(2.0, 0.1 * np.ceil(10 * max(np.abs(d['xs'][:, :, 2]).max() for d in _spag.values()) + 1))

with mpl.rc_context({'font.family': 'serif', 'font.size': 12, 'text.usetex': True}):
    _ncol = 3
    _nrow = int(np.ceil(len(_spag) / _ncol))
    fig, axes = plt.subplots(_nrow, _ncol, figsize=(11.5, 3.3 * _nrow * _ylim / 2.0), layout='constrained',
                             sharex=True, sharey=True, dpi=200, squeeze=False)
    for ax in axes.flat[len(_spag):]:
        ax.set_visible(False)
    for k, (ax, (_e, d)) in enumerate(zip(axes.flat, _spag.items())):
        _draw_spaghetti(ax, d['xs'], d['coll'], _speed_norm, _cmap)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-_ylim, _ylim); ax.set_aspect('equal')
        ax.set_title(rf'$\varepsilon = {_e:.1f}$', fontsize=15)
        _ok = ~d['coll'] & ~np.isnan(d['t'])
        ax.text(0.02, 0.03, rf'coll.\ {d["coll"].sum()}/{len(d["xs"])}, '
                            rf'$\bar t_{{\mathrm{{goal}}}}$ {np.nanmean(d["t"][_ok]):.2f}\,s',
                transform=ax.transAxes, fontsize=9, zorder=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
        if k % _ncol == 0:
            ax.set_ylabel(r'$p_y$ (m)', fontsize=14)
        if k >= len(_spag) - _ncol:
            ax.set_xlabel(r'$p_x$ (m)', fontsize=14)
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_speed_norm)
    _sm.set_array([])
    fig.colorbar(_sm, ax=axes, pad=0.01, shrink=0.9).set_label(r'speed (m/s)', fontsize=14)
    axes.flat[0].legend(handles=_LEGEND, fontsize=10, loc='upper left')
    fig.suptitle(rf'\textbf{{{SPAG_POLICY}}} ({len(_spag_idx)}/{N_MC} trajectories per $\varepsilon$)', fontsize=16)
    plt.show()
'''

CMP_MD = r'''## Fig. 2 with the certified DMR-CBF

The paper's spaghetti figure, redrawn from the compare notebook's caches (`spaghetti_eps_*.pkl`, the same 200 MC samples),
with `CMP_POLICY` (a key of `VARIANTS`) in place of the original DMR-CBF (`PSHR-CBF`). Every other panel, the speed colour scale
(computed from the original policies, as in the compare notebook) and the layout are unchanged. One figure per ε with a cache.'''

CMP_CODE = r'''CMP_EPS    = [0.1, 0.3, 0.5]       # eps values with a spaghetti_eps_*.pkl cache
CMP_POLICY = 'PGD DMR-CBF'         # drawn in the DMR-CBF panel
CMP_ORDER = ['GUARDIAN', 'CBF', 'R-CBF (0.2)', 'R-CBF (1.0)',
             'Duality CBF', 'R-CBF-QP', 'DMR-CBF*', 'NMR-CBF (RL)']
CMP_TITLES = {'DMR-CBF*': r'\textbf{DMR-CBF}', 'NMR-CBF (RL)': r'\textbf{NMR-CBF}'}   # DMR-CBF* = CMP_POLICY

_h_sub = jax.jit(jax.vmap(jax.vmap(h_raw)))

def _mean_t_goal(xs, coll):
    # Mean first time in the goal region (p_x >= 0, speed <= VEL_TOL) over collision-free trajectories that get there
    in_goal = (xs[:, :, 0] >= 0) & (np.hypot(xs[:, :, 1], xs[:, :, 3]) <= VEL_TOL)
    ok      = in_goal.any(1) & ~coll
    return np.argmax(in_goal[ok], 1).mean() * DT if ok.any() else np.nan

def draw_cmp(_e, policy=None, suptitle=True, annotate=True, save=None):
    """Fig. 2-style 2x4 spaghetti panels at eps=_e with `policy` in the DMR-CBF panel; returns the collision counts."""
    policy = policy or CMP_POLICY
    _cache = _mc_results_dir / f'spaghetti_eps_{_e:.3f}.pkl'
    if not (_cache.exists() and _variant_file(policy, _e) is not None):
        print(f'  eps={_e:.2f}: missing {_cache.name} or {policy} results, skipping')
        return None
    with open(_cache, 'rb') as _f:
        _sp = pickle.load(_f)
    _trajs = {n: np.asarray(v) for n, v in _sp['sub_trajs'].items()}
    _viols = {n: np.asarray(_h_sub(jnp.asarray(xs))).max(axis=(1, 2)) > 0 for n, xs in _trajs.items()}

    # Speed scale from the original policies, exactly as the compare notebook computes it
    _norm = plt.Normalize(0, float(np.percentile(np.concatenate(
        [np.hypot(_trajs[n][~_viols[n], :, 1], _trajs[n][~_viols[n], :, 3]).ravel()
         for n in _sp['plot_order'] if (~_viols[n]).any()]), 99)))

    _c = _load_cert(_e, np.asarray(_sp['sub_idx']), policy)
    _trajs['DMR-CBF*'], _viols['DMR-CBF*'] = _c['xs'], _c['coll']
    assert _trajs['DMR-CBF*'].shape == _trajs['PSHR-CBF'].shape
    assert np.allclose(_c['xs'][:, 0], _trajs['PSHR-CBF'][:, 0]), 'initial states differ from the cache'

    _ylim = max(2.0, 0.1 * np.ceil(10 * max(np.abs(_trajs[n][:, :, 2]).max() for n in CMP_ORDER) + 1))
    with mpl.rc_context({'font.family': 'serif', 'font.size': 12, 'text.usetex': True}):
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.4 * _ylim / 2.0), layout='constrained',
                                 sharex=True, sharey=True, dpi=200)
        for k, (ax, _name) in enumerate(zip(axes.flat, CMP_ORDER)):
            _draw_spaghetti(ax, _trajs[_name], _viols[_name], _norm, _cmap)
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-_ylim, _ylim); ax.set_aspect('equal')
            ax.set_title(CMP_TITLES.get(_name, rf'\textbf{{{_name}}}'), fontsize=18)
            if annotate:
                _tg = _mean_t_goal(_trajs[_name], _viols[_name])
                ax.text(0.02, 0.03, rf'coll.\ {_viols[_name].sum()}/{len(_viols[_name])}, $\bar t_{{\mathrm{{goal}}}}$ '
                                    + (rf'{_tg:.2f}\,s' if np.isfinite(_tg) else '--'),
                        transform=ax.transAxes, fontsize=13, zorder=7,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
            if k % 4 == 0:
                ax.set_ylabel(r'$p_y$ (m)', fontsize=16)
            if k >= 4:
                ax.set_xlabel(r'$p_x$ (m)', fontsize=16)
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
        _sm.set_array([])
        fig.colorbar(_sm, ax=axes, pad=0.01, shrink=0.92).set_label(r'speed (m/s)', fontsize=16)
        axes.flat[0].legend(handles=_LEGEND, fontsize=14, loc='upper left')
        if suptitle:
            fig.suptitle(rf'$\varepsilon = {_e:.1f}$ (DMR-CBF panel: {policy})', fontsize=18)
        if save is not None:
            fig.savefig(save, dpi=400, bbox_inches='tight')
        plt.show()
    return {n: int(_viols[n].sum()) for n in CMP_ORDER}

for _e in CMP_EPS:
    _counts = draw_cmp(_e)
    if _counts is not None:
        print(f'  eps={_e:.1f} collisions: ' + ', '.join(f'{n} {c}' for n, c in _counts.items()))
'''
