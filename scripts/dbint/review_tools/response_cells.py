RESP_MD = r'''# Figures for the response letter (R1 Comment 2)

The original DMR-CBF (the paper's constraint (7), `review_dmr_eps_*_N200.pkl`) against the revised DMR-CBF
($a + b(\hat x)u \ge \bar G$ over $\bar X \cap C$ with projected gradient, `PGD DMR-CBF`), on the paper's 200-trajectory
spaghetti subset. Both use the same initial states and biases. $M(u^\star_t) = \min_{s} [a(s) + b(s)u^\star_t]$ is
evaluated with the exact terms, so $M < 0$ means the CBF condition is not certified at that step. Figures are saved to
`rebuttal/figs/`.'''

RESP_CODE = r'''RESP_EPS      = [0.1, 0.3, 0.5]                       # columns of the M(u*) figure
RESP_SPAG_EPS = [0.1, 0.3, 0.5]                       # columns of the spaghetti figure
RESP_ROWS     = {'DMR-CBF': r'\textbf{Original}', 'PGD DMR-CBF': r'\textbf{Revised}'}
RESP_M_YLIM   = (-0.4, 1.5)
RESP_T_MAX    = 15.0                                  # s, x-axis of the M(u*) figure
RESP_FIG_DIR  = Path('../../rebuttal/figs')
RESP_FIG_DIR.mkdir(parents=True, exist_ok=True)
_RESP_RC = {'font.family': 'serif', 'font.size': 12, 'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}'}

VARIANTS['DMR-CBF'] = 'review_dmr'                      # original constraint (7), N = 200 subset
_resp_idx = np.random.default_rng(42).choice(N_MC, size=200, replace=False)   # paper's spaghetti subset
_resp = {(p, e): _load_variant(p, e, _resp_idx) for p in RESP_ROWS for e in sorted(set(RESP_EPS + RESP_SPAG_EPS))
         if _variant_file(p, e) is not None}
print('missing:', [k for k in ((p, e) for p in RESP_ROWS for e in RESP_SPAG_EPS) if k not in _resp] or 'none')

def _draw_M(ax, M):
    t   = np.arange(M.shape[1]) * DT
    pts = np.stack([np.broadcast_to(t, M.shape), np.clip(M, *RESP_M_YLIM)], axis=2)
    seg = np.concatenate([pts[:, :-1, None, :], pts[:, 1:, None, :]], axis=2).reshape(-1, 2, 2)
    neg = (np.minimum(M[:, :-1], M[:, 1:]) < 0).ravel()
    ax.add_collection(LineCollection(seg[~neg], colors='tab:blue', lw=0.5, alpha=0.25, zorder=3))
    ax.add_collection(LineCollection(seg[neg], colors='tab:red', lw=0.9, alpha=0.8, zorder=4))
    ax.axhline(0, color='k', lw=0.8, ls='--', zorder=5)
    ax.set_xlim(0, RESP_T_MAX); ax.set_ylim(*RESP_M_YLIM)
    ax.text(0.98, 0.95, rf'$M < 0$ on {np.mean(M < 0):.1%} of steps'.replace('%', r'\%'),
            transform=ax.transAxes, ha='right', va='top', fontsize=13, zorder=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

with mpl.rc_context(_RESP_RC):
    fig, axes = plt.subplots(len(RESP_ROWS), len(RESP_EPS), figsize=(4.2 * len(RESP_EPS), 2.9 * len(RESP_ROWS)),
                             layout='constrained', sharex=True, sharey=True, dpi=200, squeeze=False)
    for i, (pol, row) in enumerate(RESP_ROWS.items()):
        for j, e in enumerate(RESP_EPS):
            ax = axes[i, j]
            if (pol, e) in _resp:
                _draw_M(ax, _resp[(pol, e)]['M'])
            if i == 0:
                ax.set_title(rf'$\varepsilon = {e:.1f}$', fontsize=16)
            if j == 0:
                ax.set_ylabel(row + '\n' + r'$M(\boldsymbol{u}^\star_t)$', fontsize=15)
            if i == len(RESP_ROWS) - 1:
                ax.set_xlabel(r'$t$ (s)', fontsize=15)
    fig.savefig(RESP_FIG_DIR / 'dbint_M_original_vs_revised.pdf', bbox_inches='tight')
    plt.show()

_resp_xs  = [_resp[k]['xs'] for k in _resp]
_resp_ok  = [~_resp[k]['coll'] for k in _resp]
_resp_norm = plt.Normalize(0, float(np.percentile(np.concatenate(
    [np.hypot(x[o, :, 1], x[o, :, 3]).ravel() for x, o in zip(_resp_xs, _resp_ok) if o.any()]), 99)))
_resp_ylim = max(2.0, 0.1 * np.ceil(10 * max(np.abs(x[:, :, 2]).max() for x in _resp_xs) + 1))

with mpl.rc_context(_RESP_RC):
    fig, axes = plt.subplots(len(RESP_ROWS), len(RESP_SPAG_EPS),
                             figsize=(3.3 * len(RESP_SPAG_EPS) + 1.2, 2.45 * len(RESP_ROWS) * _resp_ylim / 2.0 + 0.6),
                             layout='constrained', sharex=True, sharey=True, dpi=200, squeeze=False)
    for i, (pol, row) in enumerate(RESP_ROWS.items()):
        for j, e in enumerate(RESP_SPAG_EPS):
            ax = axes[i, j]
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-_resp_ylim, _resp_ylim); ax.set_aspect('equal')
            if i == 0:
                ax.set_title(rf'$\varepsilon = {e:.1f}$', fontsize=16)
            if j == 0:
                ax.set_ylabel(row + '\n' + r'$p_y$ (m)', fontsize=15)
            if i == len(RESP_ROWS) - 1:
                ax.set_xlabel(r'$p_x$ (m)', fontsize=15)
            if (pol, e) not in _resp:
                continue
            d = _resp[(pol, e)]
            _draw_spaghetti(ax, d['xs'], d['coll'], _resp_norm, _cmap)
            _tg = _mean_t_goal(d['xs'], d['coll'])
            ax.text(0.03, 0.03, rf'$\bar t_{{\mathrm{{goal}}}}$ {_tg:.2f}\,s'
                                + (rf', coll.\ {d["coll"].sum()}' if d['coll'].any() else ''),
                    transform=ax.transAxes, fontsize=13, zorder=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_resp_norm)
    _sm.set_array([])
    fig.colorbar(_sm, ax=axes, pad=0.01, shrink=0.9).set_label(r'speed (m/s)', fontsize=15)
    fig.savefig(RESP_FIG_DIR / 'dbint_spaghetti_original_vs_revised.pdf', bbox_inches='tight')
    plt.show()

for e in RESP_SPAG_EPS:
    print(f'eps={e:.1f}: ' + ', '.join(
        f'{p}: M<0 {np.mean(_resp[(p, e)]["M"] < 0):6.2%}, min M {_resp[(p, e)]["M"].min():+.3f}, '
        f'coll {_resp[(p, e)]["coll"].sum()}, t_goal {_mean_t_goal(_resp[(p, e)]["xs"], _resp[(p, e)]["coll"]):.2f} s'
        for p in RESP_ROWS if (p, e) in _resp))
'''

RESP_BAR_MD = r'''## Outcome bars, original vs revised DMR-CBF (full 1000 samples)

The paper's Fig. 2 bars for the two DMR-CBF formulations only: the original from the compare notebook's MC caches
(`PSHR-CBF` in `mc_eps_*.pkl`), the revised from `review_pgdcert_eps_*_N1000.pkl`. The mean-time colour scale is the one
the compare notebook uses for Fig. 2 (Nominal's fastest mean time to the slowest of all approaches), so colours match the paper.'''

RESP_BAR_CODE = r'''RESP_BAR_EPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def _bar_outcomes_cache(m, name):
    viol = np.asarray(m['violation_array'][name], bool)
    rch  = np.asarray(m['reached_array'][name], bool) & ~viol
    t    = np.asarray(m['reach_time_array'][name])
    return rch.mean(), viol.mean(), (np.nanmean(t[rch]) if rch.any() else np.nan)

def _bar_outcomes_variant(policy, e):
    r   = _load_variant(policy, e)
    rch = r['reached'] & ~r['coll']
    return rch.mean(), r['coll'].mean(), (np.nanmean(r['t_reach'][rch]) if rch.any() else np.nan)

_bar_caches = {}
for e in RESP_BAR_EPS:
    with open(_mc_results_dir / f'mc_eps_{e:.3f}.pkl', 'rb') as f:
        _bar_caches[e] = pickle.load(f)
_bar = {'DMR-CBF':     [_bar_outcomes_cache(_bar_caches[e], 'PSHR-CBF') for e in RESP_BAR_EPS],
        'PGD DMR-CBF': [_bar_outcomes_variant('PGD DMR-CBF', e) for e in RESP_BAR_EPS]}
assert all(len(_load_variant('PGD DMR-CBF', e)['coll']) == N_MC for e in RESP_BAR_EPS)

# Fig. 2 colour scale: Nominal's fastest mean time to the slowest mean time of any approach (or of the revised DMR-CBF)
_bar_all = [_bar_outcomes_cache(m, n)[2] for m in _bar_caches.values() for n in m['violation_array']]
_bar_nom = [_bar_outcomes_cache(m, 'Nominal')[2] for m in _bar_caches.values()]
_bar_norm = plt.Normalize(np.nanmin(_bar_nom), np.nanmax(_bar_all + [o[2] for o in _bar['PGD DMR-CBF']]))
_bar_cmap = plt.cm.plasma_r

with mpl.rc_context(_RESP_RC):
    fig, axes = plt.subplots(1, len(RESP_ROWS), figsize=(4.6 * len(RESP_ROWS), 3.6), layout='constrained',
                             sharey=True, dpi=200)
    x = np.arange(len(RESP_BAR_EPS))
    for ax, (pol, title) in zip(axes, RESP_ROWS.items()):
        rch, uns, mt = (100 * np.array([o[0] for o in _bar[pol]]), 100 * np.array([o[1] for o in _bar[pol]]),
                        np.array([o[2] for o in _bar[pol]]))
        tmo = 100 - rch - uns
        ax.bar(x, rch, color=[_bar_cmap(_bar_norm(t)) if np.isfinite(t) else '0.85' for t in mt], zorder=3)
        ax.bar(x, tmo, bottom=rch, color='grey', zorder=3)
        ax.bar(x, uns, bottom=rch + tmo, color='k', zorder=3)
        for i, t in enumerate(mt):
            if np.isfinite(t) and rch[i] >= 3:
                ax.text(i, 1.5, rf'\textbf{{{t:.1f}s}}', ha='center', va='bottom', fontsize=12,
                        color='white' if t > 8.0 else 'black', zorder=5)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.2f}' for e in RESP_BAR_EPS], rotation=45, ha='right', fontsize=13)
        ax.set_xlabel(r'$\varepsilon$', fontsize=18)
        ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=0.3, zorder=0)
    axes[0].set_ylabel(r'\% trajectories', fontsize=16)
    _sm = plt.cm.ScalarMappable(cmap=_bar_cmap, norm=_bar_norm)
    _sm.set_array([])
    fig.colorbar(_sm, ax=axes, pad=0.02).set_label('Mean reach time (s)', fontsize=15)
    _h = [plt.Rectangle((0, 0), 1, 1, color=_bar_cmap(0.5), label='Goal Reached (color = mean time)'),
          plt.Rectangle((0, 0), 1, 1, color='grey', label='Timeout'),
          plt.Rectangle((0, 0), 1, 1, color='k', label='Unsafe')]
    fig.legend(_h, [h.get_label() for h in _h], loc='lower center', ncol=3, fontsize=13, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(RESP_FIG_DIR / 'dbint_bars_original_vs_revised.pdf', bbox_inches='tight')
    plt.show()

for pol in RESP_ROWS:
    print(f'{pol:<12} ' + '  '.join(f'eps {e:.1f}: reach {o[0]:.1%} unsafe {o[1]:.1%} t {o[2]:.2f}s'
                                     for e, o in zip(RESP_BAR_EPS, _bar[pol])))
'''
