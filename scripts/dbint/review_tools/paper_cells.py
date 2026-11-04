PAPER_MD = r'''# Revised paper figures (Fig. 2 and Fig. 3)

The paper's Fig. 2 (outcome bars, all 1000 samples) and Fig. 3 (ε = 0.3 trajectories) with `PAPER_POLICY` in the
DMR-CBF panel. Every other panel comes from the compare notebook's caches, unchanged. Fig. 2 copies the compare
notebook's bar-chart cell (`d4911f1d`), including its mean-time colour scale; Fig. 3 uses `draw_cmp` from the cell above.
Files are written to `rebuttal/figs/` with the paper's file names and a new revision suffix.'''

PAPER_CODE = r'''import copy

PAPER_POLICY     = 'PGD DMR-CBF'
PAPER_EPS        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PAPER_TRAJ_EPS   = 0.3
PAPER_ANNOTATE   = False      # Fig. 3: add collision / time-to-goal labels (the paper's Fig. 3 has none)
PAPER_FIG2_FILE  = RESP_FIG_DIR / 'full_scaling_r8.png'
PAPER_FIG3_FILE  = RESP_FIG_DIR / 'dbint_trajectories_0.3_r4.png'

# MC caches with the DMR-CBF entry ('PSHR-CBF') replaced by PAPER_POLICY's full Monte Carlo results
_paper_cached = []
for e in PAPER_EPS:
    with open(_mc_results_dir / f'mc_eps_{e:.3f}.pkl', 'rb') as f:
        m = copy.deepcopy(pickle.load(f))
    r = _load_variant(PAPER_POLICY, e)
    assert len(r['coll']) == len(m['violation_array']['PSHR-CBF']) == N_MC
    m['violation_array']['PSHR-CBF']  = r['coll']
    m['reached_array']['PSHR-CBF']    = r['reached'] & ~r['coll']
    m['reach_time_array']['PSHR-CBF'] = r['t_reach']
    m['violation_rates']['PSHR-CBF']  = float(r['coll'].mean())
    m['goal_rates']['PSHR-CBF']       = float((r['reached'] & ~r['coll']).mean())
    _paper_cached.append(m)

eps_vals   = [float(r['STATE_EPS'][0]) for r in _paper_cached]
x          = np.arange(len(eps_vals))
PLOT_ORDER = ['GUARDIAN', 'CBF', 'R-CBF (0.2)', 'R-CBF (1.0)', 'Duality CBF', 'R-CBF-QP', 'PSHR-CBF', 'NMR-CBF (RL)',
              'NMR-CBF', 'Nominal']
approaches = [a for a in PLOT_ORDER if a in _paper_cached[0]['violation_rates']]
x_labels   = [f'{e:.2f}' for e in eps_vals]
MEAN_TIME_WHITE_THRESH = 8.0
SLOW_COLOR, UNSAFE_COLOR = 'grey', 'k'
_rt_cmap = plt.cm.plasma_r

def _mean_reach_time(r, approach):
    """Mean reach time over safe-reached ICs only. Non-finishing ICs excluded."""
    rt_arr      = np.array(r.get('reach_time_array', {}).get(approach, []))
    viol_arr    = np.array(r['violation_array'].get(approach, []), dtype=bool)
    reached_arr = np.array(r['reached_array'].get(approach,  []), dtype=bool)
    safe        = reached_arr & (~viol_arr)
    return float(np.nanmean(rt_arr[safe])) if (safe.any() and rt_arr.size) else float('nan')

_all_mean_times = {a: [_mean_reach_time(r, a) for r in _paper_cached] for a in approaches}
_nom_means  = _all_mean_times.get('Nominal', list(_all_mean_times.values())[0])
_rt_tmin    = float(np.nanmin(_nom_means)) if not all(np.isnan(v) for v in _nom_means) else 0.0
_all_finite = [v for vals in _all_mean_times.values() for v in vals if not np.isnan(v)]
_rt_norm    = plt.Normalize(vmin=_rt_tmin, vmax=float(max(_all_finite)))

with mpl.rc_context({'font.family': 'serif', 'font.size': 12, 'text.usetex': True}):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), layout='constrained', sharey=True, dpi=400)
    axes_grid = axes
    axes = axes.flatten()
    for i_ax, (ax, approach) in enumerate(zip(axes, approaches)):
        reached_pct = np.array([100 * float(r['goal_rates'].get(approach, 0.0)) for r in _paper_cached])
        unsafe_pct  = np.array([100 * float(r['violation_rates'].get(approach, 0.0)) for r in _paper_cached])
        slow_pct    = 100.0 - reached_pct - unsafe_pct
        for i in range(len(eps_vals)):
            mt = _all_mean_times[approach][i]
            ax.bar(i, reached_pct[i], color=_rt_cmap(_rt_norm(mt)) if not np.isnan(mt) else '0.85', zorder=3)
        ax.bar(x, slow_pct,   color=SLOW_COLOR,   bottom=reached_pct,            zorder=3)
        ax.bar(x, unsafe_pct, color=UNSAFE_COLOR, bottom=reached_pct + slow_pct, zorder=3)
        if approach != 'Nominal':
            for i in range(len(eps_vals)):
                mt = _all_mean_times[approach][i]
                if np.isnan(mt) or reached_pct[i] < 3:
                    continue
                ax.text(i, 1.5, rf'\textbf{{{{{mt:.1f}s}}}}', ha='center', va='bottom', fontsize=10,
                        color='white' if mt > MEAN_TIME_WHITE_THRESH else 'black', zorder=5)
        _title = {'PSHR-CBF': 'DMR-CBF', 'NMR-CBF (RL)': 'NMR-CBF', 'NMR-CBF': 'NMR-CBF (pretrain)'}.get(approach, approach)
        ax.set_title(rf'\textbf{{{_title}}}', fontsize=16)
        ax.set_xticks(x)
        if i_ax < 4:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False)
        else:
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=13)
            ax.set_xlabel(r'$\varepsilon$', fontsize=18)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.tick_params(axis='y', labelsize=12)
    for ax in axes[len(approaches):]:
        ax.set_visible(False)
    axes[0].set_ylabel(r'\% trajectories', fontsize=16)
    if len(approaches) > 4:
        axes[4].set_ylabel(r'\% trajectories', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap=_rt_cmap, norm=_rt_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_grid, pad=0.02)
    cbar.set_label('Mean reach time (s)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    _handles = [plt.Rectangle((0, 0), 1, 1, color=_rt_cmap(0.5), label='Goal Reached (color = mean time)'),
                plt.Rectangle((0, 0), 1, 1, color=SLOW_COLOR,   label='Timeout'),
                plt.Rectangle((0, 0), 1, 1, color=UNSAFE_COLOR, label='Unsafe')]
    fig.legend(_handles, [h.get_label() for h in _handles], loc='lower center', ncol=3, fontsize=16,
               bbox_to_anchor=(0.5, -0.10))
    fig.savefig(PAPER_FIG2_FILE, dpi=400, bbox_inches='tight')
    plt.show()

for approach in approaches:
    print(f'{approach:<14} ' + '  '.join(f'{t:5.2f}' for t in _all_mean_times[approach]))

draw_cmp(PAPER_TRAJ_EPS, PAPER_POLICY, suptitle=False, annotate=PAPER_ANNOTATE, save=PAPER_FIG3_FILE)
print('saved', PAPER_FIG2_FILE, PAPER_FIG3_FILE)
'''
