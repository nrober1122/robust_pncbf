"""
Diagnostic visualization for Stage 2 failure trajectories.

Usage:
    python scripts/dbint/visualize_failure_data.py \
        --data-path runs/learned_rcbf_dbint/dbint/stage2_data.pkl \
        --pncbf-ckpt runs/pncbf_dbint/.../ckpts/5000
"""

import pathlib
import pickle

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.collections import LineCollection

import run_config.int_avoid.doubleintwall_cfg
from mrncbf.dyn.doubleint_wall import DoubleIntWall
from mrncbf.pncbf.pncbf import PNCBF
from mrncbf.utils.ckpt_utils import load_ckpt
from mrncbf.utils.jax_utils import jax_default_x32
from mrncbf.utils.logging import set_logger_format


def main(
    data_path: pathlib.Path = typer.Option(..., help="Path to stage2_data.pkl"),
    pncbf_ckpt: pathlib.Path = typer.Option(..., help="Path to pretrained PNCBF checkpoint"),
    out_dir: pathlib.Path = typer.Option("plots/diagnostics", help="Output directory for plots"),
    seed: int = 7957821,
    max_traj_per_eps: int = 10,
):
    jax_default_x32()
    set_logger_format()

    out_dir.mkdir(parents=True, exist_ok=True)

    task = DoubleIntWall()
    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    nom_pol = task.nom_pol_osc

    # Load PNCBF
    logger.info("Loading PNCBF...")
    nom_pol = task.nom_pol_osc
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, pncbf_ckpt)
    logger.info("Loaded PNCBF from {}!".format(pncbf_ckpt))

    # Load Stage 2 data
    logger.info(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        all_traj_np = pickle.load(f)
    trajectories = [jax.tree.map(jnp.array, t) for t in all_traj_np]
    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # Group by epsilon
    eps_groups = {}
    for traj in trajectories:
        eps = float(traj.epsilon[0])
        if eps not in eps_groups:
            eps_groups[eps] = []
        eps_groups[eps].append(traj)

    logger.info(f"Epsilon levels: {sorted(eps_groups.keys())}")

    # ================================================================
    # PLOT 1: Phase portraits colored by h(x_true)
    # ================================================================
    logger.info("Plotting phase portraits...")

    fig, axes = plt.subplots(1, len(eps_groups), figsize=(5 * len(eps_groups), 5), squeeze=False)

    for idx, (eps, trajs) in enumerate(sorted(eps_groups.items())):
        ax = axes[0, idx]
        ax.set_title(f"ε = {eps:.3f}")
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")

        for traj in trajs[:max_traj_per_eps]:
            x_true = np.array(traj.x_true)  # (T, nx)
            h_max = np.array(traj.h_V_true.max(axis=-1))  # (T,) max over constraints

            # Color by h: blue=safe (h<0), red=violated (h>0)
            colors = np.where(h_max > 0, 'red', 'steelblue')

            # Plot as scatter with alpha for time progression
            T = len(x_true)
            alphas = np.linspace(0.2, 1.0, T)
            for t in range(T - 1):
                ax.plot(
                    x_true[t:t+2, 0], x_true[t:t+2, 1],
                    color=colors[t], alpha=alphas[t], linewidth=0.8,
                )
            # Mark start
            ax.plot(x_true[0, 0], x_true[0, 1], 'go', markersize=4, zorder=5)
            # Mark first violation if any
            viol_idx = np.where(h_max > 0)[0]
            if len(viol_idx) > 0:
                t0 = viol_idx[0]
                ax.plot(x_true[t0, 0], x_true[t0, 1], 'rx', markersize=6, zorder=5)

        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase Portraits (blue=safe, red=violated, x=first violation)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_portraits.png", dpi=150, bbox_inches='tight')
    logger.info(f"  Saved phase_portraits.png")
    plt.close(fig)

    # ================================================================
    # PLOT 2: h(x_true) over time
    # ================================================================
    logger.info("Plotting h(x_true) over time...")

    fig, axes = plt.subplots(1, len(eps_groups), figsize=(5 * len(eps_groups), 4), squeeze=False)

    for idx, (eps, trajs) in enumerate(sorted(eps_groups.items())):
        ax = axes[0, idx]
        ax.set_title(f"ε = {eps:.3f}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("max h(x_true)")
        ax.axhline(0, color='black', linewidth=1, linestyle='--', label='safety boundary')

        for traj in trajs[:max_traj_per_eps]:
            h_max = np.array(traj.h_V_true.max(axis=-1))
            color = 'red' if np.any(h_max > 0) else 'steelblue'
            ax.plot(h_max, color=color, alpha=0.5, linewidth=0.8)

        ax.grid(True, alpha=0.3)

    fig.suptitle("h(x_true) over time (>0 = violation)")
    fig.tight_layout()
    fig.savefig(out_dir / "h_true_over_time.png", dpi=150, bbox_inches='tight')
    logger.info(f"  Saved h_true_over_time.png")
    plt.close(fig)

    # ================================================================
    # PLOT 3: Lie derivative discrepancies
    # ================================================================
    logger.info("Plotting Lie derivative discrepancies...")

    fig, axes = plt.subplots(2, len(eps_groups), figsize=(5 * len(eps_groups), 8), squeeze=False)

    for idx, (eps, trajs) in enumerate(sorted(eps_groups.items())):
        # Row 0: |ḣ(x̂) - ḣ(x_true)| over time
        ax = axes[0, idx]
        ax.set_title(f"ε = {eps:.3f}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("|ḣ(x̂) - ḣ(x_true)|")

        all_disc = []
        for traj in trajs[:max_traj_per_eps]:
            hdot_hat = np.array(traj.h_Lfh_hat + traj.h_Lgh_u_hat)
            hdot_true = np.array(traj.h_Lfh_true + traj.h_Lgh_u_true)
            disc = np.abs(hdot_hat - hdot_true).max(axis=-1)  # Max over constraints
            ax.plot(disc, alpha=0.4, linewidth=0.6)
            all_disc.append(disc)

        if all_disc:
            all_disc = np.stack(all_disc)
            ax.plot(np.median(all_disc, axis=0), 'k-', linewidth=2, label='median')
            ax.plot(np.percentile(all_disc, 95, axis=0), 'k--', linewidth=1, label='95th pct')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: ‖Lgh‖ over time
        ax = axes[1, idx]
        ax.set_xlabel("timestep")
        ax.set_ylabel("‖L_g h(x̂)‖")

        for traj in trajs[:max_traj_per_eps]:
            Lgh_norm = np.array(traj.h_Lgh_norm.max(axis=-1))
            ax.plot(Lgh_norm, alpha=0.4, linewidth=0.6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lie derivative discrepancy and ‖Lgh‖")
    fig.tight_layout()
    fig.savefig(out_dir / "lie_deriv_discrepancy.png", dpi=150, bbox_inches='tight')
    logger.info(f"  Saved lie_deriv_discrepancy.png")
    plt.close(fig)

    # ================================================================
    # PLOT 4: Distribution of worst-case targets
    # ================================================================
    logger.info("Computing worst-case target distribution (subset)...")

    from mrncbf.mrncbf.mrncbf_utils import _make_worstcase_fn

    alpha_safe, alpha_unsafe = 2.0, 100.0
    gamma1, gamma2 = 0.1304, 0.0001  # From your logs
    V_shift = 1e-3

    worstcase_fn = _make_worstcase_fn(
        alg, alpha_safe, alpha_unsafe,
        gamma1, gamma2, V_shift, n_samples=64,
    )

    # Sample a subset of states across all epsilon levels
    rng = jr.PRNGKey(999)
    n_samples_per_eps = 200
    target_data = {eps: [] for eps in eps_groups}

    for eps, trajs in sorted(eps_groups.items()):
        logger.info(f"  Computing targets for ε={eps:.3f}...")
        count = 0
        for traj in trajs:
            T = traj.x_true.shape[0]
            for t in range(30, T, 10):
                if count >= n_samples_per_eps:
                    break
                x_hat_t = traj.x_hat[t]
                u_t = traj.u_applied[t]
                if jnp.any(jnp.isnan(x_hat_t)) or jnp.any(jnp.isnan(u_t)):
                    continue

                rng, wc_key = jr.split(rng)
                h_Delta_tgt, h_Lgh_norm, h_V_hat = worstcase_fn(
                    x_hat_t, u_t, eps, wc_key
                )

                if not jnp.any(jnp.isnan(h_Delta_tgt)):
                    target_data[eps].append({
                        'delta_tgt': np.array(h_Delta_tgt),
                        'Lgh_norm': np.array(h_Lgh_norm),
                        'h_V': np.array(h_V_hat),
                        'x_hat': np.array(x_hat_t),
                    })
                    count += 1
            if count >= n_samples_per_eps:
                break

    # Plot target distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): Histogram of Delta targets per epsilon
    ax = axes[0, 0]
    for eps in sorted(target_data.keys()):
        if target_data[eps]:
            vals = np.array([d['delta_tgt'].max() for d in target_data[eps]])
            ax.hist(vals, bins=50, alpha=0.5, label=f"ε={eps:.3f}", density=True)
    ax.set_xlabel("max Δ_target (over constraints)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of worst-case Δ targets")
    ax.legend(fontsize=8)
    ax.axvline(3.0, color='red', linestyle='--', label='Delta_max=3')
    ax.axvline(-3.0, color='red', linestyle='--')
    ax.grid(True, alpha=0.3)

    # (0,1): Delta target vs ‖Lgh‖
    ax = axes[0, 1]
    for eps in sorted(target_data.keys()):
        if target_data[eps]:
            Lgh = np.array([d['Lgh_norm'].max() for d in target_data[eps]])
            delta = np.array([d['delta_tgt'].max() for d in target_data[eps]])
            ax.scatter(Lgh, delta, s=8, alpha=0.4, label=f"ε={eps:.3f}")
    ax.set_xlabel("max ‖L_g h(x̂)‖")
    ax.set_ylabel("max Δ_target")
    ax.set_title("Δ_target vs ‖Lgh‖")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): Delta target vs h(x̂)
    ax = axes[1, 0]
    for eps in sorted(target_data.keys()):
        if target_data[eps]:
            hV = np.array([d['h_V'].max() for d in target_data[eps]])
            delta = np.array([d['delta_tgt'].max() for d in target_data[eps]])
            ax.scatter(hV, delta, s=8, alpha=0.4, label=f"ε={eps:.3f}")
    ax.set_xlabel("max h(x̂) (CBF value)")
    ax.set_ylabel("max Δ_target")
    ax.set_title("Δ_target vs h(x̂) — are outliers near boundary?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): State-space scatter of outlier targets
    ax = axes[1, 1]
    all_deltas_flat = []
    all_x_flat = []
    for eps in sorted(target_data.keys()):
        for d in target_data[eps]:
            all_deltas_flat.append(d['delta_tgt'].max())
            all_x_flat.append(d['x_hat'])

    if all_x_flat:
        all_deltas_flat = np.array(all_deltas_flat)
        all_x_flat = np.array(all_x_flat)
        sc = ax.scatter(
            all_x_flat[:, 0], all_x_flat[:, 1],
            c=np.clip(all_deltas_flat, -5, 10),
            s=8, alpha=0.6, cmap='RdYlBu_r',
        )
        plt.colorbar(sc, ax=ax, label='max Δ_target')
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")
        ax.set_title("State space: where are large targets?")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Worst-case target diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "target_diagnostics.png", dpi=150, bbox_inches='tight')
    logger.info(f"  Saved target_diagnostics.png")
    plt.close(fig)

    # ================================================================
    # SUMMARY STATS
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for eps, trajs in sorted(eps_groups.items()):
        n_violated = sum(1 for t in trajs if np.any(np.array(t.h_V_true) > 0))
        n_total = len(trajs)

        # Collect stats
        all_h_true = np.concatenate([np.array(t.h_V_true) for t in trajs])
        all_Lfh = np.concatenate([np.array(t.h_Lfh_hat) for t in trajs])
        all_Lgh_norm = np.concatenate([np.array(t.h_Lgh_norm) for t in trajs])
        all_hdot_disc = np.concatenate([
            np.abs(np.array(t.h_Lfh_hat + t.h_Lgh_u_hat) - np.array(t.h_Lfh_true + t.h_Lgh_u_true))
            for t in trajs
        ])

        logger.info(f"  ε={eps:.3f}: {n_violated}/{n_total} violated")
        logger.info(f"    h(x_true): mean={all_h_true.mean():.4f}, "
                     f"max={all_h_true.max():.4f}")
        logger.info(f"    ‖Lgh‖:    mean={all_Lgh_norm.mean():.4f}, "
                     f"max={all_Lgh_norm.max():.4f}, "
                     f"99th={np.percentile(all_Lgh_norm, 99):.4f}")
        logger.info(f"    |ḣ disc|:  mean={all_hdot_disc.mean():.4f}, "
                     f"max={all_hdot_disc.max():.4f}, "
                     f"99th={np.percentile(all_hdot_disc, 99):.4f}")

        if target_data.get(eps):
            tgt_vals = np.array([d['delta_tgt'].max() for d in target_data[eps]])
            logger.info(f"    Δ_target:  mean={tgt_vals.mean():.4f}, "
                         f"max={tgt_vals.max():.4f}, "
                         f"99th={np.percentile(tgt_vals, 99):.4f}, "
                         f"frac>3={np.mean(tgt_vals > 3):.1%}")

    logger.info(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)