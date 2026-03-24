"""
Training script for the learned robust CBF residual (Δ_θ).

Usage:
    # Stage 2+3+4 from scratch (requires a pretrained PNCBF checkpoint):
    python scripts/dbint/learned_rcbf_dbint.py --name my_run --pncbf-ckpt runs/pncbf_dbint/.../ckpts/5000

    # Resume from a Δ_θ checkpoint:
    python scripts/dbint/learned_rcbf_dbint.py --name my_run --pncbf-ckpt ... --delta-ckpt runs/learned_rcbf/.../ckpts/3000
"""

import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import typer
from loguru import logger

import wandb
from mrncbf.dyn.doubleint_wall import DoubleIntWall
from mrncbf.plotting.plot_task_summary import plot_task_summary
from mrncbf.plotting.plotter import MPPlotter
from mrncbf.pncbf.pncbf import PNCBF
from mrncbf.training.ckpt_manager import get_ckpt_manager
from mrncbf.training.run_dir import init_wandb_and_get_run_dir
from mrncbf.utils.ckpt_utils import load_ckpt, load_ckpt_with_step
from mrncbf.utils.jax_utils import jax2np, jax_default_x32
from mrncbf.utils.logging import set_logger_format

# These come from the new learned_rcbf module
from mrncbf.mrncbf.fine_tune import (
    make_stage4_loss,
    stage4_train,
)
from mrncbf.networks.temporal_residual import ResidualCorrection
from mrncbf.qp.min_norm_cbf import rcbf_qp_learned, min_norm_cbf, rcbf_qp_linear, compute_rcbf_gammas
from mrncbf.mrncbf.pretrain import pretrain_Delta
from mrncbf.mrncbf.mrncbf_utils import HistoryBuffer, collect_failure_rollout


import run_config.int_avoid.doubleintwall_cfg


# ============================================================================
# Configuration
# ============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class LearnedRCBFCfg:
    """Configuration for the learned residual training pipeline."""
 
    # ── Architecture ──
    hidden_dim: int = 64
    gru_dim: int = 64
    use_temporal: bool = True
    Delta_max: float = 3.0
    window_len: int = 30
 
    # ── Analytical baseline ──
    # Initial gamma values; will be recomputed from sigma_hat if
    # compute_gammas_from_data is True.
    gamma1: float = 0.1
    gamma2: float = 0.1
    R_floor: float = 0.0
    compute_gammas_from_data: bool = True
 
    # ── CBF filter ──
    alpha_safe: float = 2.0
    alpha_unsafe: float = 100.0
    V_shift: float = 1e-3
 
    # ── Stage 2: failure data collection ──
    noise_schedule: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5]
    )
    episodes_per_noise: int = 50
    failure_episode_length: int = 300
 
    # ── Stage 3: supervised pre-training ──
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 256
 
    # ── Stage 4: end-to-end fine-tuning ──
    stage4_iterations: int = 5000
    stage4_lr: float = 3e-4
    episodes_per_iter: int = 16
    episode_length: int = 200
 
    # Cost weights for Stage 4
    w_safety: float = 100.0
    w_conserv: float = 0.1
    w_reg: float = 0.01
 
    # ── Logging ──
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000
    eval_episodes: int = 100
 
 
# ============================================================================
# Evaluation
# ============================================================================
 
def evaluate_learned_rcbf(
    pncbf: PNCBF,
    Delta_net: ResidualCorrection,
    Delta_params,
    cfg: LearnedRCBFCfg,
    noise_levels: list,
    num_episodes: int = 100,
    episode_length: int = 300,
    scenarios: list = None,
    rng_key=None,
):
    """
    Evaluate the learned robust CBF across noise levels and scenarios.
 
    Compares:
      - no_robust:     R = 0
      - analytical:    R = gamma1 * ||Lgh|| + gamma2^2 * ||Lgh||^2
      - learned:       R = max(R_floor, R_analytical + Delta_theta)
 
    Returns dict of metrics.
    """
    if scenarios is None:
        scenarios = ['gaussian', 'epsilon_spike', 'vio_drift']
    if rng_key is None:
        rng_key = jr.PRNGKey(9999)
 
    task = pncbf.task
    Vh_apply = ft.partial(pncbf.get_Vh, params=pncbf.Vh.params)
 
    results = {}
 
    for scenario in scenarios:
        for epsilon in noise_levels:
            key_tag = f"{scenario}/eps={epsilon:.3f}"
 
            metrics = {
                'no_robust_safe': 0, 'analytical_safe': 0, 'learned_safe': 0,
                'no_robust_conserv': 0.0, 'analytical_conserv': 0.0, 'learned_conserv': 0.0,
                'avg_Delta': 0.0, 'total_steps': 0,
            }
 
            for ep in range(num_episodes):
                rng_key, ep_key = jr.split(rng_key)
                x0_key, noise_key_base = jr.split(ep_key)
                x_true = task.sample_train_x0(x0_key, 1)[0]
 
                history = HistoryBuffer.create(cfg.window_len, task.nx, task.nh)
 
                violated = {'no_robust': False, 'analytical': False, 'learned': False}
 
                for t in range(episode_length):
                    noise_key_base, step_key = jr.split(noise_key_base)
 
                    # Generate noise based on scenario
                    delta, reported_eps = _generate_scenario_noise(
                        step_key, x_true, t, scenario, epsilon, task.nx
                    )
                    x_hat = x_true + delta
 
                    # ── Shared CBF ingredients ──
                    h_V = Vh_apply(x_hat) + cfg.V_shift
                    hx_Vx = jax.jacobian(Vh_apply)(x_hat)
                    f = task.f(x_hat)
                    G = task.G(x_hat)
                    u_nom = pncbf.nom_pol(x_hat)
 
                    h_LG = hx_Vx @ G
                    h_Lgh_norm = jnp.linalg.norm(h_LG, axis=-1)
 
                    from mrncbf.qp.min_norm_cbf import (
                        min_norm_cbf, rcbf_qp_linear,
                        _resolve_alpha_local,
                    )
                    alpha = jnp.where(
                        jnp.all(h_V < 0), cfg.alpha_safe, cfg.alpha_unsafe
                    )
 
                    # ── Method 1: No robustification ──
                    u_norob, _, _ = min_norm_cbf(
                        alpha, task.u_min, task.u_max,
                        h_V, hx_Vx, f, G, u_nom,
                    )
 
                    # ── Method 2: Analytical only ──
                    u_anal, _, _ = rcbf_qp_linear(
                        alpha, task.u_min, task.u_max,
                        h_V, hx_Vx, f, G, u_nom,
                        gamma1=cfg.gamma1, gamma2=cfg.gamma2,
                    )
 
                    # ── Method 3: Learned ──
                    h_Delta = Delta_net.apply(
                        Delta_params,
                        x_hat, h_Lgh_norm, h_V, reported_eps,
                        history.data,
                    )
                    u_learned, _, _ = rcbf_qp_learned(
                        alpha, task.u_min, task.u_max,
                        h_V, hx_Vx, f, G, u_nom,
                        cfg.gamma1, cfg.gamma2, h_Delta, cfg.R_floor,
                    )
 
                    # ── Step true dynamics for each method ──
                    f_true = task.f(x_true)
                    G_true = task.G(x_true)
 
                    for method, u in [('no_robust', u_norob), ('analytical', u_anal), ('learned', u_learned)]:
                        xdot = f_true + G_true @ u
                        x_next = x_true + task.dt * xdot
                        h_next = task.h_components(x_next)
                        if jnp.any(h_next > 0):
                            violated[method] = True
 
                    # Use learned control for the actual trajectory
                    xdot = f_true + G_true @ u_learned
                    x_true = x_true + task.dt * xdot
 
                    # Track metrics
                    metrics['no_robust_conserv'] += float(jnp.sum((u_norob - u_nom) ** 2))
                    metrics['analytical_conserv'] += float(jnp.sum((u_anal - u_nom) ** 2))
                    metrics['learned_conserv'] += float(jnp.sum((u_learned - u_nom) ** 2))
                    metrics['avg_Delta'] += float(jnp.mean(h_Delta))
                    metrics['total_steps'] += 1
 
                    # Update history
                    history = history.append(x_hat, reported_eps, h_Lgh_norm)
 
                if not violated['no_robust']:
                    metrics['no_robust_safe'] += 1
                if not violated['analytical']:
                    metrics['analytical_safe'] += 1
                if not violated['learned']:
                    metrics['learned_safe'] += 1
 
            S = max(metrics['total_steps'], 1)
            results[key_tag] = {
                'no_robust_safety_rate': metrics['no_robust_safe'] / num_episodes,
                'analytical_safety_rate': metrics['analytical_safe'] / num_episodes,
                'learned_safety_rate': metrics['learned_safe'] / num_episodes,
                'no_robust_conserv': metrics['no_robust_conserv'] / S,
                'analytical_conserv': metrics['analytical_conserv'] / S,
                'learned_conserv': metrics['learned_conserv'] / S,
                'avg_Delta': metrics['avg_Delta'] / S,
            }
 
    return results
 
 
def _generate_scenario_noise(rng_key, x_true, t, scenario, epsilon, nx):
    """Generate noise and reported epsilon for different test scenarios."""
    if scenario == 'gaussian':
        delta = jax.random.uniform(rng_key, (nx,), minval=-epsilon, maxval=epsilon)
        return delta, epsilon
 
    elif scenario == 'epsilon_spike':
        # Small actual noise, but reported epsilon spikes for t in [50, 70)
        delta = 0.01 * jax.random.normal(rng_key, (nx,))
        is_spike = (t >= 50) & (t < 70)
        reported_eps = jnp.where(is_spike, 10.0 * epsilon, epsilon)
        return delta, reported_eps
 
    elif scenario == 'vio_drift':
        direction = jnp.ones(nx) / jnp.sqrt(nx)
        drift = epsilon * jnp.minimum(t / 100.0, 3.0)
        delta = drift * direction + 0.1 * epsilon * jax.random.normal(rng_key, (nx,))
        reported_eps = epsilon * (1.0 + t / 100.0)
        return delta, reported_eps
 
    elif scenario == 'intermittent':
        dropout = jax.random.bernoulli(rng_key, p=0.1)
        k1, k2 = jr.split(rng_key)
        delta = jnp.where(
            dropout,
            3.0 * epsilon * jax.random.normal(k1, (nx,)),
            0.01 * jax.random.normal(k2, (nx,)),
        )
        reported_eps = jnp.where(dropout, 3.0 * epsilon, epsilon)
        return delta, reported_eps
 
    else:
        delta = jax.random.uniform(rng_key, (nx,), minval=-epsilon, maxval=epsilon)
        return delta, epsilon
 
 
# ============================================================================
# Main training script
# ============================================================================

def main(
    name: str = typer.Option(..., help="Name of the run."),
    group: str = typer.Option(None),
    pncbf_ckpt: pathlib.Path = typer.Option(..., help="Path to pretrained PNCBF checkpoint."),
    delta_ckpt: pathlib.Path = typer.Option(None, help="Resume from Δ_θ checkpoint."),
    seed: int = 7957821,
    skip_stage2: bool = typer.Option(False, help="Skip Stage 2 (use if resuming from Stage 4)."),
    skip_stage3: bool = typer.Option(False, help="Skip Stage 3 (use if resuming from Stage 4)."),
    no_temporal: bool = typer.Option(False, help="Ablation: disable GRU temporal head."),
):
    jax_default_x32()
    set_logger_format()
 
    task = DoubleIntWall()
    cfg = LearnedRCBFCfg(use_temporal=not no_temporal)
 
    BASE_CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
 
    # ── Load pretrained PNCBF ──
    logger.info("Loading PNCBF from {}...".format(pncbf_ckpt))
    nom_pol = task.nom_pol_osc
    alg: PNCBF = PNCBF.create(seed, task, BASE_CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, pncbf_ckpt)
    logger.info("Loaded PNCBF from {}!".format(pncbf_ckpt))
 
    # ── Create Δ_θ network ──
    Delta_net = ResidualCorrection(
        nh=task.nh,
        hidden_dim=cfg.hidden_dim,
        gru_dim=cfg.gru_dim,
        use_temporal=cfg.use_temporal,
        Delta_max=cfg.Delta_max,
    )
 
    # Initialize network
    rng = jr.PRNGKey(seed + 1000)
    feat_dim = task.nx + 1 + task.nh
    dummy_args = (
        jnp.zeros(task.nx),                              # x_hat
        jnp.zeros(task.nh),                              # h_Lgh_norm
        jnp.zeros(task.nh),                              # h_val
        0.0,                                              # epsilon
        jnp.zeros((cfg.window_len, feat_dim)),            # history_seq
    )
    Delta_params = Delta_net.init(rng, *dummy_args)
 
    n_params = sum(x.size for x in jax.tree.leaves(Delta_params))
    logger.info(f"Δ_θ network: {n_params} parameters, temporal={cfg.use_temporal}")
 
    # ── Setup wandb and directories ──
    extras = {
        "pncbf_ckpt": str(pncbf_ckpt),
        "use_temporal": cfg.use_temporal,
        "noise_schedule": cfg.noise_schedule,
        "Delta_max": cfg.Delta_max,
        "w_safety": cfg.w_safety,
        "w_conserv": cfg.w_conserv,
        "w_reg": cfg.w_reg,
    }
    # Reuse the wandb init pattern from the base script
    wandb.init(
        project="learned_rcbf_dbint",
        name=name,
        group=group,
        config=extras,
    )
    run_dir = pathlib.Path(f"runs/learned_rcbf_dbint/{name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
 
    plotter = MPPlotter(task, plot_dir)
    ckpt_manager = get_ckpt_manager(ckpt_dir)
 
    # ====================================================================
    # STAGE 2: Collect failure data under measurement noise
    # ====================================================================
 
    stage2_cache = run_dir / "stage2_data.pkl"
 
    if not skip_stage2:
        # Check for cached data first
        if stage2_cache.exists():
            logger.info("Found cached Stage 2 data at {}, loading...".format(stage2_cache))
            import pickle
            with open(stage2_cache, "rb") as f:
                all_trajectories_np = pickle.load(f)
            # Convert back to JAX arrays
            all_trajectories = [
                jax.tree.map(jnp.array, t) for t in all_trajectories_np
            ]
            logger.info(f"  Loaded {len(all_trajectories)} cached trajectories.")
        else:
            logger.info("=" * 60)
            logger.info("STAGE 2: Collecting failure data")
            logger.info("=" * 60)
 
            all_trajectories = []
            rng_collect = jr.PRNGKey(seed + 2000)
 
            for epsilon in cfg.noise_schedule:
                logger.info(f"  Collecting {cfg.episodes_per_noise} episodes at ε={epsilon:.3f}...")
 
                eps_violations = 0
                for ep in range(cfg.episodes_per_noise):
                    rng_collect, ep_key, x0_key = jr.split(rng_collect, 3)
                    x0 = task.sample_train_x0(x0_key, 1)[0]
 
                    traj = collect_failure_rollout(
                        pncbf=alg,
                        x0_true=x0,
                        epsilon=epsilon,
                        episode_length=cfg.failure_episode_length,
                        alpha_safe=cfg.alpha_safe,
                        alpha_unsafe=cfg.alpha_unsafe,
                        V_shift=cfg.V_shift,
                        rng_key=ep_key,
                    )
                    all_trajectories.append(traj)
 
                    # Check if any violation occurred
                    h_max = traj.h_V_true.max()
                    if h_max > 0:
                        eps_violations += 1
 
                rate = eps_violations / cfg.episodes_per_noise
                logger.info(f"    ε={epsilon:.3f}: {eps_violations}/{cfg.episodes_per_noise} "
                            f"violated ({rate:.1%}) without robustification")
                wandb.log({
                    f"stage2/violation_rate_eps{epsilon:.3f}": rate,
                })
 
            logger.info(f"  Collected {len(all_trajectories)} total trajectories.")
 
            # Save to cache
            logger.info(f"  Saving Stage 2 data to {stage2_cache}...")
            import pickle
            all_trajectories_np = [
                jax.tree.map(lambda x: np.array(x), t) for t in all_trajectories
            ]
            with open(stage2_cache, "wb") as f:
                pickle.dump(all_trajectories_np, f)
            logger.info("  Saved.")
    else:
        if stage2_cache.exists():
            logger.info("Loading cached Stage 2 data (--skip-stage2 but cache exists)...")
            import pickle
            with open(stage2_cache, "rb") as f:
                all_trajectories_np = pickle.load(f)
            all_trajectories = [
                jax.tree.map(jnp.array, t) for t in all_trajectories_np
            ]
        else:
            all_trajectories = None
            logger.info("Skipping Stage 2 (--skip-stage2, no cache found).")
 
    # ====================================================================
    # STAGE 2.5: Compute analytical gamma values from data
    # ====================================================================
 
    if cfg.compute_gammas_from_data and all_trajectories is not None:
        logger.info("Computing analytical gamma values from failure data...")
 
        # Estimate sigma_hat: max control deviation within noise ball
        # Use a representative noise level (median of schedule)
        eps_repr = cfg.noise_schedule[len(cfg.noise_schedule) // 2]
        rng_gamma = jr.PRNGKey(seed + 3000)
        n_sigma_samples = 500
 
        # Sample states from the collected data
        all_x_hat = jnp.concatenate([t.x_hat for t in all_trajectories], axis=0)
        sample_idx = jr.choice(rng_gamma, len(all_x_hat), shape=(100,), replace=False)
        sample_states = all_x_hat[sample_idx]
 
        sigma_hats = []
        for x_hat in sample_states:
            rng_gamma, sample_key = jr.split(rng_gamma)
            x_perturbed = x_hat + jax.random.uniform(
                sample_key, shape=(n_sigma_samples, task.nx),
                minval=-eps_repr, maxval=eps_repr,
            )
            u_hat = alg.nom_pol(x_hat)
            u_perturbed = jax.vmap(alg.nom_pol)(x_perturbed)
            norms = jnp.linalg.norm(u_perturbed - u_hat, axis=-1)
            # Filter NaNs (from states where the policy is ill-defined)
            norms = jnp.where(jnp.isnan(norms), 0.0, norms)
            sigma = jnp.max(norms)
            if not jnp.isnan(sigma):
                sigma_hats.append(sigma)
 
        if len(sigma_hats) == 0:
            logger.warning("  All sigma_hat samples were NaN! Using default gammas.")
        else:
            sigma_hat = float(jnp.max(jnp.array(sigma_hats)))
            gamma1, gamma2 = compute_rcbf_gammas(sigma_hat)
            cfg.gamma1 = float(gamma1)
            cfg.gamma2 = float(gamma2)
 
        logger.info(f"  sigma_hat={sigma_hat:.4f}, γ₁={cfg.gamma1:.4f}, γ₂={cfg.gamma2:.4f}")
        wandb.log({
            "stage2/sigma_hat": sigma_hat,
            "stage2/gamma1": cfg.gamma1,
            "stage2/gamma2": cfg.gamma2,
        })
 
    # ====================================================================
    # STAGE 3: Supervised pre-training of Δ_θ
    # ====================================================================
 
    if not skip_stage3 and all_trajectories is not None:
        logger.info("=" * 60)
        logger.info("STAGE 3: Supervised pre-training of Δ_θ")
        logger.info("=" * 60)
 
        Delta_params = pretrain_Delta(
            Delta_net=Delta_net,
            trajectories=all_trajectories,
            pncbf=alg,
            alpha_safe=cfg.alpha_safe,
            alpha_unsafe=cfg.alpha_unsafe,
            gamma1=cfg.gamma1,
            gamma2=cfg.gamma2,
            window_len=cfg.window_len,
            num_epochs=cfg.pretrain_epochs,
            batch_size=cfg.pretrain_batch_size,
            lr=cfg.pretrain_lr,
        )
        logger.info("Stage 3 complete.")
 
        # Save Stage 3 checkpoint
        stage3_dir = ckpt_dir / "stage3"
        stage3_dir.mkdir(exist_ok=True)
        # Simple save — in practice you'd use orbax or the project's ckpt utils
        import pickle
        with open(stage3_dir / "Delta_params.pkl", "wb") as f:
            pickle.dump(jax2np(Delta_params), f)
        logger.info(f"  Saved Stage 3 Δ_θ to {stage3_dir}")
    elif delta_ckpt is not None:
        logger.info(f"Loading Δ_θ from {delta_ckpt}...")
        import pickle
        with open(delta_ckpt / "Delta_params.pkl", "rb") as f:
            Delta_params_np = pickle.load(f)
        Delta_params = jax.tree.map(jnp.array, Delta_params_np)
        logger.info("Loaded Δ_θ checkpoint.")
    else:
        logger.info("Skipping Stage 3 (--skip-stage3), using randomly initialized Δ_θ.")
 
    # Free failure data memory before Stage 4
    del all_trajectories
 
    # ====================================================================
    # STAGE 4: End-to-end fine-tuning through differentiable QP
    # ====================================================================
 
    logger.info("=" * 60)
    logger.info("STAGE 4: End-to-end fine-tuning")
    logger.info("=" * 60)
 
    Delta_params = stage4_train(
        pncbf=alg,
        Delta_net=Delta_net,
        Delta_params_init=Delta_params,
        alpha_safe=cfg.alpha_safe,
        alpha_unsafe=cfg.alpha_unsafe,
        gamma1=cfg.gamma1,
        gamma2=cfg.gamma2,
        R_floor=cfg.R_floor,
        V_shift=cfg.V_shift,
        noise_schedule=cfg.noise_schedule,
        num_iterations=cfg.stage4_iterations,
        episodes_per_iter=cfg.episodes_per_iter,
        episode_length=cfg.episode_length,
        window_len=cfg.window_len,
        lr=cfg.stage4_lr,
        seed=seed + 4000,
        # Pass logging hooks
        log_every=cfg.log_every,
        eval_every=cfg.eval_every,
        ckpt_every=cfg.ckpt_every,
        wandb_log=True,
        ckpt_dir=ckpt_dir,
        eval_fn=ft.partial(
            evaluate_learned_rcbf,
            pncbf=alg,
            Delta_net=Delta_net,
            cfg=cfg,
            noise_levels=cfg.noise_schedule[:3],  # Subset for speed during training
            num_episodes=30,
            episode_length=200,
        ),
    )
 
    # ====================================================================
    # FINAL EVALUATION
    # ====================================================================
 
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
 
    final_results = evaluate_learned_rcbf(
        pncbf=alg,
        Delta_net=Delta_net,
        Delta_params=Delta_params,
        cfg=cfg,
        noise_levels=cfg.noise_schedule,
        num_episodes=cfg.eval_episodes,
        episode_length=cfg.failure_episode_length,
        scenarios=['gaussian', 'epsilon_spike', 'vio_drift', 'intermittent'],
    )
 
    logger.info("")
    logger.info(f"{'Scenario':<35s}  {'NoRob':>7s}  {'Anal':>7s}  {'Learned':>7s}  {'AvgΔ':>8s}")
    logger.info("-" * 75)
    for key, vals in sorted(final_results.items()):
        logger.info(
            f"  {key:<33s}  "
            f"{vals['no_robust_safety_rate']:>6.1%}  "
            f"{vals['analytical_safety_rate']:>6.1%}  "
            f"{vals['learned_safety_rate']:>6.1%}  "
            f"{vals['avg_Delta']:>+7.4f}"
        )
        wandb.log({
            f"final/{key}/no_robust_safety": vals['no_robust_safety_rate'],
            f"final/{key}/analytical_safety": vals['analytical_safety_rate'],
            f"final/{key}/learned_safety": vals['learned_safety_rate'],
            f"final/{key}/analytical_conserv": vals['analytical_conserv'],
            f"final/{key}/learned_conserv": vals['learned_conserv'],
            f"final/{key}/avg_Delta": vals['avg_Delta'],
        })
 
    # Save final checkpoint
    logger.info("Saving final Δ_θ checkpoint...")
    import pickle
    final_ckpt = ckpt_dir / "final"
    final_ckpt.mkdir(exist_ok=True)
    with open(final_ckpt / "Delta_params.pkl", "wb") as f:
        pickle.dump(jax2np(Delta_params), f)
    with open(final_ckpt / "config.pkl", "wb") as f:
        pickle.dump(cfg, f)
    logger.info(f"Saved to {final_ckpt}")
 
    wandb.finish()
    logger.info("Done!")
 
 
if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)