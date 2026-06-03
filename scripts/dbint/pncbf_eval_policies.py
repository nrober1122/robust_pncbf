import functools as ft
import pathlib

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.lines import Line2D

import run_config.int_avoid.doubleintwall_cfg
from mrncbf.dyn.doubleint_wall import DoubleIntWall
from mrncbf.dyn.sim_cts import SimCtsReal
from mrncbf.plotting.contour_utils import centered_norm
from mrncbf.plotting.plotstyle import PlotStyle
from mrncbf.pncbf.pncbf import PNCBF
from mrncbf.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from mrncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from mrncbf.utils.logging import set_logger_format
from mrncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval_policies")
    task = DoubleIntWall()
    nom_pol = task.nom_pol_rng3
    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    T = 80
    tf = T * task.dt
    noise_scale = 0.0
    alpha = 0.01
    boundary_thresh = 0.3   # keep x0s with |max Vh| < this
    max_x0s = 60            # cap to keep runtime manageable

    # ── Vh on grid ───────────────────────────────────────────────────────────────
    logger.info("Computing Vh on grid...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=64)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh  = bbh_Vh.max(-1)

    # ── Vh contour (reference) ────────────────────────────────────────────────────
    norm   = centered_norm(bbh_Vh.min(), bbh_Vh.max())
    levels = 31
    fig, axes = plt.subplots(1, task.nh, figsize=(8, 4), layout="constrained")
    axes = np.atleast_1d(axes)
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii],
                          norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        cs1 = ax.contour(bb_Xs, bb_Ys, bbh_Vh[:, :, ii],
                         levels=[0.0], colors=[PlotStyle.ZeroColor],
                         alpha=0.98, linewidths=1.0)
        cbar = fig.colorbar(cs0, ax=ax)
        cbar.add_lines(cs1)
        task.plot_phase(ax)
        ax.set_title(task.h_labels[ii])
    fig.savefig(plot_dir / "Vh_contour.pdf")
    plt.close(fig)

    # ── Sample x0s near the Vh=0 boundary ────────────────────────────────────────
    bb_x_flat  = bb_x.reshape(-1, task.nx)
    bb_Vh_flat = bb_Vh.reshape(-1)
    near_mask  = (bb_Vh_flat < 0) & (bb_Vh_flat > -boundary_thresh)
    x0s = bb_x_flat[near_mask]
    logger.info(f"Found {len(x0s)} boundary x0s (|Vh| < {boundary_thresh})")
    if len(x0s) > max_x0s:
        idx  = np.round(np.linspace(0, len(x0s) - 1, max_x0s)).astype(int)
        x0s  = x0s[idx]
        logger.info(f"Subsampled to {max_x0s}")

    # ── Policy definitions ────────────────────────────────────────────────────────
    POLICIES = {
        "cbf_sloped": ft.partial(
            alg.get_cbf_control_sloped, alpha, alpha, V_shift=1e-2,
        ),
        "rcbf_qp": ft.partial(
            alg.get_rcbf_qp_control, alpha, alpha,
            nnv_filter=None, epsilon=noise_scale, V_shift=1e-2,
        ),
        "gcbf": ft.partial(
            alg.get_gcbf_control, alpha, alpha,
            nnv_filter=None, epsilon=noise_scale, V_shift=1e-2,
        ),
    }

    # ── Run rollouts ──────────────────────────────────────────────────────────────
    Vh_along = jax_jit(rep_vmap(alg.get_Vh, rep=1))   # (T, nx) -> (T, nh)

    all_trajs = {}   # pol_name -> list of (T_x, violated: bool)
    for pol_name, pol in POLICIES.items():
        logger.info(f"  Rolling out {pol_name} ({len(x0s)} trajectories)...")
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt,
                         use_obs=False, use_pid=False, max_steps=T * 2 + 3)
        rollout_fn = jax_jit(
            ft.partial(sim.rollout_plot, noise_scale=noise_scale, rng_key=rng_key)
        )
        trajs = []
        for x0_i in x0s:
            T_x, _, _, _, _ = jax2np(rollout_fn(x0_i))
            T_Vh = jax2np(Vh_along(T_x)).max(axis=-1)   # (T,)
            violated = bool(np.any(T_Vh > 0.0))
            trajs.append((T_x, violated))
        all_trajs[pol_name] = trajs
        n_safe = sum(1 for _, v in trajs if not v)
        n_viol = sum(1 for _, v in trajs if v)
        logger.info(f"    safe: {n_safe}  violated: {n_viol}")

    # ── Three phase plots ─────────────────────────────────────────────────────────
    logger.info("Plotting boundary trajectory comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained",
                             sharex=True, sharey=True)
    for ax, (pol_name, trajs) in zip(axes, all_trajs.items()):
        n_safe = sum(1 for _, v in trajs if not v)
        n_viol = sum(1 for _, v in trajs if v)
        for T_x, violated in trajs:
            color = "C3" if violated else "C0"
            ax.plot(T_x[:, 0], T_x[:, 1], color=color, lw=0.5, alpha=0.5)
        x0_arr = np.array([T_x[0] for T_x, _ in trajs])
        ax.scatter(x0_arr[:, 0], x0_arr[:, 1], s=6, color="k", zorder=5, linewidths=0)
        ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0],
                   colors=[PlotStyle.ZeroColor], alpha=0.9, linewidths=1.5)
        task.plot_phase(ax)
        ax.set_title(f"{pol_name}   (safe: {n_safe} / viol: {n_viol})")
        ax.set_xlabel("x1 (position)")
    axes[0].set_ylabel("x2 (velocity)")

    legend_elems = [
        Line2D([0], [0], color="C0", lw=1.5, label="Safe"),
        Line2D([0], [0], color="C3", lw=1.5, label="Violated (Vh > 0)"),
    ]
    axes[-1].legend(handles=legend_elems, fontsize=8, loc="best")

    fig.savefig(plot_dir / "boundary_trajs.pdf")
    plt.close(fig)
    logger.info(f"Saved → {plot_dir / 'boundary_trajs.pdf'}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
