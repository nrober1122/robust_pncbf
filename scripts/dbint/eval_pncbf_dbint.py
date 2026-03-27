import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
import jax


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
    plot_dir = mkdir(run_path / "eval")
    task = DoubleIntWall()
    nom_pol = task.nom_pol_rng3
    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    x0 = np.array([0.8, 0.3])
    T = 80
    tf = T * task.dt
    noise_scale = 0.1

    # Original nominal policy.
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True)
    T_x_nom, T_t_nom, _, T_x_nom_noisy, T_controls_nom = jax2np(
        jax_jit(
            ft.partial(
                sim.rollout_plot,
                noise_scale=noise_scale,
                rng_key=rng_key
            )
        )(x0)
    )
    if T_x_nom_noisy is None:
        T_x_nom_noisy = T_x_nom

    alphas = np.array([1.0])

    # --- Policy definitions ---
    POLICIES = {
        "cbf_sloped": lambda alpha_safe, alpha_unsafe: ft.partial(
            alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2
        ),
        "rcbf_qp": lambda alpha_safe, alpha_unsafe: ft.partial(
            alg.get_rcbf_qp_control, alpha_safe, alpha_unsafe,
            nnv_filter=None, epsilon=noise_scale, V_shift=1e-2
        ),
    }

    def rollout_for_pol_and_alpha(pol_fn, alpha_safe):
        alpha_unsafe = alpha_safe
        pol = pol_fn(alpha_safe, alpha_unsafe)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
        T_x, T_t, _, T_x_noisy, T_controls = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
        return T_x, T_t, T_x_noisy, T_controls

    logger.info("Sim both policies for different alphas...")

    # results[pol_name] = dict with stacked arrays per alpha
    results = {}

    for pol_name, pol_fn in POLICIES.items():
        logger.info(f"  Running policy: {pol_name}")
        bT_x, bT_t, bT_x_noisy, bT_controls = [], [], [], []

        for alpha in alphas:
            T_x, T_t, T_x_noisy, T_controls = jax2np(
                jax_jit(ft.partial(rollout_for_pol_and_alpha, pol_fn))(alpha)
            )
            bT_x.append(T_x)
            bT_t.append(T_t)
            bT_x_noisy.append(T_x_noisy if T_x_noisy is not None else T_x)
            bT_controls.append(T_controls)

        results[pol_name] = {
            "bT_x":        np.stack(bT_x,        axis=0),
            "bT_t":        np.stack(bT_t,         axis=0),
            "bT_x_noisy":  np.stack(bT_x_noisy,  axis=0),
            "bT_controls": np.stack(bT_controls,  axis=0),
        }

    logger.info("bTh_Vh for trajectories...")
    Th_Vh_nom = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom, 0))).squeeze()
    Th_Vh_nom_noisy = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom_noisy, 0))).squeeze()
    T_Vh_nom = np.max(Th_Vh_nom, axis=-1)
    T_Vh_nom_noisy = np.max(Th_Vh_nom_noisy, axis=-1)

    # Compute Vh for each policy's trajectories.
    for pol_name, res in results.items():
        res["bTh_Vh"]       = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(res["bT_x"]))
        res["bTh_Vh_noisy"] = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(res["bT_x_noisy"]))
        res["bT_Vh"]        = np.max(res["bTh_Vh"],       axis=-1)
        res["bT_Vh_noisy"]  = np.max(res["bTh_Vh_noisy"], axis=-1)

    logger.info("bbh_Vh...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh  = bbh_Vh.max(-1)

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    # One color per policy, one linestyle per alpha.
    POL_COLORS = {name: f"C{ii}" for ii, name in enumerate(results)}
    ALPHA_LS   = ["-", "--", ":"]   # up to 3 alphas; extend if needed

    # ── Phase plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C3", ls="--", label="Nominal")
    if T_x_nom_noisy is not None:
        ax.plot(T_x_nom_noisy[:, 0], T_x_nom_noisy[:, 1],
                color="C3", ls=":", alpha=0.6, label="Nominal (noisy)")
    for pol_name, res in results.items():
        color = POL_COLORS[pol_name]
        for ii, alpha in enumerate(alphas):
            ls = ALPHA_LS[ii % len(ALPHA_LS)]
            ax.plot(res["bT_x"][ii, :, 0], res["bT_x"][ii, :, 1],
                    color=color, lw=0.8, ls=ls,
                    label=f"{pol_name} α={alpha}", zorder=100)
            ax.plot(res["bT_x_noisy"][ii, :, 0], res["bT_x_noisy"][ii, :, 1],
                    color=color, lw=0.5, ls=ls, alpha=0.45)
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0],
               colors=[PlotStyle.ZeroColor], alpha=0.6, linewidths=1.0)
    task.plot_phase(ax)
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(plot_dir / "eval_phase.pdf")
    plt.close(fig)

    # ── Vh contour ──────────────────────────────────────────────────────────────
    norm   = centered_norm(bbh_Vh.min(), bbh_Vh.max())
    levels = 31
    fig, axes = plt.subplots(1, task.nh, figsize=(8, 4), layout="constrained")
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii],
                          norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        cs1 = ax.contour(bb_Xs, bb_Ys, bbh_Vh[:, :, ii],
                         levels=[0.0], colors=[PlotStyle.ZeroColor],
                         alpha=0.98, linewidths=1.0)
        cbar = fig.colorbar(cs0, ax=ax)
        cbar.add_lines(cs1)
        task.plot_phase(ax)
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vh.pdf")
    plt.close(fig)

    # ── States and controls over time ───────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), layout="constrained")
    axes[0].plot(T_t_nom, T_x_nom[:, 0],      color="C3", ls="-",  label="x1 nominal")
    axes[0].plot([T_t_nom[0], T_t_nom[-1]], [1.0, 1.0],
                 color="C3", ls=":", label="Wall")
    axes[1].plot(T_t_nom, T_x_nom[:, 1],      color="C3", ls="-",  label="x2 nominal")
    axes[2].plot(T_t_nom, T_controls_nom[:, 0], color="C3", ls="-", label="u nominal")
    if T_x_nom_noisy is not None:
        axes[0].plot(T_t_nom, T_x_nom_noisy[:, 0], color="C3", ls="--", alpha=0.6, label="x1 nominal (noisy)")
        axes[1].plot(T_t_nom, T_x_nom_noisy[:, 1], color="C3", ls="--", alpha=0.6, label="x2 nominal (noisy)")
    for pol_name, res in results.items():
        color = POL_COLORS[pol_name]
        for ii, alpha in enumerate(alphas):
            ls  = ALPHA_LS[ii % len(ALPHA_LS)]
            lbl = f"{pol_name} α={alpha}"
            axes[0].plot(res["bT_t"][ii], res["bT_x"][ii, :, 0], color=color, ls=ls, label=f"x1 {lbl}")
            axes[1].plot(res["bT_t"][ii], res["bT_x"][ii, :, 1], color=color, ls=ls, label=f"x2 {lbl}")
            axes[2].plot(res["bT_t"][ii], res["bT_controls"][ii, :, 0], color=color, ls=ls, label=f"u {lbl}")
            axes[0].plot(res["bT_t"][ii], res["bT_x_noisy"][ii, :, 0], color=color, ls=ls, alpha=0.45)
            axes[1].plot(res["bT_t"][ii], res["bT_x_noisy"][ii, :, 1], color=color, ls=ls, alpha=0.45)
    axes[0].set_ylabel("x1")
    axes[1].set_ylabel("x2")
    axes[2].set_ylabel("u")
    for ax in axes:
        ax.legend(fontsize=7, ncols=2)
    fig.savefig(plot_dir / "eval_traj_ctrl.pdf")
    plt.close(fig)

    # ── Vh along trajectories ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.plot(T_t_nom, T_Vh_nom, color="C3", ls="-", label="Vh nominal")
    if T_x_nom_noisy is not None:
        ax.plot(T_t_nom, T_Vh_nom_noisy, color="C3", ls="--", alpha=0.6, label="Vh nominal (noisy)")
    for pol_name, res in results.items():
        color = POL_COLORS[pol_name]
        for ii, alpha in enumerate(alphas):
            ls  = ALPHA_LS[ii % len(ALPHA_LS)]
            lbl = f"{pol_name} α={alpha}"
            ax.plot(res["bT_t"][ii], res["bT_Vh"][ii],       color=color, ls=ls,          label=f"Vh {lbl}")
            ax.plot(res["bT_t"][ii], res["bT_Vh_noisy"][ii], color=color, ls=ls, alpha=0.45)
    ax.set_ylabel("Vh")
    ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.legend(fontsize=7, ncols=2)
    plt.savefig(plot_dir / "eval_traj_Vh.pdf")
    plt.close(fig)

# def main(ckpt_path: pathlib.Path):
#     jax_default_x32()
#     set_logger_format()
#     seed = 0
#     rng_key = jax.random.PRNGKey(seed)

#     run_path = get_run_path_from_ckpt(ckpt_path)
#     plot_dir = mkdir(run_path / "eval")

#     task = DoubleIntWall()

#     # nom_pol = task.nom_pol_osc
#     nom_pol = task.nom_pol_rng3

#     CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
#     alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
#     alg = load_ckpt(alg, ckpt_path)
#     logger.info("Loaded ckpt from {}!".format(ckpt_path))

#     # Plot how V varies along a trajectory.
#     # x0 = np.array([-0.6, 1.7])
#     # x0 = np.array([0.5, -1.7])
#     x0 = np.array([0.8, 0.3])
#     T = 80
#     tf = T * task.dt
#     noise_scale = 0.2

#     # Original nominal policy.
#     logger.info("Sim nom...")
#     sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True)
#     T_x_nom, T_t_nom, _, T_x_nom_noisy, T_controls_nom = jax2np(
#         jax_jit(
#             ft.partial(
#                 sim.rollout_plot,
#                 noise_scale=noise_scale,
#                 rng_key=rng_key
#             )
#         )(x0)
#     )
#     if T_x_nom_noisy is None:
#         T_x_nom_noisy = T_x_nom

#     # alphas = np.array([0.1, 1.0, 5.0, 10.0])
#     # alphas = np.array([0.001, 0.01, 0.1, 5.0])
#     alphas = np.array([0.01, 0.1, 0.5])

#     def int_pol_for_alpha(alpha_safe):
#         alpha_unsafe = alpha_safe
#         ***pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
#         # pol = ft.partial(alg.get_rcbf_control, alpha_safe, alpha_unsafe, V_shift=1e-2, rho_scale=1.5)
#         ***pol = ft.partial(alg.get_rcbf_qp_control, alpha_safe, alpha_unsafe, nnv_filter=None, epsilon=noise_scale, V_shift=1e-2)
#         sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
#         T_x, T_t, _, T_x_noisy, T_controls = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
#         return T_x, T_t, T_x_noisy, T_controls

#     logger.info("Sim pol for different alphas...")
#     bT_x, bT_t = [], []
#     bT_x_noisy = []
#     bT_controls = []
#     for alpha in alphas:
#         T_x, T_t, T_x_noisy, T_controls = jax2np(jax_jit(int_pol_for_alpha)(alpha))
#         bT_x.append(T_x)
#         bT_t.append(T_t)
#         if T_x_noisy is not None:
#             bT_x_noisy.append(T_x_noisy)
#         else:
#             bT_x_noisy.append(T_x)
#         bT_controls.append(T_controls)
#     bT_x = np.stack(bT_x, axis=0)
#     bT_t = np.stack(bT_t, axis=0)
#     bT_x_noisy = np.stack(bT_x_noisy, axis=0)
#     bT_controls = np.stack(bT_controls, axis=0)

#     logger.info("bTh_Vh for trajectories...")
#     Th_Vh_nom = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom, 0))).squeeze()
#     Th_Vh_nom_noisy = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom_noisy, 0))).squeeze()
#     T_Vh_nom = np.max(Th_Vh_nom, axis=-1)
#     T_Vh_nom_noisy = np.max(Th_Vh_nom_noisy, axis=-1)

#     bTh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bT_x))
#     bTh_Vh_noisy = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bT_x_noisy))
#     bT_Vh = np.max(bTh_Vh, axis=-1)
#     bT_Vh_noisy = np.max(bTh_Vh_noisy, axis=-1)

#     logger.info("bbh_Vh...")
#     bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
#     bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
#     bb_Vh = bbh_Vh.max(-1)
    
#     #####################################################
#     logger.info("Plotting...")
#     h_labels = task.h_labels

#     # Phase plot
#     fig, ax = plt.subplots(layout="constrained")
#     ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C3", ls="--", label="Nominal")
#     if T_x_nom_noisy is not None:
#         ax.plot(T_x_nom_noisy[:, 0], T_x_nom_noisy[:, 1], color="C3", ls=":", alpha=0.6, label="Nominal (noisy)")
#     for ii, alpha in enumerate(alphas):
#         ax.plot(bT_x[ii, :, 0], bT_x[ii, :, 1], color=f"C{ii}", lw=0.5, ls="--", label=f"QP ({alpha})", zorder=100)
#         if bT_x_noisy[ii] is not None:
#             ax.plot(bT_x_noisy[ii, :, 0], bT_x_noisy[ii, :, 1], color=f"C{ii}", lw=0.5, ls=":", alpha=0.6)
#     ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.6, linewidths=1.0)
#     task.plot_phase(ax)
#     ax.legend()
#     fig.savefig(plot_dir / "eval_phase.pdf")
#     plt.close(fig)

#     # Vh contour plot
#     norm = centered_norm(bbh_Vh.min(), bbh_Vh.max())
#     levels = 31
#     figsize = (8, 4)
#     fig, axes = plt.subplots(1, task.nh, figsize=figsize, layout="constrained")
#     for ii, ax in enumerate(axes):
#         cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
#         cs1 = ax.contour(
#             bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
#         )
#         cbar = fig.colorbar(cs0, ax=ax)
#         cbar.add_lines(cs1)
#         task.plot_phase(ax)
#         ax.set_title(h_labels[ii])
#     fig.savefig(plot_dir / "eval_Vh.pdf")
#     plt.close(fig)

#     # States and controls over time
#     fig, axes = plt.subplots(3, 1, figsize=(8, 6), layout="constrained")
#     axes[0].plot(T_t_nom, T_x_nom[:, 0], label="x1 (no filter)", color="C3", ls="-")
#     axes[0].plot([T_t_nom[0], T_t_nom[-1]], [1.0, 1.0], label="Wall", color="C3", ls=":")
#     axes[1].plot(T_t_nom, T_x_nom[:, 1], label="x2 (no filter)", color="C3", ls="-")
#     axes[2].plot(T_t_nom, T_controls_nom[:, 0], label="u (no filter)", color="C3", ls="-")
#     if T_x_nom_noisy is not None:
#         axes[0].plot(T_t_nom, T_x_nom_noisy[:, 0], label="x1 estimate (no filter)", color="C3", ls="--", alpha=0.6)
#         axes[1].plot(T_t_nom, T_x_nom_noisy[:, 1], label="x2 estimate (no filter)", color="C3", ls="--", alpha=0.6)
#     for ii, alpha in enumerate(alphas):
#         axes[0].plot(bT_t[ii], bT_x[ii, :, 0], label=f"x1 (QP {alpha})", color=f"C{ii}", ls="-")
#         axes[1].plot(bT_t[ii], bT_x[ii, :, 1], label=f"x2 (QP {alpha})", color=f"C{ii}", ls="-")
#         axes[2].plot(bT_t[ii], bT_controls[ii, :, 0], label=f"u (QP {alpha})", color=f"C{ii}", ls="-")
#         if bT_x_noisy[ii] is not None:
#             axes[0].plot(bT_t[ii], bT_x_noisy[ii, :, 0], label=f"x1 estimate (QP {alpha})", color=f"C{ii}", ls="--", alpha=0.6)
#             axes[1].plot(bT_t[ii], bT_x_noisy[ii, :, 1], label=f"x2 estimate (QP {alpha})", color=f"C{ii}", ls="--", alpha=0.6)
#     axes[0].set_ylabel("x1")
#     axes[1].set_ylabel("x2")
#     axes[2].set_ylabel("u")
#     axes[0].legend()
#     axes[1].legend()
#     axes[2].legend()
#     # axes[0].set_ylim(0.0, 2.5)
#     fig.savefig(plot_dir / "eval_traj_ctrl.pdf")
#     plt.close(fig)

#     # Vh along trajectories
#     fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
#     ax.plot(T_t_nom, T_Vh_nom, label="Vh (no filter)", color="C3", ls="-")
#     if T_x_nom_noisy is not None:
#         ax.plot(T_t_nom, T_Vh_nom_noisy, label="Vh estimate (no filter)", color="C3", ls="--", alpha=0.6)
#     for ii, alpha in enumerate(alphas):
#         ax.plot(bT_t[ii], bT_Vh[ii], label=f"Vh (QP {alpha})", color=f"C{ii}", ls="-")
#         if bT_x_noisy[ii] is not None:
#             ax.plot(bT_t[ii], bT_Vh_noisy[ii], label=f"Vh estimate (QP {alpha})", color=f"C{ii}", ls="--", alpha=0.6)
#     ax.set_ylabel("Vh")
#     ax.legend()
#     plt.savefig(plot_dir / "eval_traj_Vh.pdf")
#     plt.close(fig)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
