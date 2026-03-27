import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
import jax


import run_config.int_avoid.singleintwall_cfg
from mrncbf.dyn.singleint_wall import SingleIntWall
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

    task = SingleIntWall()

    nom_pol = task.nom_pol_osc

    CFG = run_config.int_avoid.singleintwall_cfg.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    noise_scale = 0.0
    noise_scales = np.array([0.0, 0.3, 0.6])
    x0_est = np.array([-0.5, 0.0])
    # x0 = np.array([0.5, 0.0])
    x0 = x0_est
    T = 40
    tf = T * task.dt
    

    # Original nominal policy.
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True)
    T_x_nom, T_t_nom, _, T_x_nom_noisy, T_controls_nom = jax2np(
        jax_jit(
            ft.partial(
                sim.rollout_plot,
                noise_scale=noise_scale,
                rng_key=rng_key,
            )
        )(x0)
    )
    if T_x_nom_noisy is None:
        T_x_nom_noisy = T_x_nom

    alphas = np.array([1.0])
    # pol_types = np.array([0, 1])

    # def int_pol_for_alpha(alpha_safe):
    #     # alpha_unsafe = 10.0
    #     alpha_unsafe = alpha_safe
    #     # pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
    #     pol = ft.partial(alg.get_rcbf_qp_control, alpha_safe, alpha_unsafe, nnv_filter=None, epsilon=noise_scale, V_shift=1e-2)
    #     sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
    #     T_x, T_t, _, T_x_noisy, T_controls = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
    #     return T_x, T_t, T_x_noisy, T_controls
    
    def int_pol_for_noise_scale(noise_scale):
        # alpha_unsafe = 10.0
        alpha_safe = 0.5
        alpha_unsafe = alpha_safe
        x0 = x0_est + noise_scale
        # pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
        pol = ft.partial(alg.get_rcbf_qp_control, alpha_safe, alpha_safe, nnv_filter=None, epsilon=noise_scale, V_shift=1e-2)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
        T_x, T_t, _, T_x_noisy, T_controls = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
        return T_x, T_t, T_x_noisy, T_controls
    
    # def test_diff_pols(pol_type):
    #     alpha_safe = 1.0
    #     alpha_unsafe = 1.0
    #     if pol_type == 1.0:
    #         print("Testing CBF control...")
    #         pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
    #         # pol = ft.partial(alg.get_cbf_control, alpha_safe)
    #     else:
    #         print("Testing RCBF QP control...")
    #         pol = ft.partial(alg.get_rcbf_qp_control, alpha_safe, alpha_unsafe, nnv_filter=None, epsilon=noise_scale, V_shift=1e-2)
    #     sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
    #     T_x, T_t, _, T_x_noisy, T_controls = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
    #     return T_x, T_t, T_x_noisy, T_controls

    logger.info("Sim pol for different alphas...")
    bT_x, bT_t, bT_x_noisy, bT_controls = [], [], [], []
    for noise_scale in noise_scales:
        T_x, T_t, T_x_noisy, T_controls = jax2np(int_pol_for_noise_scale(noise_scale))
        bT_x.append(T_x)
        bT_t.append(T_t)
        bT_x_noisy.append(T_x_noisy if T_x_noisy is not None else T_x)
        bT_controls.append(T_controls)
    # for pol_type in alphas:
    #     T_x, T_t, T_x_noisy, T_controls = jax2np(test_diff_pols(pol_type))
    #     bT_x.append(T_x)
    #     bT_t.append(T_t)
    #     bT_x_noisy.append(T_x_noisy if T_x_noisy is not None else T_x)
    #     bT_controls.append(T_controls)
    # for alpha in alphas:
    #     T_x, T_t, T_x_noisy, T_controls = jax2np(jax_jit(int_pol_for_alpha)(alpha))
    #     bT_x.append(T_x)
    #     bT_t.append(T_t)
    #     if T_x_noisy is not None:
    #         bT_x_noisy.append(T_x_noisy)
    #     else:
    #         bT_x_noisy.append(T_x)
    #     bT_controls.append(T_controls)
    bT_x = np.stack(bT_x, axis=0)
    bT_t = np.stack(bT_t, axis=0)
    bT_x_noisy = np.stack(bT_x_noisy, axis=0)
    bT_controls = np.stack(bT_controls, axis=0)

    logger.info("bTh_Vh for trajectories...")
    Th_Vh_nom = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom, 0))).squeeze()
    Th_Vh_nom_noisy = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(np.expand_dims(T_x_nom_noisy, 0))).squeeze()
    T_Vh_nom = np.max(Th_Vh_nom, axis=-1)
    T_Vh_nom_noisy = np.max(Th_Vh_nom_noisy, axis=-1)

    bTh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bT_x))
    bTh_Vh_noisy = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bT_x_noisy))
    bT_Vh = np.max(bTh_Vh, axis=-1)
    bT_Vh_noisy = np.max(bTh_Vh_noisy, axis=-1)

    # get_contour_x0 returns the standard (bb_x, bb_Xs, bb_Ys) 2D mesh.
    # The dummy state collapses the Y axis to a single row, so we slice [0, :]
    # to recover the 1D position axis for plotting.
    logger.info("bbh_Vh for contour grid...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh = bbh_Vh.max(-1)

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    # Vh: squeeze out the degenerate dummy axis and plot as 1D lines.
    fig, axes = plt.subplots(1, task.nh, figsize=(8, 3), layout="constrained")
    for ii, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(bb_Xs[0, :], bbh_Vh[0, :, ii], label=h_labels[ii])
        ax.axhline(0.0, color=PlotStyle.ZeroColor, lw=1.0, alpha=0.98)
        ax.axvline(-task.pos_wall, color="k", ls="--", lw=0.8, label="Wall")
        ax.axvline(task.pos_wall, color="k", ls="--", lw=0.8)
        ax.set_xlabel(task.x_labels[0])
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vh.pdf")
    plt.close(fig)

    # States and controls over time.
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), layout="constrained")
    axes[0].plot(T_t_nom, T_x_nom[:, 0], label="x (no filter)", color="C3", ls="-")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.8, label="Wall")
    axes[0].axhline(-1.0, color="k", ls=":", lw=0.8)
    axes[1].plot(T_t_nom, T_controls_nom[:, 0], label="u (no filter)", color="C3", ls="-")
    if T_x_nom_noisy is not None:
        axes[0].plot(T_t_nom, T_x_nom_noisy[:, 0], label="x estimate (no filter)", color="C3", ls="--", alpha=0.6)
    for ii, alpha in enumerate(noise_scales):
        axes[0].plot(bT_t[ii], bT_x[ii, :, 0], label=f"x (QP {alpha})", color=f"C{ii}", ls="-")
        axes[1].plot(bT_t[ii], bT_controls[ii, :, 0], label=f"u (QP {alpha})", color=f"C{ii}", ls="-")
        if bT_x_noisy[ii] is not None:
            axes[0].plot(bT_t[ii], bT_x_noisy[ii, :, 0], label=f"x estimate (QP {alpha})", color=f"C{ii}", ls="--", alpha=0.6)
    axes[0].set_ylabel("p")
    axes[1].set_ylabel("u")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(plot_dir / "eval_traj_ctrl.pdf")
    plt.close(fig)

    # Vh along trajectories.
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(T_t_nom, T_Vh_nom, label="Vh (no filter)", color="C3", ls="-")
    if T_x_nom_noisy is not None:
        ax.plot(T_t_nom, T_Vh_nom_noisy, label="Vh estimate (no filter)", color="C3", ls="--", alpha=0.6)
    for ii, alpha in enumerate(alphas):
        ax.plot(bT_t[ii], bT_Vh[ii], label=f"Vh (QP {alpha})", color=f"C{ii}", ls="-")
        if bT_x_noisy[ii] is not None:
            ax.plot(bT_t[ii], bT_Vh_noisy[ii], label=f"Vh estimate (QP {alpha})", color=f"C{ii}", ls="--", alpha=0.6)
    ax.axhline(0.0, color=PlotStyle.ZeroColor, lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel("Vh")
    ax.legend()
    plt.savefig(plot_dir / "eval_traj_Vh.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)