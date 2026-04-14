import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
import jax

import run_config.int_avoid.error_cfg
from pncbf.dyn.error import Error
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.plotting.contour_utils import centered_norm
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.pncbf.pncbf import PNCBF
from pncbf.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

def make_plots(plot_dir, state_history):
        # plot leader and follower trajectories BEV trajectories
    fig, ax = plt.subplots()
    ax.plot(state_history[:, 13], state_history[:, 14], color=f"C{1}", label=f"Leader")
    ax.plot(state_history[:, 16], state_history[:, 17], color=f"C{2}", label=f"Follower")
    ax.set(xlabel="X", ylabel="Y")
    ax.set_aspect("equal")
    ax.legend()
    fig.suptitle("Leader and Follower PNCBF BEV")
    fig.savefig(plot_dir / "traj_policy.png")
    plt.close(fig)

    # plot leader x, y, theta
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(state_history[:, 13].size), state_history[:, 13], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 14].size), state_history[:, 14], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 15].size), state_history[:, 15], color=f"C{1}")
    axes[0].set_ylabel("X")
    axes[1].set_ylabel("Y")
    axes[2].set_ylabel("THETA")
    plt.tight_layout()
    fig.suptitle("Leader Position")
    fig.savefig(plot_dir / "xytheta_leader.png")

    # plot follower x, y, theta
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(state_history[:, 16].size), state_history[:, 16], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 17].size), state_history[:, 17], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 18].size), state_history[:, 18], color=f"C{1}")
    axes[0].set_ylabel("X")
    axes[1].set_ylabel("Y")
    axes[2].set_ylabel("THETA")
    plt.tight_layout()
    fig.suptitle("Follower Position")
    fig.savefig(plot_dir / "xytheta_follower.png")


    # plot EX, EY, THETAREL
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(state_history[:, 0].size), state_history[:, 0], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 3].size), state_history[:, 3], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 8].size), state_history[:, 8], color=f"C{1}")
    axes[0].set_ylabel("EX")
    axes[1].set_ylabel("EY")
    axes[2].set_ylabel("THETAREL")
    plt.tight_layout()
    fig.savefig(plot_dir / "ex_ey_thetarel.png")

    # plot leader surge and yaw states
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(np.arange(state_history[:, 11].size), state_history[:, 11], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 12].size), state_history[:, 12], color=f"C{1}")
    axes[0].set_ylabel("SURGE")
    axes[1].set_ylabel("YAWRATE")
    fig.suptitle("Leader Internal States")
    plt.tight_layout()
    fig.savefig(plot_dir / "leader_surge_yaw.png")

    # plot follower surge and yaw states
    fig, axes = plt.subplots(4, 1)
    axes[0].plot(np.arange(state_history[:, 9].size), state_history[:, 9], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 6].size), state_history[:, 6], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 10].size), state_history[:, 10], color=f"C{1}")
    axes[3].plot(np.arange(state_history[:, 7].size), state_history[:, 7], color=f"C{1}")
    axes[0].set_ylabel("SURGE")
    axes[1].set_ylabel("EU")
    axes[2].set_ylabel("YAWRATE")
    axes[3].set_ylabel("ER")
    fig.suptitle("Follower Internal States")
    plt.tight_layout()
    fig.savefig(plot_dir / "follower_surge_yaw.png")

    # ex and derivatives plots
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(state_history[:, 0].size), state_history[:, 0], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 1].size), state_history[:, 1], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 2].size), state_history[:, 2], color=f"C{1}")
    axes[0].set_ylabel("EX")
    axes[1].set_ylabel("EX_DOT")
    axes[2].set_ylabel("EX_DDOT")
    fig.suptitle("EX and Derivatives")
    plt.tight_layout()
    fig.savefig(plot_dir / "ex_ders.png")

    # ey and derivatives plot
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(state_history[:, 3].size), state_history[:, 3], color=f"C{1}")
    axes[1].plot(np.arange(state_history[:, 4].size), state_history[:, 4], color=f"C{1}")
    axes[2].plot(np.arange(state_history[:, 5].size), state_history[:, 5], color=f"C{1}")
    axes[0].set_ylabel("EY")
    axes[1].set_ylabel("EX_DOT")
    axes[2].set_ylabel("EY_DDOT")
    fig.suptitle("EY and Derivatives")
    plt.tight_layout()
    fig.savefig(plot_dir / "ey_ders.png")



def main(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    seed = 0
    rng_key = jax.random.PRNGKey(seed)

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    task = Error()

    nom_pol = task.nom_pol_goto
    # nom_pol = task.nom_pol_zero

    CFG = run_config.int_avoid.error_cfg.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    # Plot how V varies along a trajectory.
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # error state
                         0, 0,                      # simplified leader state
                         0, 0, np.pi/2,             # leader position
                         task._mx, 0, np.pi/2],     # follower position
                         dtype=np.float32)
    T = 4000
    tf = T * task.dt
    noise_scale = 0.0

    # Original nominal policy.
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True, max_steps=5000)
    T_x_nom, T_t_nom, _, T_x_nom_noisy = jax2np(
        jax_jit(ft.partial(sim.rollout_plot, noise_scale=noise_scale, rng_key=rng_key))(x0)
    )

    alphas = np.array([0.001, 0.01, 0.1, 3.0])

    def int_pol_for_alpha(alpha_safe):
        alpha_unsafe = 10.0
        pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=10000)
        T_x, T_t, _, T_x_noisy = sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)
        return T_x, T_t, T_x_noisy

    logger.info("Sim pol for different alphas...")
    bT_x, bT_t = [], []
    bT_x_noisy = []
    for alpha in alphas:
        T_x, T_t, T_x_noisy = jax2np(jax_jit(int_pol_for_alpha)(alpha))
        bT_x.append(T_x)
        bT_t.append(T_t)
        bT_x_noisy.append(T_x_noisy)
    bT_x = np.stack(bT_x, axis=0)
    bT_t = np.stack(bT_t, axis=0)
    bT_x_noisy = np.stack(bT_x_noisy, axis=0)

    logger.info("bbh_Vh...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh = bbh_Vh.max(-1)

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 3], color="C3", ls="--", label="Nominal")
    if T_x_nom_noisy is not None:
        ax.plot(T_x_nom_noisy[:, 0], T_x_nom_noisy[:, 3], color="C3", ls=":", alpha=0.6, label="Nominal (noisy)")
    for ii, alpha in enumerate(alphas):
        ax.plot(bT_x[ii, :, 0], bT_x[ii, :, 3], color=f"C{ii}", lw=0.5, ls="--", label=f"QP ({alpha})", zorder=100)
        if bT_x_noisy[ii] is not None:
            ax.plot(bT_x_noisy[ii, :, 0], bT_x_noisy[ii, :, 3], color=f"C{ii}", lw=0.5, ls=":", alpha=0.6)
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.6, linewidths=1.0)
    task.plot_phase(ax)
    ax.legend()
    fig.savefig(plot_dir / "eval_phase.png")
    plt.close(fig)

    norm = centered_norm(bbh_Vh.min(), bbh_Vh.max())

    levels = 31
    figsize = (8, 4)
    fig, axes = plt.subplots(1, task.nh, figsize=figsize, layout="constrained", squeeze=False)
    axes = axes.flatten()
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        cs1 = ax.contour(
            bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
        )
        cbar = fig.colorbar(cs0, ax=ax)
        cbar.add_lines(cs1)
        task.plot_phase(ax)
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vh.png")
    plt.close(fig)

    # make nominal plots
    make_plots(plot_dir / "nominal", T_x_nom)

    # make policy plots
    make_plots(plot_dir / "policy", bT_x[-1])    

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
