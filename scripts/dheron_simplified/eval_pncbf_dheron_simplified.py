import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
import jax

import run_config.int_avoid.dheron_simplified_cbf
from pncbf.dyn.dheron_simplified import DHeronSimplified
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.plotting.contour_utils import centered_norm
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.pncbf.pncbf import PNCBF
from pncbf.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

def make_plots(plot_dir, time, state_history, control_history):
    # plot leader and follower trajectories BEV trajectories
    fig, ax = plt.subplots()
    ax.plot(state_history[:, 3], state_history[:, 4], color=f"C{1}", label=f"Leader")
    ax.plot(state_history[:, 6], state_history[:, 7], color=f"C{2}", label=f"Follower")
    ax.set(xlabel="X", ylabel="Y")
    ax.set_aspect("equal")
    ax.legend()
    fig.suptitle("Leader and Follower PNCBF BEV")
    fig.savefig(plot_dir / "traj_policy.png")
    plt.close(fig)

    # plot leader x, y, theta
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(time, state_history[:, 3], color=f"C{1}")
    axes[1].plot(time, state_history[:, 4], color=f"C{1}")
    axes[2].plot(time, state_history[:, 5], color=f"C{1}")
    axes[0].set_ylabel("X")
    axes[1].set_ylabel("Y")
    axes[2].set_ylabel("THETA")
    plt.tight_layout()
    fig.suptitle("Leader Position")
    fig.savefig(plot_dir / "xytheta_leader.png")
    plt.close(fig)

    # plot follower x, y, theta
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(time, state_history[:, 6], color=f"C{1}")
    axes[1].plot(time, state_history[:, 7], color=f"C{1}")
    axes[2].plot(time, state_history[:, 8], color=f"C{1}")
    axes[0].set_ylabel("X")
    axes[1].set_ylabel("Y")
    axes[2].set_ylabel("THETA")
    plt.tight_layout()
    fig.suptitle("Follower Position")
    fig.savefig(plot_dir / "xytheta_follower.png")
    plt.close(fig)

    # plot EX, EY, THETAREL
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(time, state_history[:, 0], color=f"C{1}")
    axes[1].plot(time, state_history[:, 1], color=f"C{1}")
    axes[2].plot(time, state_history[:, 2], color=f"C{1}")
    axes[0].set_ylabel("EX")
    axes[1].set_ylabel("EY")
    axes[2].set_ylabel("THETAREL")
    plt.tight_layout()
    fig.savefig(plot_dir / "ex_ey_thetarel.png")
    plt.close(fig)

    # plot control values
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(time, control_history[:, 0], color=f"C{1}")
    axes[1].plot(time, control_history[:, 1], color=f"C{1}")
    axes[0].set_ylabel("SURGE_CMD")
    axes[1].set_ylabel("YAWRATE_CMD")
    fig.suptitle("Control Values")
    plt.tight_layout()
    fig.savefig(plot_dir / "control_values.png")
    plt.close(fig)

def eval_ckpt(ckpt_path: pathlib.Path):
    print(f"Evaluation: {ckpt_path}")

    jax_default_x32()
    set_logger_format()

    seed = 0
    rng_key = jax.random.PRNGKey(seed)

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / ckpt_path.stem)
    print(plot_dir)

    task = DHeronSimplified()
    nom_pol = task.nom_pol_straight

    CFG = run_config.int_avoid.dheron_simplified_cbf.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info(f"Loaded ckpt from {ckpt_path}!")

    x0 = np.array([
        0, 0, np.pi/2,
        0, 0, np.pi/2,
        task._mx, -5, np.pi/2
    ], dtype=np.float32)

    T = 8000
    tf = T * task.dt
    noise_scale = 0.0

    # Nominal rollout
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True, max_steps=10000)
    T_x_nom, T_t_nom, _, T_x_nom_noisy, T_controls_nom = jax2np(
        jax_jit(ft.partial(sim.rollout_plot, noise_scale=noise_scale, rng_key=rng_key))(x0)
    )

    alphas = np.array([0.001, 0.01, 0.1, 3.0])

    def int_pol_for_alpha(alpha_safe):
        alpha_unsafe = 10.0
        pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=20000)
        return sim.rollout_plot(x0, noise_scale=noise_scale, rng_key=rng_key)

    logger.info("Sim pol for different alphas...")
    bT_x, bT_t, bT_controls, bT_x_noisy = [], [], [], []

    for alpha in alphas:
        T_x, T_t, _, T_x_noisy, T_controls = jax2np(
            jax_jit(int_pol_for_alpha)(alpha)
        )
        bT_x.append(T_x)
        bT_t.append(T_t)
        bT_controls.append(T_controls)
        bT_x_noisy.append(T_x_noisy)

    bT_x = np.stack(bT_x, axis=0)
    bT_t = np.stack(bT_t, axis=0)
    bT_x_noisy = np.stack(bT_x_noisy, axis=0)

    # Compute value function contour
    logger.info("bbh_Vh...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh = bbh_Vh.max(-1)

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 3], color="C3", ls="--", label="Nominal")

    for ii, alpha in enumerate(alphas):
        ax.plot(bT_x[ii, :, 0], bT_x[ii, :, 3], color=f"C{ii}", lw=0.5, ls="--", label=f"QP ({alpha})")

    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor])
    task.plot_phase(ax)
    ax.legend()
    fig.savefig(plot_dir / "eval_phase.png")
    plt.close(fig)

    # Save plots
    for ii, alpha in enumerate(alphas):
        alpha_str = str(alpha)
        mkdir(plot_dir / alpha_str / "policy")
        mkdir(plot_dir / alpha_str/ "nominal")
        make_plots(plot_dir / alpha_str / "nominal", T_t_nom, T_x_nom, T_controls_nom)
        make_plots(plot_dir / alpha_str / "policy", bT_t[-1], bT_x[-1], bT_controls[-1])


def main(run_path: pathlib.Path):
    run_path = pathlib.Path(run_path)
    ckpt_dir = run_path / "ckpts"

    if not ckpt_dir.exists():
        raise ValueError(f"No ckpts directory found at {ckpt_dir}")

    target_steps = {"80000"}

    ckpt_paths = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name in target_steps
    ]


    logger.info(f"Found {len(ckpt_paths)} checkpoints.")

    for ckpt_path in ckpt_paths:
        eval_ckpt(ckpt_path)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)