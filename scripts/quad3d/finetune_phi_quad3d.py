"""RL fine-tune of PhiNet for the 3D quadrotor NMR-CBF.

Initializes from a supervised-trained `phi_params.pkl` (output of the notebook's
`nmr-train` cell) and shrinks φ where safety slack exists, recovering CBF-like
aggression at low ε while preserving the φ ≥ 0 structure (softplus head).

Unlike the dbint2d variant, the reward is **closeness to the nominal control**
(‖u − u_nom‖²) rather than progress toward a goal.  This matches the CBF
safety filter's defined objective `min ‖u − u_nom‖²` and gives a denser,
lower-variance training signal — every step contributes, not just net
displacement.

Run:
    python scripts/quad3d/finetune_phi_quad3d.py --name rl_v1 --wandb
"""
from __future__ import annotations

import pathlib
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import typer
from loguru import logger

# scripts/quad3d/ → repo root /src is two parents up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mrncbf.utils.jax_utils import jax_default_x32
from mrncbf.utils.logging import set_logger_format

import quad3d_quad as quad
from quad3d_quad import (
    ALPHA_QP,
    DT,
    EPS_MAX,
    GOAL_MC,
    NMR_WEIGHTS_DIR,
    NH,
    NU,
    NX,
    OBSTACLE,
    B_hocbf,
    G,
    f,
    h_raw,
    nom_pol_goto,
    phi_net,
    task,
    u_lb,
    u_ub,
)


# ─── Initial-state sampler ────────────────────────────────────────────────────
# 12-D box covering modest hover-flight envelope.  Angles bounded to ±0.3 rad
# to stay clear of HOCBF singularities (matches the PhiNet training range in
# the notebook's `nmr-train` cell).

_X_LO = jnp.array([-2.0, -2.0,  0.5, -1.5, -1.5, -1.5,
                   -0.3, -0.3, -0.5, -1.0, -1.0, -0.5])
_X_HI = jnp.array([ 2.0,  2.0,  2.5,  1.5,  1.5,  1.5,
                    0.3,  0.3,  0.5,  1.0,  1.0,  0.5])


def sample_x0(key: jax.Array, n: int) -> jax.Array:
    """Sample n initial states uniformly in the envelope, rejecting any inside
    the obstacle sphere (with a 0.1 m buffer)."""
    cands = _X_LO + jr.uniform(key, (n * 4, NX)) * (_X_HI - _X_LO)
    cx, cy, cz, r_obs = OBSTACLE
    d2 = (cands[:, 0] - cx) ** 2 + (cands[:, 1] - cy) ** 2 + (cands[:, 2] - cz) ** 2
    outside = d2 > (r_obs + 0.1) ** 2
    return cands[outside][:n]


# ─── Differentiable min-norm CBF projection (multi-constraint) ───────────────
# Same single-active-constraint approximation used in the dbint version, but
# generalised to nh > 1.  Argmax of deficit selects the most-violating
# constraint at the current state and we project u_nom against that one.
# Reverse-mode differentiable through the rollout (argmax has zero gradient;
# gradient flows through the active row's a and rhs).

def diff_min_norm_cbf(h_B, J_B, f_x, G_x, u_nom, phi):
    """Closed-form min-norm projection for the φ-tightened CBF constraint.

    Solves (approximately, ignoring the box for the projection step then
    clipping):
        min_u  ½‖u − u_nom‖²
        s.t.   J_B·G_x · u + (J_B·f_x + α·h_B + φ) ≤ 0,   u_lb ≤ u ≤ u_ub.

    For nh > 1, picks the constraint with the largest deficit and projects
    against just that one.  Equivalent to `min_norm_cbf` when the active set
    is a singleton and the box is non-binding; box-feasible (still safe) when
    the box binds.
    """
    a       = J_B @ G_x                                  # (nh, nu)
    rhs     = J_B @ f_x + ALPHA_QP * h_B + phi           # (nh,)
    deficit = a @ u_nom + rhs                             # (nh,) — positive = u_nom violates
    active  = jnp.argmax(deficit)
    a_act   = a[active]
    gain    = jnp.maximum(deficit[active], 0.0) / (jnp.dot(a_act, a_act) + 1e-12)
    u_proj  = u_nom - gain * a_act
    return jnp.clip(u_proj, u_lb, u_ub)


# ─── Episode loss ────────────────────────────────────────────────────────────
# Per-step cost:
#     w_safety * safety  +  w_unom * ‖u − u_nom‖²  +  w_reg * mean(φ²)
# All terms are added (no negation): we want to MINIMIZE deviation from u_nom,
# which matches the CBF filter's design objective.

def make_episode_loss(episode_length: int, w_safety: float, w_unom: float, w_reg: float):
    def episode_loss(phi_params, x0, eps, rng_key):
        noise_keys = jr.split(rng_key, episode_length)

        def step(x, noise_key):
            delta  = jr.uniform(noise_key, (NX,), minval=-eps, maxval=eps)
            xhat   = x + delta
            phi    = phi_net.apply({"params": phi_params}, xhat, eps)       # (nh,)
            h_B    = B_hocbf(xhat)
            J_B    = jax.jacobian(B_hocbf)(xhat)
            u_nom  = nom_pol_goto(xhat, GOAL_MC)
            u      = diff_min_norm_cbf(h_B, J_B, f(xhat), G(xhat), u_nom, phi)
            x_next = x + DT * (f(x) + G(x) @ u)

            h_next = h_raw(x_next)
            delta_buf = 0.03   # safety buffer
            safety = jnp.sum(jnp.maximum(0.0, h_next + delta_buf) ** 2)

            u_dev  = jnp.sum((u - u_nom) ** 2)

            # mean over nh so w_reg is comparable across systems with different nh
            reg    = jnp.mean(phi ** 2)

            step_cost = w_safety * safety + w_unom * u_dev + w_reg * reg
            return x_next, (step_cost, phi.mean(), safety, u_dev)

        _, (step_costs, phis, safeties, u_devs) = jax.lax.scan(step, x0, noise_keys)
        return step_costs.sum(), (phis.mean(), safeties.sum(), u_devs.sum())

    return episode_loss


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(
    name: str = typer.Option(..., help="Name of the run."),
    group: str = typer.Option(None, help="Wandb group."),
    init_ckpt: pathlib.Path = typer.Option(
        None,
        help="Path to phi_params.pkl. Defaults to the supervised checkpoint in NMR_WEIGHTS_DIR.",
    ),
    seed: int = 0,
    num_iterations: int = 1000,
    episodes_per_iter: int = 32,
    episode_length: int = 500,
    lr: float = 3e-5,
    grad_clip: float = 1.0,
    w_safety: float = 1000.0,
    w_unom: float = 5.0,
    w_reg: float = 0.01,
    log_every: int = 50,
    ckpt_every: int = 1000,
    wandb: bool = typer.Option(False, "--wandb", help="Enable Wandb logging."),
):
    jax_default_x32()
    set_logger_format()

    if init_ckpt is None:
        init_ckpt = NMR_WEIGHTS_DIR / "phi_params.pkl"
    if not init_ckpt.exists():
        raise FileNotFoundError(
            f"phi_params init checkpoint not found at {init_ckpt}. "
            "Run the supervised `nmr-train` cell in quad3d_cbf_compare.ipynb first."
        )
    logger.info(f"Loading phi_params from {init_ckpt}")
    phi_params = quad.load_phi_params(init_ckpt)

    out_dir = NMR_WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if wandb:
        import wandb as _wandb
        _wandb.init(
            project="finetune_phi_quad3d",
            name=name,
            group=group,
            config={
                "seed": seed,
                "num_iterations": num_iterations,
                "episodes_per_iter": episodes_per_iter,
                "episode_length": episode_length,
                "lr": lr,
                "grad_clip": grad_clip,
                "w_safety": w_safety,
                "w_unom": w_unom,
                "w_reg": w_reg,
                "init_ckpt": str(init_ckpt),
            },
        )
    else:
        _wandb = None

    # ── Build loss + optimizer ──
    episode_loss = make_episode_loss(episode_length, w_safety, w_unom, w_reg)
    batched_loss = jax.vmap(episode_loss, in_axes=(None, 0, None, 0))

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr),
    )
    opt_state = optimizer.init(phi_params)

    @jax.jit
    def train_step(phi_params, opt_state, x0s, eps, rng_keys):
        def batch_mean_loss(p):
            losses, (phis, safeties, u_devs) = batched_loss(p, x0s, eps, rng_keys)
            return jnp.mean(losses), (jnp.mean(phis), jnp.mean(safeties), jnp.mean(u_devs))

        (loss, (mean_phi, mean_safety, mean_udev)), grads = jax.value_and_grad(
            batch_mean_loss, has_aux=True
        )(phi_params)
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
        updates, opt_state_new = optimizer.update(grads, opt_state, phi_params)
        params_new = optax.apply_updates(phi_params, updates)
        return params_new, opt_state_new, loss, gnorm, mean_phi, mean_safety, mean_udev

    # ── Train loop ──
    rng = jr.PRNGKey(seed)
    t0 = time.time()
    for it in range(num_iterations):
        rng, k_x0, k_eps, k_roll = jr.split(rng, 4)
        x0s = sample_x0(k_x0, episodes_per_iter)
        # Bias eps toward smaller values (** 5) — matches dbint variant.
        eps = jr.uniform(k_eps, (NX,)) ** 5 * EPS_MAX
        ep_keys = jr.split(k_roll, episodes_per_iter)

        phi_params, opt_state, loss, gnorm, mean_phi, mean_safety, mean_udev = train_step(
            phi_params, opt_state, x0s, eps, ep_keys
        )

        if it % log_every == 0:
            loss_v = float(loss)
            gnorm_v = float(gnorm)
            phi_v = float(mean_phi)
            safety_v = float(mean_safety)
            udev_v = float(mean_udev)
            eps_avg = float(jnp.mean(eps))
            logger.info(
                f"  iter {it:5d}  ε̄={eps_avg:.3f}  "
                f"loss={loss_v:+.3e}  |∇|={gnorm_v:.2e}  "
                f"φ̄={phi_v:.3e}  safety={safety_v:.2e}  ‖Δu‖²={udev_v:.2e}"
            )
            if _wandb is not None:
                _wandb.log(
                    {
                        "finetune_phi/loss": loss_v,
                        "finetune_phi/grad_norm": gnorm_v,
                        "finetune_phi/mean_phi": phi_v,
                        "finetune_phi/safety_violation": safety_v,
                        "finetune_phi/u_dev_sq": udev_v,
                        "finetune_phi/epsilon_mean": eps_avg,
                    },
                    step=it,
                )

        if it > 0 and it % ckpt_every == 0:
            ckpt_path = out_dir / f"phi_params_finetune_{name}_iter{it}.pkl"
            with open(ckpt_path, "wb") as f_:
                pickle.dump(jax.tree.map(lambda a: np.array(a), phi_params), f_)
            logger.info(f"  Saved checkpoint to {ckpt_path}")

    final_path = out_dir / f"phi_params_finetune_{name}.pkl"
    with open(final_path, "wb") as f_:
        pickle.dump(jax.tree.map(lambda a: np.array(a), phi_params), f_)
    logger.info(f"Done in {time.time() - t0:.1f}s. Final checkpoint: {final_path}")

    if _wandb is not None:
        _wandb.finish()


if __name__ == "__main__":
    typer.run(main)
