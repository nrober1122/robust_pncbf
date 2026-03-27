import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pathlib
from jaxtyping import Float
import numpy as np

from mrncbf.qp.min_norm_cbf import rcbf_qp_learned
from mrncbf.networks.temporal_residual import ResidualCorrection
from mrncbf.pncbf.pncbf import _resolve_alpha as _resolve_alpha_local
from mrncbf.mrncbf.mrncbf_utils import rk4_step

class RolloutCarry(NamedTuple):
    x_true: jnp.ndarray         # (nx,)
    history: jnp.ndarray         # (window_len, feat_dim)
    total_cost: float
 
 
def make_stage4_loss(
    pncbf,
    Delta_net: ResidualCorrection,
    alpha_safe: float,
    alpha_unsafe: float,
    gamma1: float,
    gamma2: float,
    R_floor: float,
    V_shift: float,
    episode_length: int,
    window_len: int,
    # Cost weights
    w_safety: float = 100.0,
    w_conserv: float = 0.1,
    w_reg: float = 0.01,
):
    """
    Returns a function:  loss(Delta_params, x0, epsilon, rng_key) -> scalar
 
    that runs a differentiable rollout and returns the total cost.
    Gradients w.r.t. Delta_params flow through the QP via jaxproxqp.
    """
    Vh_apply = ft.partial(pncbf.get_Vh, params=pncbf.Vh.params)
    task = pncbf.task
    nx, nh, nu = task.nx, task.nh, task.nu
    feat_dim = nx + 1 + nh
 
    def episode_loss(Delta_params, x0_true, epsilon, rng_key):
 
        init_carry = RolloutCarry(
            x_true=x0_true,
            history=jnp.zeros((window_len, feat_dim)),
            total_cost=0.0,
        )
 
        noise_keys = jr.split(rng_key, episode_length)
 
        def step_fn(carry, noise_key):
            x_true = carry.x_true
 
            # 1. Measurement noise
            delta = jax.random.uniform(
                noise_key, shape=(nx,), minval=-epsilon, maxval=epsilon
            )
            x_hat = x_true + delta
 
            # 2. CBF ingredients at x̂
            h_V = Vh_apply(x_hat) + V_shift
            hx_Vx = jax.jacobian(Vh_apply)(x_hat)
            f = task.f(x_hat)
            G = task.G(x_hat)
            u_nom = pncbf.nom_pol(x_hat)
 
            alpha = _resolve_alpha_local(h_V, alpha_safe, alpha_unsafe)
 
            h_LG = hx_Vx @ G
            h_Lgh_norm = jnp.linalg.norm(h_LG, axis=-1)
 
            # 3. Learned residual
            h_Delta = Delta_net.apply(
                Delta_params,
                x_hat, h_Lgh_norm, h_V, epsilon,
                carry.history,
            )
 
            # 4. Solve QP
            u_opt, r, sol = rcbf_qp_learned(
                alpha, task.u_min, task.u_max,
                h_V, hx_Vx, f, G, u_nom,
                gamma1, gamma2, h_Delta, R_floor,
            )
            u_safe = task.chk_u(u_opt)
 
            # 5. Step TRUE dynamics
            dynamics_fn = lambda x, u: task.f(x) + task.G(x) @ u
            x_true_next = rk4_step(dynamics_fn, x_true, u_safe, task.dt)
 
            # 6. Cost
            h_V_true_next = Vh_apply(x_true_next) + V_shift
            h_h_true_next = task.h_components(x_true_next)
 
            # Safety: penalize actual constraint violations
            safety_cost = w_safety * jnp.sum(
                jnp.maximum(0.0, h_h_true_next) ** 2
            )
 
            # Conservatism: penalize control deviation
            conserv_cost = w_conserv * jnp.sum((u_safe - u_nom) ** 2)
 
            # Regularization: prefer small corrections
            reg_cost = w_reg * jnp.sum(h_Delta ** 2)
 
            step_cost = safety_cost + conserv_cost + reg_cost
 
            # 7. Update history
            new_entry = jnp.concatenate([
                x_hat, jnp.array([epsilon]), h_Lgh_norm
            ])
            new_history = jnp.concatenate([
                carry.history[1:], new_entry[None, :]
            ], axis=0)
 
            new_carry = RolloutCarry(
                x_true=x_true_next,
                history=new_history,
                total_cost=carry.total_cost + step_cost,
            )
            return new_carry, step_cost
 
        final_carry, step_costs = jax.lax.scan(step_fn, init_carry, noise_keys)
        return final_carry.total_cost
 
    return episode_loss
 
 
def stage4_train(
    pncbf,
    Delta_net: ResidualCorrection,
    Delta_params_init,
    alpha_safe: float,
    alpha_unsafe: float,
    gamma1: float,
    gamma2: float,
    R_floor: float = 0.0,
    V_shift: float = 1e-3,
    noise_schedule: list = [0.01, 0.05, 0.1, 0.2],
    num_iterations: int = 5000,
    episodes_per_iter: int = 16,
    episode_length: int = 200,
    window_len: int = 30,
    lr: float = 3e-4,
    seed: int = 0,
    # ── Logging and checkpointing hooks ──
    log_every: int = 50,
    eval_every: int = 500,
    ckpt_every: int = 1000,
    wandb_log: bool = False,
    ckpt_dir=None,
    eval_fn=None,           # Callable(Delta_params) -> dict of metrics
):
    """Stage 4: End-to-end fine-tuning through differentiable QP rollouts."""
    try:
        import wandb as _wandb
    except ImportError:
        _wandb = None
 
    from loguru import logger
 
    episode_loss_fn = make_stage4_loss(
        pncbf, Delta_net,
        alpha_safe, alpha_unsafe,
        gamma1, gamma2, R_floor, V_shift,
        episode_length, window_len,
    )
 
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(Delta_params_init)
    Delta_params = Delta_params_init
    rng = jr.PRNGKey(seed)
 
    # Batch over multiple episodes
    batched_loss = jax.vmap(episode_loss_fn, in_axes=(None, 0, None, 0))
 
    @jax.jit
    def train_step(Delta_params, opt_state, x0s, epsilon, rng_keys):
        def batch_mean_loss(params):
            costs = batched_loss(params, x0s, epsilon, rng_keys)
            return jnp.mean(costs)
 
        loss, grads = jax.value_and_grad(batch_mean_loss)(Delta_params)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, Delta_params)
        params_new = optax.apply_updates(Delta_params, updates)
        return params_new, opt_state_new, loss, grad_norm
 
    for iteration in range(num_iterations):
        rng, key_x0, key_eps, key_rollout = jr.split(rng, 4)
 
        # Sample initial states
        x0s = pncbf.task.sample_train_x0(key_x0, episodes_per_iter)
 
        # Noise curriculum: easy → medium → full range
        progress = iteration / num_iterations
        if progress < 0.3:
            epsilon = noise_schedule[0]
        elif progress < 0.7:
            epsilon = noise_schedule[len(noise_schedule) // 2]
        else:
            epsilon = jr.choice(key_eps, jnp.array(noise_schedule))
 
        ep_keys = jr.split(key_rollout, episodes_per_iter)
 
        Delta_params, opt_state, loss, grad_norm = train_step(
            Delta_params, opt_state, x0s, epsilon, ep_keys
        )
 
        # ── Logging ──
        if iteration % log_every == 0:
            loss_val = float(loss)
            grad_val = float(grad_norm)
            eps_val = float(epsilon)
            logger.info(
                f"  [Stage 4] iter {iteration:5d}  ε={eps_val:.3f}  "
                f"loss={loss_val:.4f}  |∇|={grad_val:.4f}"
            )
            if wandb_log and _wandb is not None:
                _wandb.log({
                    "stage4/loss": loss_val,
                    "stage4/grad_norm": grad_val,
                    "stage4/epsilon": eps_val,
                    "stage4/progress": progress,
                }, step=iteration)
 
        # ── Evaluation ──
        if iteration % eval_every == 0 and eval_fn is not None:
            logger.info("  [Stage 4] Evaluating...")
            eval_results = eval_fn(Delta_params=Delta_params)
            for key, vals in sorted(eval_results.items()):
                logger.info(
                    f"    {key:<30s}  "
                    f"anal={vals['analytical_safety_rate']:.0%}  "
                    f"learned={vals['learned_safety_rate']:.0%}  "
                    f"Δ={vals['avg_Delta']:+.3f}"
                )
                if wandb_log and _wandb is not None:
                    _wandb.log({
                        f"stage4_eval/{key}/learned_safety": vals['learned_safety_rate'],
                        f"stage4_eval/{key}/analytical_safety": vals['analytical_safety_rate'],
                        f"stage4_eval/{key}/avg_Delta": vals['avg_Delta'],
                    }, step=iteration)
 
        # ── Checkpointing ──
        if iteration % ckpt_every == 0 and ckpt_dir is not None:
            import pickle
            ckpt_path = pathlib.Path(ckpt_dir) / f"stage4_iter{iteration}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path / "Delta_params.pkl", "wb") as f:
                pickle.dump(
                    jax.tree.map(lambda x: np.array(x), Delta_params), f
                )
            logger.info(f"  [Stage 4] Saved checkpoint to {ckpt_path}")
 
    return Delta_params