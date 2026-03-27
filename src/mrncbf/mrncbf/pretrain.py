import functools as ft
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from mrncbf.networks.temporal_residual import ResidualCorrection
from mrncbf.mrncbf.mrncbf_utils import (
    HistoryBuffer,
    FailureData,
    collect_failure_rollout,
    compute_ideal_residual,
    compute_worstcase_residual,
    _make_worstcase_fn,
)


def pretrain_Delta(
    Delta_net: ResidualCorrection,
    trajectories: list,         # List of FailureData from Stage 2
    pncbf,                      # PNCBF instance (for CBF evaluation)
    alpha_safe: float,
    alpha_unsafe: float,
    gamma1: float,
    gamma2: float,
    V_shift: float = 1e-3,
    window_len: int = 30,
    num_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_worstcase_samples: int = 64,
):
    """
    Stage 3: Supervised pre-training of Δ_θ on worst-case residual.
 
    For each (x̂, ε) pair from the trajectory data, we sample K candidate
    true states within the uncertainty ball and compute the worst-case
    CBF constraint discrepancy. This gives a deterministic, learnable target:
 
        Δ_target = R_worst(x̂, ε) - R_analytical(x̂)
 
    Unlike the realized-error target, this doesn't depend on the specific
    noise draw and should be a smooth function of (x̂, ε).
    """
    from loguru import logger as _logger
 
    nx, nh = pncbf.task.nx, pncbf.task.nh
 
    # Initialize network
    rng = jr.PRNGKey(42)
    feat_dim = nx + 1 + nh
    dummy_args = (
        jnp.zeros(nx),
        jnp.zeros(nh),
        jnp.zeros(nh),
        0.0,
        jnp.zeros((window_len, feat_dim)),
    )
    params = Delta_net.init(rng, *dummy_args)
 
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
 
    # ---- Build dataset using worst-case targets ----
    _logger.info("  Computing worst-case targets...")
 
    # Build JIT-compiled function (compiled once, fast for all samples)
    worstcase_fn = _make_worstcase_fn(
        pncbf, alpha_safe, alpha_unsafe,
        gamma1, gamma2, V_shift, n_worstcase_samples,
    )
 
    all_x_hat = []
    all_h_Lgh = []
    all_h_V = []
    all_eps = []
    all_hist = []
    all_target = []
 
    # Subsample timesteps to keep dataset manageable
    stride = 3
 
    n_traj = len(trajectories)
    for traj_idx, traj in enumerate(trajectories):
        T = traj.x_true.shape[0]
 
        if traj_idx % 50 == 0:
            _logger.info(f"    Processing trajectory {traj_idx}/{n_traj}...")
 
        for t in range(window_len, T, stride):
            x_hat_t = traj.x_hat[t]
            eps_t = float(traj.epsilon[t])
            u_t = traj.u_applied[t]
 
            if jnp.any(jnp.isnan(x_hat_t)) or jnp.any(jnp.isnan(u_t)):
                continue
 
            rng, wc_key = jr.split(rng)
            h_Delta_tgt, h_Lgh_norm, h_V_hat = worstcase_fn(
                x_hat_t, u_t, eps_t, wc_key
            )
 
            if jnp.any(jnp.isnan(h_Delta_tgt)) or jnp.any(jnp.isnan(h_Lgh_norm)):
                continue
 
            # Build history window
            hist_x = traj.x_hat[t - window_len:t]
            hist_eps = traj.epsilon[t - window_len:t, None]
            hist_Lgh = traj.h_Lgh_norm[t - window_len:t]
            history = jnp.concatenate([hist_x, hist_eps, hist_Lgh], axis=-1)
 
            if jnp.any(jnp.isnan(history)):
                continue
 
            all_x_hat.append(x_hat_t)
            all_h_Lgh.append(h_Lgh_norm)
            all_h_V.append(h_V_hat)
            all_eps.append(eps_t)
            all_hist.append(history)
            all_target.append(h_Delta_tgt)
 
    if len(all_x_hat) == 0:
        raise ValueError("All training samples contain NaN! Check Stage 2 data quality.")
 
    dataset = {
        'x_hat': jnp.stack(all_x_hat),
        'h_Lgh': jnp.stack(all_h_Lgh),
        'h_V': jnp.stack(all_h_V),
        'eps': jnp.array(all_eps),
        'hist': jnp.stack(all_hist),
        'target': jnp.stack(all_target),
    }
    N = dataset['x_hat'].shape[0]
 
    total_possible = sum((t.x_true.shape[0] - window_len) // stride for t in trajectories)
    _logger.info(f"  Pretrain dataset: {N}/{total_possible} samples "
                 f"({N/max(total_possible,1):.1%} clean)")
 
    # Report target statistics BEFORE clipping
    tgt = dataset['target']
    _logger.info(f"  Target stats (raw): mean={float(jnp.nanmean(tgt)):.4f}, "
                 f"std={float(jnp.nanstd(tgt)):.4f}, "
                 f"min={float(jnp.nanmin(tgt)):.4f}, "
                 f"max={float(jnp.nanmax(tgt)):.4f}")
 
    # Clip targets to network's output range [-Delta_max, Delta_max]
    # This prevents outliers from dominating the loss
    Delta_max = Delta_net.Delta_max
    dataset['target'] = jnp.clip(dataset['target'], -Delta_max, Delta_max)
 
    tgt_clipped = dataset['target']
    _logger.info(f"  Target stats (clipped to ±{Delta_max}): "
                 f"mean={float(jnp.nanmean(tgt_clipped)):.4f}, "
                 f"std={float(jnp.nanstd(tgt_clipped)):.4f}")
 
    # ---- Training with Huber loss ----
    huber_delta = 1.0  # Transition point between L1 and L2 behavior
 
    @jax.jit
    def train_step(params, opt_state, batch_idx):
        def loss_fn(params):
            def single(i):
                pred = Delta_net.apply(
                    params,
                    dataset['x_hat'][i],
                    dataset['h_Lgh'][i],
                    dataset['h_V'][i],
                    dataset['eps'][i],
                    dataset['hist'][i],
                )
                # Huber loss: less sensitive to outliers than MSE
                diff = pred - dataset['target'][i]
                abs_diff = jnp.abs(diff)
                huber = jnp.where(
                    abs_diff <= huber_delta,
                    0.5 * diff**2,
                    huber_delta * (abs_diff - 0.5 * huber_delta),
                )
                return jnp.mean(huber)
            return jnp.mean(jax.vmap(single)(batch_idx))
 
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss
 
    for epoch in range(num_epochs):
        rng, shuffle_key = jr.split(rng)
        perm = jr.permutation(shuffle_key, N)
 
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N - batch_size, batch_size):
            batch_idx = perm[i:i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch_idx)
            epoch_loss += loss
            n_batches += 1
 
        if epoch % 20 == 0:
            print(f"  Pretrain epoch {epoch:4d}, loss: {epoch_loss / max(n_batches, 1):.6f}")
 
    return params


# def pretrain_Delta(
#     Delta_net: ResidualCorrection,
#     trajectories: list,         # List of FailureData from Stage 2
#     pncbf,                      # PNCBF instance (for CBF evaluation)
#     alpha_safe: float,
#     alpha_unsafe: float,
#     gamma1: float,
#     gamma2: float,
#     V_shift: float = 1e-3,
#     window_len: int = 30,
#     num_epochs: int = 200,
#     batch_size: int = 256,
#     lr: float = 1e-3,
#     n_worstcase_samples: int = 64,
# ):
#     """
#     Stage 3: Supervised pre-training of Δ_θ on worst-case residual.
 
#     For each (x̂, ε) pair from the trajectory data, we sample K candidate
#     true states within the uncertainty ball and compute the worst-case
#     CBF constraint discrepancy. This gives a deterministic, learnable target:
 
#         Δ_target = R_worst(x̂, ε) - R_analytical(x̂)
 
#     Unlike the realized-error target, this doesn't depend on the specific
#     noise draw and should be a smooth function of (x̂, ε).
#     """
#     from loguru import logger as _logger
 
#     nx, nh = pncbf.task.nx, pncbf.task.nh
 
#     # Initialize network
#     rng = jr.PRNGKey(42)
#     feat_dim = nx + 1 + nh
#     dummy_args = (
#         jnp.zeros(nx),
#         jnp.zeros(nh),
#         jnp.zeros(nh),
#         0.0,
#         jnp.zeros((window_len, feat_dim)),
#     )
#     params = Delta_net.init(rng, *dummy_args)
 
#     optimizer = optax.adam(lr)
#     opt_state = optimizer.init(params)
 
#     # ---- Build dataset using worst-case targets ----
#     _logger.info("  Computing worst-case targets (this may take a while)...")
 
#     all_x_hat = []
#     all_h_Lgh = []
#     all_h_V = []
#     all_eps = []
#     all_hist = []
#     all_target = []
 
#     # Subsample timesteps to keep computation tractable
#     # (computing Jacobians for every timestep of every trajectory is expensive)
#     stride = 3  # Use every 3rd timestep
 
#     n_traj = len(trajectories)
#     for traj_idx, traj in enumerate(trajectories):
#         T = traj.x_true.shape[0]
 
#         if traj_idx % 50 == 0:
#             _logger.info(f"    Processing trajectory {traj_idx}/{n_traj}...")
 
#         for t in range(window_len, T, stride):
#             x_hat_t = traj.x_hat[t]
#             eps_t = float(traj.epsilon[t])
#             u_t = traj.u_applied[t]
 
#             # Skip NaN states
#             if jnp.any(jnp.isnan(x_hat_t)) or jnp.any(jnp.isnan(u_t)):
#                 continue
 
#             rng, wc_key = jr.split(rng)
#             try:
#                 h_R_worst, h_Delta_tgt, h_Lgh_norm, h_V_hat = compute_worstcase_residual(
#                     pncbf, x_hat_t, u_t, eps_t,
#                     alpha_safe, alpha_unsafe,
#                     gamma1, gamma2, V_shift,
#                     n_samples=n_worstcase_samples,
#                     rng_key=wc_key,
#                 )
#             except Exception:
#                 continue
 
#             # Skip NaN results
#             if jnp.any(jnp.isnan(h_Delta_tgt)) or jnp.any(jnp.isnan(h_Lgh_norm)):
#                 continue
 
#             # Build history window
#             hist_x = traj.x_hat[t - window_len:t]
#             hist_eps = traj.epsilon[t - window_len:t, None]
#             hist_Lgh = traj.h_Lgh_norm[t - window_len:t]
#             history = jnp.concatenate([hist_x, hist_eps, hist_Lgh], axis=-1)
 
#             if jnp.any(jnp.isnan(history)):
#                 continue
 
#             all_x_hat.append(x_hat_t)
#             all_h_Lgh.append(h_Lgh_norm)
#             all_h_V.append(h_V_hat)
#             all_eps.append(eps_t)
#             all_hist.append(history)
#             all_target.append(h_Delta_tgt)
 
#     if len(all_x_hat) == 0:
#         raise ValueError("All training samples contain NaN! Check Stage 2 data quality.")
 
#     dataset = {
#         'x_hat': jnp.stack(all_x_hat),
#         'h_Lgh': jnp.stack(all_h_Lgh),
#         'h_V': jnp.stack(all_h_V),
#         'eps': jnp.array(all_eps),
#         'hist': jnp.stack(all_hist),
#         'target': jnp.stack(all_target),
#     }
#     N = dataset['x_hat'].shape[0]
 
#     total_possible = sum((t.x_true.shape[0] - window_len) // stride for t in trajectories)
#     _logger.info(f"  Pretrain dataset: {N}/{total_possible} samples "
#                  f"({N/max(total_possible,1):.1%} clean)")
 
#     # Report target statistics
#     tgt = dataset['target']
#     _logger.info(f"  Target stats: mean={float(jnp.nanmean(tgt)):.4f}, "
#                  f"std={float(jnp.nanstd(tgt)):.4f}, "
#                  f"min={float(jnp.nanmin(tgt)):.4f}, "
#                  f"max={float(jnp.nanmax(tgt)):.4f}")
 
#     # ---- Training ----
#     @jax.jit
#     def train_step(params, opt_state, batch_idx):
#         def loss_fn(params):
#             def single(i):
#                 pred = Delta_net.apply(
#                     params,
#                     dataset['x_hat'][i],
#                     dataset['h_Lgh'][i],
#                     dataset['h_V'][i],
#                     dataset['eps'][i],
#                     dataset['hist'][i],
#                 )
#                 return jnp.mean((pred - dataset['target'][i]) ** 2)
#             return jnp.mean(jax.vmap(single)(batch_idx))
 
#         loss, grads = jax.value_and_grad(loss_fn)(params)
#         updates, opt_state_new = optimizer.update(grads, opt_state, params)
#         params_new = optax.apply_updates(params, updates)
#         return params_new, opt_state_new, loss
 
#     for epoch in range(num_epochs):
#         rng, shuffle_key = jr.split(rng)
#         perm = jr.permutation(shuffle_key, N)
 
#         epoch_loss = 0.0
#         n_batches = 0
#         for i in range(0, N - batch_size, batch_size):
#             batch_idx = perm[i:i + batch_size]
#             params, opt_state, loss = train_step(params, opt_state, batch_idx)
#             epoch_loss += loss
#             n_batches += 1
 
#         if epoch % 20 == 0:
#             print(f"  Pretrain epoch {epoch:4d}, loss: {epoch_loss / max(n_batches, 1):.6f}")
 
#     return params