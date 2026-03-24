import functools as ft
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from mrncbf.networks.temporal_residual import ResidualCorrection
from mrncbf.mrncbf.mrncbf_utils import HistoryBuffer, FailureData, collect_failure_rollout, compute_ideal_residual


def pretrain_Delta(
    Delta_net: ResidualCorrection,
    trajectories: list,         # List of FailureData from Stage 2
    pncbf,                      # For task info
    alpha_safe: float,
    alpha_unsafe: float,
    gamma1: float,
    gamma2: float,
    window_len: int = 30,
    num_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """
    Stage 3: Supervised pre-training of Δ_θ on the ideal residual.
 
    Loss = mean_i ‖Δ_θ_i(x̂, ε, history) - Δ_target_i‖²
    """
    nx, nh = pncbf.task.nx, pncbf.task.nh
 
    # Initialize network
    rng = jr.PRNGKey(42)
    feat_dim = nx + 1 + nh
    dummy_args = (
        jnp.zeros(nx),                         # x_hat
        jnp.zeros(nh),                         # h_Lgh_norm
        jnp.zeros(nh),                         # h_val
        0.0,                                    # epsilon
        jnp.zeros((window_len, feat_dim)),      # history_seq
    )
    params = Delta_net.init(rng, *dummy_args)
 
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
 
    # ---- Build dataset ----
    # Flatten trajectories into (x̂, h_Lgh_norm, h_V, ε, history, Δ_target)
    all_x_hat = []
    all_h_Lgh = []
    all_h_V = []
    all_eps = []
    all_hist = []
    all_target = []
 
    for traj in trajectories:
        T = traj.x_true.shape[0]
        h_Delta_tgt = compute_ideal_residual(
            traj, alpha_safe, alpha_unsafe, gamma1, gamma2
        )
 
        for t in range(window_len, T):
            # Skip this sample if any field has NaN
            target_t = h_Delta_tgt[t]
            x_hat_t = traj.x_hat[t]
            h_Lgh_t = traj.h_Lgh_norm[t]
            h_V_t = traj.h_V_hat[t]
            eps_t = traj.epsilon[t]
 
            if (jnp.any(jnp.isnan(target_t)) or jnp.any(jnp.isnan(x_hat_t))
                    or jnp.any(jnp.isnan(h_Lgh_t)) or jnp.any(jnp.isnan(h_V_t))
                    or jnp.isnan(eps_t)):
                continue
 
            # Build history window
            hist_x = traj.x_hat[t - window_len:t]              # (W, nx)
            hist_eps = traj.epsilon[t - window_len:t, None]     # (W, 1)
            hist_Lgh = traj.h_Lgh_norm[t - window_len:t]       # (W, nh)
            history = jnp.concatenate([hist_x, hist_eps, hist_Lgh], axis=-1)
 
            # Skip if history has NaN
            if jnp.any(jnp.isnan(history)):
                continue
 
            all_x_hat.append(x_hat_t)
            all_h_Lgh.append(h_Lgh_t)
            all_h_V.append(h_V_t)
            all_eps.append(eps_t)
            all_hist.append(history)
            all_target.append(target_t)
 
    if len(all_x_hat) == 0:
        raise ValueError("All training samples contain NaN! Check Stage 2 data quality.")
 
    dataset = {
        'x_hat': jnp.stack(all_x_hat),          # (N, nx)
        'h_Lgh': jnp.stack(all_h_Lgh),          # (N, nh)
        'h_V': jnp.stack(all_h_V),              # (N, nh)
        'eps': jnp.array(all_eps),               # (N,)
        'hist': jnp.stack(all_hist),             # (N, W, feat_dim)
        'target': jnp.stack(all_target),         # (N, nh)
    }
    N = dataset['x_hat'].shape[0]
 
    # Diagnostic: report how much data survived filtering
    from loguru import logger as _logger
    total_possible = sum(t.x_true.shape[0] - window_len for t in trajectories)
    _logger.info(f"  Pretrain dataset: {N}/{total_possible} samples "
                 f"({N/total_possible:.1%} clean, {total_possible - N} dropped as NaN)")
 
    # ---- Training ----
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
                return jnp.mean((pred - dataset['target'][i]) ** 2)
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