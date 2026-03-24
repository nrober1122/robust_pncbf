import functools as ft
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from mrncbf.dyn.dyn_types import State

class HistoryBuffer(NamedTuple):
    """Fixed-size ring buffer for GRU input, carried through rollouts."""
    data: jnp.ndarray   # (window_len, feat_dim)
 
    @classmethod
    def create(cls, window_len: int, nx: int, nh: int):
        """feat_dim = nx + 1 (epsilon) + nh (Lgh norms)"""
        feat_dim = nx + 1 + nh
        return cls(data=jnp.zeros((window_len, feat_dim)))
 
    def append(self, x_hat, epsilon, h_Lgh_norm):
        """Shift buffer and append new entry."""
        new_entry = jnp.concatenate([
            x_hat,                       # (nx,)
            jnp.array([epsilon]),        # (1,)
            h_Lgh_norm,                  # (nh,)
        ])
        new_data = jnp.concatenate([
            self.data[1:],               # Drop oldest
            new_entry[None, :],          # Append newest
        ], axis=0)
        return self._replace(data=new_data)


class FailureData(NamedTuple):
    """Per-timestep data from a rollout under measurement noise."""
    x_true: jnp.ndarray        # (nx,)
    x_hat: jnp.ndarray         # (nx,)
    epsilon: float
    h_V_true: jnp.ndarray      # (nh,) CBF values at true state
    h_V_hat: jnp.ndarray       # (nh,) CBF values at estimated state
    h_Lgh_norm: jnp.ndarray    # (nh,)
    h_Lfh_true: jnp.ndarray    # (nh,) Lie derivs at true state
    h_Lfh_hat: jnp.ndarray     # (nh,) Lie derivs at est. state
    h_Lgh_u_true: jnp.ndarray  # (nh,) Lgh @ u at true state
    h_Lgh_u_hat: jnp.ndarray   # (nh,) Lgh @ u at est. state
    u_applied: jnp.ndarray     # (nu,)


def collect_failure_rollout(
    pncbf,                  # PNCBF instance
    x0_true: State,
    epsilon: float,
    episode_length: int,
    alpha_safe: float,
    alpha_unsafe: float,
    V_shift: float,
    rng_key,
):
    """
    Roll out the nominal (non-robust) PNCBF safety filter under
    additive measurement noise. Collect per-step data for computing
    ideal residuals.
    """
    Vh_apply = ft.partial(pncbf.get_Vh, params=pncbf.Vh.params)
 
    def step_fn(carry, key):
        x_true = carry
 
        # Simulate noise
        delta = jax.random.uniform(
            key, shape=x_true.shape, minval=-epsilon, maxval=epsilon
        )
        x_hat = x_true + delta
 
        # Evaluate CBF at both states
        h_V_hat = Vh_apply(x_hat) + V_shift
        h_V_true = Vh_apply(x_true) + V_shift
 
        hx_Vx_hat = jax.jacobian(Vh_apply)(x_hat)
        hx_Vx_true = jax.jacobian(Vh_apply)(x_true)
 
        f_hat = pncbf.task.f(x_hat)
        G_hat = pncbf.task.G(x_hat)
        f_true = pncbf.task.f(x_true)
        G_true = pncbf.task.G(x_true)
 
        # Lie derivatives
        h_Lfh_hat = jnp.sum(hx_Vx_hat * f_hat, axis=-1)
        h_Lfh_true = jnp.sum(hx_Vx_true * f_true, axis=-1)
        h_LG_hat = hx_Vx_hat @ G_hat     # (nh, nu)
        h_LG_true = hx_Vx_true @ G_true
        h_Lgh_norm = jnp.linalg.norm(h_LG_hat, axis=-1)
 
        # Get control from nominal (non-robust) filter
        u_safe = pncbf.get_cbf_control_sloped(
            alpha_safe, alpha_unsafe, x_hat, V_shift
        )
 
        # Per-constraint Lgh @ u at both states
        h_Lgh_u_hat = jnp.sum(h_LG_hat * u_safe, axis=-1)
        h_Lgh_u_true = jnp.sum(h_LG_true * u_safe, axis=-1)
 
        # Step true dynamics
        xdot = f_true + G_true @ u_safe
        x_true_next = x_true + pncbf.task.dt * xdot
 
        data = FailureData(
            x_true=x_true, x_hat=x_hat, epsilon=epsilon,
            h_V_true=h_V_true, h_V_hat=h_V_hat,
            h_Lgh_norm=h_Lgh_norm,
            h_Lfh_true=h_Lfh_true, h_Lfh_hat=h_Lfh_hat,
            h_Lgh_u_true=h_Lgh_u_true, h_Lgh_u_hat=h_Lgh_u_hat,
            u_applied=u_safe,
        )
 
        return x_true_next, data
 
    keys = jr.split(rng_key, episode_length)
    _, trajectory = jax.lax.scan(step_fn, x0_true, keys)
    return trajectory  # Each field has leading dim (episode_length,)


def compute_ideal_residual(
    traj: FailureData,
    alpha_safe: float,
    alpha_unsafe: float,
    gamma1: float,
    gamma2: float,
):
    """
    Compute per-constraint ideal residual Δ_target = R_ideal - R_analytical.
 
    R_ideal_i ≈ |ḣ_i(x̂, u) - ḣ_i(x_true, u)|
              + |α_i * h_i(x̂) - α_i * h_i(x_true)|
 
    This measures the total per-constraint discrepancy from evaluating
    at x̂ instead of x_true.
    """
    # Per-constraint full derivative: ḣ_i = Lfh_i + Lgh_i @ u
    h_hdot_hat = traj.h_Lfh_hat + traj.h_Lgh_u_hat     # (T, nh)
    h_hdot_true = traj.h_Lfh_true + traj.h_Lgh_u_true   # (T, nh)
 
    # Alpha terms
    alpha = jnp.where(
        jnp.all(traj.h_V_hat < 0, axis=-1, keepdims=True),
        alpha_safe, alpha_unsafe
    )
    h_alpha_hat = alpha * traj.h_V_hat
    h_alpha_true = alpha * traj.h_V_true
 
    # Ideal correction per constraint
    h_R_ideal = (
        jnp.abs(h_hdot_hat - h_hdot_true)
        + jnp.abs(h_alpha_hat - h_alpha_true)
    )  # (T, nh)
 
    # Analytical baseline per constraint
    h_rho_anal = (
        gamma1 * traj.h_Lgh_norm
        + gamma2**2 * traj.h_Lgh_norm**2
    )  # (T, nh)
 
    # Residual: what the analytical term gets wrong
    h_Delta_target = h_R_ideal - h_rho_anal  # (T, nh) — can be negative!
 
    return h_Delta_target