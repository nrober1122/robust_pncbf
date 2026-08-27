"""Shared primitives for the 2D double-integrator (quadruped) obstacle-avoidance task.

Used by:
  - scripts/dbint/dbint2d_cbf_compare.ipynb (rollout/comparison notebook)
  - scripts/dbint/finetune_phi_dbint2d.py    (RL fine-tune of phi)

Holds analytical dynamics (f, G), HOCBF (h_raw, B_hocbf), the goal-go nominal
policy, the PhiNet definition, and a loader for trained phi weights.
"""
from pathlib import Path
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp


# ─── Constants ────────────────────────────────────────────────────────────────

UMAX = 1.0
R_OBS = 0.25
DT = 0.05
T = 400
ALPHA_HOCBF = 2.0
ALPHA_QP = 1.0
H_SHIFT = 0.0   # safety margin folded into HOCBF to cover Euler integration leak

u_lb = jnp.array([-UMAX, -UMAX])
u_ub = jnp.array([UMAX, UMAX])

# Default deployment-time uncertainty (used by the notebook). Fine-tuning
# samples per-iter eps in [0, EPS_MAX] instead.
STATE_EPS = jnp.array([0.1, 0.05, 0.1, 0.05]) * 3

# Per-dim max uncertainty radius the PhiNet was trained over.
EPS_MAX = jnp.array([0.7, 0.3, 0.7, 0.3])

GOAL_MC = jnp.array([2.0, 0.0, 0.0, 0.0])

# Absolute path so notebook and scripts both resolve to the same location.
NMR_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "runs"
    / "nmr_cbf_dbint2d"
    / "policies"
    # / "quadruped"
    / "obs0.25"
)

# QUADRUPED_PARAMS
# UMAX = 2.0
# R_OBS = 0.43
# ALPHA_HOCBF = 3.0
# ALPHA_QP = 2.0
# EPS_MAX = jnp.array([0.5, 0.25, 0.5, 0.25])

# ─── Dynamics + HOCBF ─────────────────────────────────────────────────────────

def f(x):
    px, vx, py, vy = x
    return jnp.array([vx, 0.0, vy, 0.0])


def G(x):
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]) # * UMAX # COMMENTING THIS OUT 


def h_raw(x):
    px, vx, py, vy = x
    return jnp.array([R_OBS - jnp.sqrt(px ** 2 + py ** 2)])  # (1,)


def B_hocbf(x, alpha=ALPHA_HOCBF):
    h = h_raw(x) + H_SHIFT
    J_h = jax.jacobian(h_raw)(x)   # derivative of constant shift is 0
    return J_h @ f(x) + alpha * h  # (1,)


def cbf_ingredients(x, alpha=ALPHA_HOCBF):
    """Return (h_B, Lf_B, LG_B) — shapes (1,), (1,), (1, 2)."""
    h_B = B_hocbf(x, alpha)
    J_B = jax.jacobian(lambda xi: B_hocbf(xi, alpha))(x)
    return h_B, J_B @ f(x), J_B @ G(x)


# Backwards-compat alias for the notebook's underscore name.
_cbf_ingredients = cbf_ingredients


# ─── Nominal policy ───────────────────────────────────────────────────────────

def nom_pol_goto(x, goal=GOAL_MC):
    px, vx, py, vy = x
    p = jnp.array([px, py])
    v = jnp.array([vx, vy])
    g = goal[:2]
    u_raw = -2.0 * (p - g) - 2.0 * v
    norm_inf = jnp.max(jnp.abs(u_raw))
    return jnp.where(norm_inf > UMAX, u_raw / norm_inf * UMAX, u_raw)


# ─── PhiNet ───────────────────────────────────────────────────────────────────

class PhiNet(nn.Module):
    hidden_dims: tuple = (64, 64)

    @nn.compact
    def __call__(self, xhat, eps):
        x = jnp.concatenate([xhat, eps])  # (8,) — state estimate + uncertainty radius
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.softplus(nn.Dense(1)(x))  # (1,), enforces phi >= 0


phi_net = PhiNet()


def load_phi_params(path: Path):
    """Load trained phi_params from a pickle file."""
    with open(path, "rb") as f_:
        return pickle.load(f_)


def init_phi_params(seed: int = 0):
    """Random init matching the notebook's PhiNet shape."""
    import jax.random as jr
    return phi_net.init(jr.PRNGKey(seed), jnp.zeros(4), jnp.zeros(4))["params"]
