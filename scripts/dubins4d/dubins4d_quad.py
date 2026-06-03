"""Shared primitives for the 4D dynamic-unicycle (quadruped) obstacle-avoidance task.

Used by:
  - scripts/dubins4d/dubins4d_cbf_compare.ipynb (rollout/comparison notebook)

State (x, y, theta, v) with controls (a, omega).
Holds analytical dynamics (f, G), HOCBF (h_raw, B_hocbf), the goal-go nominal
policy, the PhiNet definition, and a loader for trained phi weights.
"""
from pathlib import Path
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp


# ─── Constants ────────────────────────────────────────────────────────────────

AMAX = 1.0
OMEGA_MAX = 1.0
R_OBS = 0.25
DT = 0.05
T = 400
ALPHA_HOCBF = 2.0
ALPHA_QP = 1.0
V_MAX_NOM = 1.0  # nominal-policy speed cap

u_lb = jnp.array([-AMAX, -OMEGA_MAX])
u_ub = jnp.array([AMAX, OMEGA_MAX])

# Default deployment-time uncertainty (used by the notebook).
# Per-dim (x, y, theta, v) — position dominant, smaller heading/speed noise.
STATE_EPS = jnp.array([0.1, 0.1, 0.05, 0.05]) * 1

# Per-dim max uncertainty radius the PhiNet was trained over.
EPS_MAX = jnp.array([0.7, 0.7, 0.3, 0.3])

# Goal in (x, y). theta and v don't matter for reach test (we use x>=0).
GOAL_MC = jnp.array([2.0, 0.0])

# Absolute path so notebook and scripts both resolve to the same location.
NMR_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "runs"
    / "nmr_cbf_dubins4d"
    / "policies"
    / "obs0.25"
)


# ─── Dynamics + HOCBF ─────────────────────────────────────────────────────────

def f(x):
    px, py, theta, v = x
    return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0.0, 0.0])


def G(x):
    # Columns: a -> v_dot (row 3), omega -> theta_dot (row 2).
    return jnp.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])


def h_raw(x):
    px, py, theta, v = x
    return jnp.array([R_OBS - jnp.sqrt(px ** 2 + py ** 2)])  # (1,)


def B_hocbf(x, alpha=ALPHA_HOCBF):
    h = h_raw(x)
    J_h = jax.jacobian(h_raw)(x)
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
    """PD steering toward (goal_x, goal_y) at speed ~min(1, dist).

    Anti-circle: v_des is gated by max(cos(err), 0) so the robot brakes
    whenever the heading is off — it turns in place instead of orbiting.
    Steering is frozen inside a small goal disk to avoid chasing atan2
    noise.
    """
    px, py, theta, v = x
    dx, dy = goal[0] - px, goal[1] - py
    dist = jnp.sqrt(dx ** 2 + dy ** 2)

    theta_des = jnp.arctan2(dy, dx)
    err = (theta_des - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # v_des = V_MAX_NOM * cos(err)+ * min(dist, 1)
    cos_err_pos = jnp.clip(jnp.cos(err), 0.0, 1.0)
    v_des = V_MAX_NOM * cos_err_pos * jnp.minimum(1.0, dist)
    a = jnp.clip(2.0 * (v_des - v), -AMAX, AMAX)

    # Freeze omega once we're inside the goal disk.
    steer_gate = jnp.where(dist > 0.1, 1.0, 0.0)
    omega = jnp.clip(2.0 * err * steer_gate, -OMEGA_MAX, OMEGA_MAX)

    return jnp.array([a, omega])


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
