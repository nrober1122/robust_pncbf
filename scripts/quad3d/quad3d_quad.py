"""Shared primitives for the 3D quadrotor obstacle-avoidance task.

Used by:
  - scripts/quad3d/quad3d_cbf_compare.ipynb  (rollout/comparison notebook)
  - scripts/quad3d/finetune_phi_quad3d.py    (RL fine-tune of phi, if added)

Mirrors the layout of scripts/dbint/dbint2d_quad.py: dynamics (f, G), HOCBF
(h_raw, B_hocbf), the goal-go nominal policy, the PhiNet definition, and a
loader for trained phi weights.

Convention
----------
State (12):  [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
Control (4): [uT, u_tau_x, u_tau_y, u_tau_z], each in [-1, 1]

Safety convention matches dbint2d:
  h_raw  : (nh,) raw constraint values, h > 0 = UNSAFE (task.h_components)
  B_hocbf: (nh,) HOCBF safety barrier,  h < 0 = SAFE (task.handcbf_B)

There is one sphere obstacle by default at (1, 0, 1.4) with radius 0.4 m,
plus the 6 base constraints (floor/ceil/roll±/pitch±).
"""
from pathlib import Path
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from mrncbf.dyn.quad3d import Quad3D


# ─── Constants ────────────────────────────────────────────────────────────────

UMAX = 1.0            # control bound (each channel of u in [-1, 1])

DT = 0.01             # control timestep (100 Hz) — matches Quad3D.DT
T = 1000              # 10 s rollouts
ALPHA_HOCBF_BASE = 2.0   # HOCBF construction alpha for attitude constraints
ALPHA_HOCBF_OBS  = 5.0   # HOCBF construction alpha for obstacle
ALPHA_HOCBF      = ALPHA_HOCBF_OBS  # backward compat scalar

ALPHA_QP_BASE = 3.0   # QP class-K gain for base constraints (floor/ceil/attitude)
ALPHA_QP_OBS  = 3.0   # QP class-K gain for obstacle (smaller = more conservative)
ALPHA_QP      = ALPHA_QP_BASE  # scalar alias (backward compat; prefer ALPHA_QP_VEC)

# Aggressive Values
# ALPHA_HOCBF_BASE = 5.0
# ALPHA_HOCBF_OBS  = 5.0
# ALPHA_QP_BASE    = 3.0
# ALPHA_QP_OBS     = 3.0

# Default obstacle: single sphere matching the old quad3d_cbf_compare notebook.
OBSTACLE = (0.0, 0.0, 1.5, 0.25)   # (cx, cy, cz, radius)

task = Quad3D(obstacles=[OBSTACLE])
NX, NU, NH = task.nx, task.nu, task.nh
Z_HOVER = task.Z_HOVER

_N_BASE = NH - len(task.obstacles)
ALPHA_QP_VEC = jnp.concatenate([
    jnp.full(_N_BASE, ALPHA_QP_BASE),
    jnp.full(len(task.obstacles), ALPHA_QP_OBS),
])

u_lb = jnp.array(task.u_min, dtype=jnp.float32)
u_ub = jnp.array(task.u_max, dtype=jnp.float32)

# Default deployment-time uncertainty (used by the notebook). Position-dominant.
# Order: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
STATE_EPS = jnp.array([
    0.10, 0.10, 0.05,   # position (m)
    0.05, 0.05, 0.05,   # velocity (m/s)
    0.02, 0.02, 0.02,   # angles   (rad)
    0.02, 0.02, 0.02,   # ang rates(rad/s)
], dtype=jnp.float32)*3
# STATE_EPS = jnp.array([
#     0.10, 0.10, 0.05,   # position (m)
#     0.05, 0.05, 0.05,   # velocity (m/s)
#     0.01, 0.01, 0.01,   # angles   (rad)
#     0.01, 0.01, 0.01,   # ang rates(rad/s)
# ], dtype=jnp.float32)*5

# Per-dim max uncertainty radius the PhiNet is trained over.
EPS_MAX = jnp.array([
    0.10, 0.10, 0.05,   # position (m)
    0.05, 0.05, 0.05,   # velocity (m/s)
    0.02, 0.02, 0.02,   # angles   (rad)
    0.02, 0.02, 0.02,   # ang rates(rad/s)
], dtype=jnp.float32)*7

# Goal: hover past the obstacle along +x.  States not listed are zero (hover).
GOAL_MC = jnp.zeros(NX, dtype=jnp.float32)
GOAL_MC = GOAL_MC.at[task.PX].set(2.0)
GOAL_MC = GOAL_MC.at[task.PZ].set(Z_HOVER)

# Absolute path so notebook and scripts both resolve to the same location.
NMR_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "runs"
    / "nmr_cbf_quad3d"
    / "policies"
    / f"obs{OBSTACLE[0]:.2f}_{OBSTACLE[1]:.2f}_{OBSTACLE[2]:.2f}_r{OBSTACLE[3]:.2f}"
)


# ─── Dynamics + HOCBF (thin wrappers) ────────────────────────────────────────

def f(x):
    return task.f(x)


def G(x):
    return task.G(x)


def h_raw(x):
    """Raw safety values (h > 0 = unsafe, h < 0 = safe), UN-CLIPPED.

    Same definitions as task.h_components but without the poly4 saturation
    to [-1, 1], matching the dbint2d convention.  Returns (nh,) for the 6
    base constraints + 1 obstacle.
    """
    pz, phi, theta = x[task.PZ], x[task.PHI], x[task.THETA]
    h_base = jnp.array([
        -(pz    - task.Z_MIN),
        -(task.Z_MAX  - pz),
        -(phi   + task.PHI_MAX),
        -(task.PHI_MAX   - phi),
        -(theta + task.THETA_MAX),
        -(task.THETA_MAX - theta),
    ])
    cx, cy, cz, r_obs = OBSTACLE
    d2 = (x[task.PX] - cx) ** 2 + (x[task.PY] - cy) ** 2 + (pz - cz) ** 2
    h_obs = 1.0 - d2 / (r_obs ** 2)
    return jnp.concatenate([h_base, jnp.array([h_obs])])


def B_hocbf(x, alpha_obs=ALPHA_HOCBF_OBS, alpha_att=ALPHA_HOCBF_BASE):
    """HOCBF safety barrier (h < 0 = safe).  Returns (nh,)."""
    return task.handcbf_B(x, alpha_obs=alpha_obs, alpha_att=alpha_att)


def cbf_ingredients(x, alpha_obs=ALPHA_HOCBF_OBS, alpha_att=ALPHA_HOCBF_BASE):
    """Return (h_B, Lf_B, LG_B) at state x.  Shapes (nh,), (nh,), (nh, nu)."""
    h_B = B_hocbf(x, alpha_obs=alpha_obs, alpha_att=alpha_att)
    J_B = jax.jacobian(lambda xi: B_hocbf(xi, alpha_obs=alpha_obs, alpha_att=alpha_att))(x)
    return h_B, J_B @ f(x), J_B @ G(x)


_cbf_ingredients = cbf_ingredients   # underscore alias used by the notebook


# ─── Nominal policy ───────────────────────────────────────────────────────────

# PD gains (chosen for stability at DT = 0.05 s — see task.nom_pol_hover).
# _KP_Z, _KD_Z     = 4.0, 3.0
# _KP_XY, _KD_XY   = 0.4, 1.2
# _KP_ATT, _KD_ATT = 4.0, 0.3
# _KP_PSI, _KD_PSI = 1.0, 0.2
_KP_Z, _KD_Z     = 4.0, 3.0
_KP_XY, _KD_XY   = 1.5, 3.0
# _KP_XY, _KD_XY   = 0.2, 1.2
_KP_ATT, _KD_ATT = 4.0, 0.3
_KP_PSI, _KD_PSI = 1.0, 0.2


def nom_pol_goto(x, goal=GOAL_MC):
    """Cascade PD goto-goal: hovers at (goal[PX], goal[PY], goal[PZ]).

    Drop-in equivalent of nom_pol_hover but tracks an arbitrary (x, y, z) goal.
    Tracking yaw_des = 0.  Velocity references read off the goal too so a moving
    goal could also be tracked.
    """
    px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
    gx, gy, gz = goal[task.PX], goal[task.PY], goal[task.PZ]
    gvx, gvy, gvz = goal[task.VX], goal[task.VY], goal[task.VZ]

    u_T = jnp.clip(-_KP_Z * (pz - gz) - _KD_Z * (vz - gvz), -UMAX, UMAX)

    theta_des = jnp.clip(-(_KP_XY * (px - gx) + _KD_XY * (vx - gvx)) / task.GRAV, -0.3, 0.3)
    phi_des   = jnp.clip( (_KP_XY * (py - gy) + _KD_XY * (vy - gvy)) / task.GRAV, -0.3, 0.3)

    u_tau_x = jnp.clip(_KP_ATT * (phi_des   - phi)   - _KD_ATT * p, -UMAX, UMAX)
    u_tau_y = jnp.clip(_KP_ATT * (theta_des - theta) - _KD_ATT * q, -UMAX, UMAX)
    u_tau_z = jnp.clip(-_KP_PSI * psi - _KD_PSI * r,                -UMAX, UMAX)

    return jnp.array([u_T, u_tau_x, u_tau_y, u_tau_z])


# ─── PhiNet ───────────────────────────────────────────────────────────────────

class PhiNet(nn.Module):
    """Learned uncertainty margin.

    Input : [xhat (nx,) || eps (nx,)] → (2*nx,)
    Output: softplus → (nh,), nonnegative per-constraint correction.
    """
    hidden_dims: tuple = (64, 64)
    n_constraints: int = NH

    @nn.compact
    def __call__(self, xhat, eps):
        x = jnp.concatenate([xhat, eps])
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.softplus(nn.Dense(self.n_constraints)(x))


phi_net = PhiNet()


def load_phi_params(path: Path):
    with open(path, "rb") as f_:
        return pickle.load(f_)


def init_phi_params(seed: int = 0):
    import jax.random as jr
    return phi_net.init(jr.PRNGKey(seed), jnp.zeros(NX), jnp.zeros(NX))["params"]
