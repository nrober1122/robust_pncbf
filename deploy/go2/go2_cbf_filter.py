"""Standalone CBF safety filter for 2D double-integrator / Go2 obstacle avoidance.

State layout : x = [px, vx, py, vy]
Control      : u = [ax, ay]  (world-frame accelerations, bounded to ±UMAX)
Convention   : h(x) > 0 is SAFE (outside obstacle), h(x) < 0 is unsafe.

No dependency on the mrncbf package — only jax, flax, jaxproxqp, numpy, pickle.
"""

import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP

# ── Tuneable constants ─────────────────────────────────────────────────────────

UMAX        = 1.0
R_OBS       = 0.25   # obstacle radius (metres)
DT          = 0.05   # control timestep (s) — 20 Hz
ALPHA_HOCBF = 2.0    # class-K gain for the HOCBF intermediate function
ALPHA_QP    = 1.0    # class-K gain used inside the CBF-QP

# Per-dim uncertainty radius [px, vx, py, vy] used at deployment.
DEFAULT_STATE_EPS = jnp.array([0.3, 0.15, 0.3, 0.15])

_u_lb = jnp.array([-UMAX, -UMAX])
_u_ub = jnp.array([ UMAX,  UMAX])


# ── System dynamics ────────────────────────────────────────────────────────────

def f(x: jnp.ndarray) -> jnp.ndarray:
    """Drift vector field: ẋ = f(x) + G(x) u."""
    return jnp.array([x[1], 0.0, x[3], 0.0])


def G(x: jnp.ndarray) -> jnp.ndarray:
    """Input matrix (4×2)."""
    return jnp.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 1.0]]) * UMAX


# ── Safety constraint (h > 0 convention) ──────────────────────────────────────

def h_safe(x: jnp.ndarray) -> jnp.ndarray:
    """h(x) = dist_to_obs_centre - R_OBS.  h > 0 ⟺ outside obstacle (safe)."""
    dist = jnp.sqrt(x[0] ** 2 + x[2] ** 2)
    return jnp.array([dist - R_OBS])   # shape (1,)


def B_hocbf(x: jnp.ndarray, alpha: float = ALPHA_HOCBF) -> jnp.ndarray:
    """Higher-order CBF for the relative-degree-2 double integrator.

    B(x) = Lf h(x) + alpha * h(x)   (shape (1,))

    B > 0 in a 'safely evolving' region.  The CBF-QP constraint enforces
    Lf B + LG B @ u + alpha_qp * B >= 0  (h>0 convention, see min_norm_cbf_h_pos).
    """
    h = h_safe(x)
    J_h = jax.jacobian(h_safe)(x)      # (1, 4)
    return J_h @ f(x) + alpha * h      # (1,)


# ── Nominal policy ─────────────────────────────────────────────────────────────

def nom_pol_goto(x: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
    """PD go-to-goal policy; output clipped to control bounds."""
    p = jnp.array([x[0], x[2]])
    v = jnp.array([x[1], x[3]])
    u_raw = -2.0 * (p - goal[:2]) - 2.0 * v
    norm_inf = jnp.max(jnp.abs(u_raw))
    return jnp.where(norm_inf > UMAX, u_raw / norm_inf * UMAX, u_raw)


# ── Min-norm CBF-QP (h > 0 convention) ────────────────────────────────────────

def min_norm_cbf_h_pos(
    alpha,
    u_lb: jnp.ndarray,
    u_ub: jnp.ndarray,
    h_V: jnp.ndarray,       # (nh,)  CBF values in h>0 convention
    hx_Vx: jnp.ndarray,     # (nh, 4) Jacobian of B_hocbf w.r.t. x
    f_x: jnp.ndarray,       # (4,)   drift at current state
    G_x: jnp.ndarray,       # (4, 2) input matrix at current state
    u_nom: jnp.ndarray,     # (2,)   nominal control
    penalty: float = 10.0,
    relax_eps1: float = 0.5,
    relax_eps2: float = 0.1,
):
    """Min-norm CBF safety filter with h>0-safe convention.

    Solves:
        min   0.5 ||u - u_nom||^2 + 0.5 * penalty * (r + relax_eps2)^2
        s.t.  LG B @ u + Lf B + alpha * B + r >= 0    (CBF constraint)
              u_lb <= u <= u_ub
              r >= -relax_eps1

    Returns (u_opt, r, sol).
    """
    nu = u_lb.shape[0]
    if h_V.ndim == 0:
        h_V   = h_V[None]
        hx_Vx = hx_Vx[None]
    nh = h_V.shape[0]
    dtype = h_V.dtype

    h_Lf_V  = (hx_Vx * f_x).sum(axis=-1)   # (nh,)
    h_LG_V  = hx_Vx @ G_x                   # (nh, nu)
    h_alphah = alpha * h_V                   # (nh,)

    # Objective: 0.5 z^T H z + g^T z,  z = [u; r]
    H = np.eye(nu + 1, dtype=np.float32)
    H[-1, -1] = penalty

    g = jnp.concatenate([-u_nom,
                          jnp.array([penalty * relax_eps2], dtype=dtype)])

    # Constraint in ProxQP upper-bound form C @ z <= upper:
    #   [-LG B, -1] @ [u, r] <= Lf B + alpha * B
    # ⟺  LG B @ u + Lf B + alpha * B + r >= 0
    C     = jnp.concatenate([-h_LG_V,
                              -jnp.ones((nh, 1), dtype=dtype)], axis=-1)
    upper = h_Lf_V + h_alphah              # (nh,)

    l_box = jnp.concatenate([u_lb, jnp.array([-relax_eps1], dtype=dtype)])
    u_box = jnp.concatenate([u_ub, jnp.array([ 1e9        ], dtype=dtype)])

    qp  = JaxProxQP.QPModel.create(H, g, C, upper, l_box, u_box)
    sol = JaxProxQP(qp, JaxProxQP.Settings.default()).solve()

    u_opt = sol.x[:nu]
    r     = sol.x[-1]
    return u_opt, r, sol


# ── Projected-gradient oracle ──────────────────────────────────────────────────

def proj_grad_max_box(scalar_fn, lo, hi, n_restarts, n_steps, lr, key):
    """Maximise scalar_fn over the box [lo, hi] via projected gradient ascent.

    Runs n_restarts random initialisations in parallel (vmap), each for
    n_steps gradient steps with step size lr.  Returns (best_val, best_x).
    """
    x0s = lo[None] + jr.uniform(key, (n_restarts, lo.shape[0])) * (hi - lo)

    def run_one(x0):
        def step(x, _):
            g = jax.grad(scalar_fn)(x)
            return jnp.clip(x + lr * g, lo, hi), None
        x_star, _ = jax.lax.scan(step, x0, None, length=n_steps)
        return scalar_fn(x_star), x_star

    vals, xs = jax.vmap(run_one)(x0s)
    best = jnp.argmax(vals)
    return vals[best], xs[best]


# ── PhiNet (NMR-CBF margin network) ───────────────────────────────────────────

class PhiNet(nn.Module):
    """Learned uncertainty margin network.

    Input : [x_hat (4,) || eps (4,)] → (8,)
    Output: softplus scalar φ ≥ 0

    Trained to approximate the worst-case HOCBF drift excess over the
    uncertainty box centred at x_hat with per-dim radius eps.
    """
    hidden_dims: tuple = (64, 64)

    @nn.compact
    def __call__(self, xhat: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([xhat, eps])
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.softplus(nn.Dense(1)(x))   # (1,), φ >= 0


_phi_net = PhiNet()


# ── Filter classes ─────────────────────────────────────────────────────────────

class NmrCbfFilter:
    """NMR-CBF: learned-margin safety filter.

    Uses PhiNet to predict worst-case drift uncertainty φ(x̂, ε) ≥ 0, then
    tightens the HOCBF barrier by φ/α before solving the CBF-QP.

    Parameters
    ----------
    phi_weights_path : path-like
        Path to the trained ``phi_params.pkl`` file.
    state_eps : array-like, shape (4,)
        Per-dim uncertainty radius [px, vx, py, vy].
    alpha : float
        Class-K gain for the CBF-QP constraint.
    """

    def __init__(self, phi_weights_path, state_eps=None, alpha: float = ALPHA_QP):
        with open(phi_weights_path, "rb") as fh:
            self._phi_params = pickle.load(fh)
        self._state_eps = (
            jnp.array(DEFAULT_STATE_EPS) if state_eps is None
            else jnp.asarray(state_eps, dtype=jnp.float32)
        )
        self._alpha = alpha
        self._jit_step = jax.jit(self._step)

    def _step(self, xhat, u_nom):
        phi  = _phi_net.apply({"params": self._phi_params}, xhat, self._state_eps)  # (1,)
        h_B  = B_hocbf(xhat)                        # (1,)
        J_B  = jax.jacobian(B_hocbf)(xhat)          # (1, 4)
        # Reduce safety margin by learned uncertainty φ/α
        h_adj = h_B - phi / self._alpha
        u_opt, _, _ = min_norm_cbf_h_pos(
            self._alpha, _u_lb, _u_ub, h_adj, J_B, f(xhat), G(xhat), u_nom
        )
        return u_opt

    def warmup(self):
        """Trigger JAX JIT compilation with a dummy state."""
        dummy_x   = jnp.array([1.0, 0.0, 1.0, 0.0])
        dummy_u   = jnp.zeros(2)
        _ = self._jit_step(dummy_x, dummy_u)

    def compute_control(self, xhat: jnp.ndarray, u_nom: jnp.ndarray) -> np.ndarray:
        """Return CBF-filtered control for state estimate xhat.

        Parameters
        ----------
        xhat  : (4,) state estimate [px, vx, py, vy]
        u_nom : (2,) nominal (goal-going) control

        Returns
        -------
        u_cbf : (2,) safe control, world-frame acceleration in [-UMAX, UMAX]
        """
        return np.array(self._jit_step(xhat, u_nom))


class PshrCbfFilter:
    """PSHR-CBF: projected-gradient oracle safety filter.

    Computes the worst-case HOCBF drift over the uncertainty box at each step
    via projected gradient ascent (no trained network needed), then solves the
    CBF-QP with the resulting oracle margin.

    Warning: substantially slower than NmrCbfFilter (~50–200 ms per call before
    JIT, much faster after).  The JIT-compiled oracle runs n_restarts parallel
    trajectories of n_steps gradient steps per constraint dimension.

    Parameters
    ----------
    state_eps : array-like, shape (4,)
        Per-dim uncertainty radius [px, vx, py, vy].
    alpha : float
        Class-K gain for the CBF-QP constraint.
    n_restarts : int
        Number of random restarts for the projected gradient oracle.
    n_steps : int
        Gradient ascent steps per restart.
    lr : float
        Gradient ascent step size.
    """

    def __init__(
        self,
        state_eps=None,
        alpha: float = ALPHA_QP,
        n_restarts: int = 4,
        n_steps: int = 25,
        lr: float = 0.05,
    ):
        self._state_eps  = (
            jnp.array(DEFAULT_STATE_EPS) if state_eps is None
            else jnp.asarray(state_eps, dtype=jnp.float32)
        )
        self._alpha      = alpha
        self._n_restarts = n_restarts
        self._n_steps    = n_steps
        self._lr         = lr
        self._jit_step   = jax.jit(self._step)

    def _step(self, xhat, u_nom, key):
        lo = xhat - self._state_eps
        hi = xhat + self._state_eps

        h_nom = B_hocbf(xhat)              # (nh,)
        J_nom = jax.jacobian(B_hocbf)(xhat)  # (nh, 4)
        Lf_nom = J_nom @ f(xhat)           # (nh,)
        nh = h_nom.shape[0]

        def worst_i(i, k):
            # Find x in box that MINIMISES Lf B + alpha*B (most dangerous).
            # proj_grad_max_box maximises, so we maximise the negation.
            _, x_wc = proj_grad_max_box(
                lambda xi: -(
                    jax.jacobian(B_hocbf)(xi) @ f(xi) + self._alpha * B_hocbf(xi)
                )[i],
                lo, hi,
                self._n_restarts, self._n_steps, self._lr, k,
            )
            h_wc_i  = B_hocbf(x_wc)[i]
            Lf_wc_i = (jax.jacobian(B_hocbf)(x_wc) @ f(x_wc))[i]
            return h_wc_i, Lf_wc_i

        keys = jr.split(key, nh)
        h_wc, Lf_wc = jax.vmap(worst_i)(jnp.arange(nh), keys)

        # Margin = how much the nominal exceeds the worst-case (h>0 convention)
        phi_oracle = (Lf_nom + self._alpha * h_nom) - (Lf_wc + self._alpha * h_wc)
        phi_oracle = jnp.maximum(phi_oracle, 0.0)   # phi >= 0

        h_adj = h_nom - phi_oracle / self._alpha
        u_opt, _, _ = min_norm_cbf_h_pos(
            self._alpha, _u_lb, _u_ub, h_adj, J_nom, f(xhat), G(xhat), u_nom
        )
        return u_opt

    def warmup(self, seed: int = 0):
        """Trigger JAX JIT compilation with a dummy state."""
        dummy_x = jnp.array([1.0, 0.0, 1.0, 0.0])
        dummy_u = jnp.zeros(2)
        dummy_k = jr.PRNGKey(seed)
        _ = self._jit_step(dummy_x, dummy_u, dummy_k)

    def compute_control(
        self,
        xhat: jnp.ndarray,
        u_nom: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> np.ndarray:
        """Return CBF-filtered control for state estimate xhat.

        Parameters
        ----------
        xhat    : (4,) state estimate [px, vx, py, vy]
        u_nom   : (2,) nominal control
        rng_key : JAX PRNGKey for the oracle random restarts

        Returns
        -------
        u_cbf : (2,) safe control, world-frame acceleration in [-UMAX, UMAX]
        """
        return np.array(self._jit_step(xhat, u_nom, rng_key))
