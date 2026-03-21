import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import shapely
from jaxproxqp.jaxproxqp import JaxProxQP
from jaxtyping import Float

from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.networks.fourier_emb import PosEmbed
from pncbf.networks.mlp import mlp_partial
from pncbf.networks.pol_det import PolDet
from pncbf.networks.train_state import TrainState
from pncbf.plotting.phase2d_utils import plot_x_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.plotting.poly_to_patch import poly_to_patch
from pncbf.qp.min_norm_cbf import min_norm_cbf
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.jax_types import Arr, BFloat, BoolScalar, FloatScalar, TFloat
from pncbf.utils.jax_utils import jax_vmap, smoothmax
from pncbf.utils.none import get_or
from pncbf.utils.rng import PRNGKey
from pncbf.utils.sampling_utils import get_mesh_np


class SingleIntWall(Task):
    # NX=2 so the base class 2D contour/mesh/phase machinery works unchanged.
    # The second state (index 1) is a dummy, always pinned to 0 — it has no
    # dynamics and is ignored everywhere except for shape compatibility.
    NX = 2
    NU = 1
    ND = 1

    P, _V_DUMMY = range(NX)
    (A,) = range(NU)

    DT = 0.1

    def __init__(self):
        self.umax = 1.0
        self._dt = SingleIntWall.DT
        self.pos_wall = 1.0

        self.nscbf = False

    @property
    def nd(self) -> int:
        return self.ND

    @property
    def n_Vobs(self) -> int:
        return 1

    @property
    def x_labels(self) -> list[str]:
        # Dummy state labelled so plots are readable but clearly not real.
        return [r"$p$", r"$\tilde{v}$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$a$"]

    @property
    def l_labels(self) -> list[str]:
        return ["dist"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$p_{l}$", r"$p_{u}$"]

    @property
    def l_scale(self) -> LFloat:
        return np.array([40.0])

    def at_goal(self, state: State) -> BoolScalar:
        p, _ = self.chk_x(state)
        return jnp.abs(p) < 1e-3

    def l_cts(self, state: State) -> LFloat:
        p, _ = self.chk_x(state)
        return jnp.abs(p)

    def l_components(self, state: State) -> LFloat:
        p, _ = self.chk_x(state)
        dist_cost = 0.5 * p ** 2
        at_goal = self.at_goal(state)
        cost = jnp.array([dist_cost]) / self.l_scale
        cost = jnp.where(at_goal, 0.0, 0.05) + 0.5 * cost
        return cost

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def max_ttc(self) -> float:
        return 5.0

    def h_components(self, state: State) -> HFloat:
        self.chk_x(state)
        p, _ = state
        h_p_ub = p - self.pos_wall
        h_p_lb = -(p + self.pos_wall)

        hs = poly4_clip_max_flat(jnp.array([h_p_lb, h_p_ub]))
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    def nscbf_sample_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.0, 2.0), (0.0, 0.0)]).T

    def get_contour_x0_nscbf(self):
        n_pts = 80
        with jax.ensure_compile_time_eval():
            bounds = np.array([(-2.0, 2.0), (0.0, 0.0)]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def nscbf_rho(self, state: State) -> FloatScalar:
        h_h = self.h_components(state)
        return smoothmax(h_h, t=0.01)

    def nscbf_phis(self, state: State) -> BFloat:
        rho, rho_x = jax.value_and_grad(self.nscbf_rho)(state)
        c1 = 1.0
        phi1 = rho + c1 * jnp.dot(rho_x, self.f(state))
        b_phis = jnp.stack([phi1])
        assert b_phis.shape == (1,)
        return b_phis

    def is_stable(self, T_state: TState) -> BoolScalar:
        return jnp.array(True)

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        p, _ = self.chk_x(state)
        obs = jnp.array([p])
        return obs, obs

    @property
    def dt(self):
        return self._dt

    def f(self, state: State) -> State:
        self.chk_x(state)
        # Single integrator: no drift. Dummy state has no dynamics either.
        return jnp.zeros(self.NX)

    def G(self, state: State):
        self.chk_x(state)
        # pdot = u * umax; dummy state is unaffected by control.
        G = np.array([[1.0],
                      [0.0]])
        return G * self.umax

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        next_state = rk4(self.dt, xdot_with_u, state)
        # Pin dummy state exactly to 0 to prevent numerical drift.
        return next_state.at[self._V_DUMMY].set(0.0)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        T_state, T_t = tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)
        # Pin dummy state.
        T_state = T_state.at[:, self._V_DUMMY].set(0.0)
        return T_state, T_t

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        # Dummy state is always 0, so give it a zero-width range.
        return np.array([(-2.5, 2.5), (0.0, 0.0)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.5, 2.5), (0.0, 0.0)]).T

    def get_paper_ci_x0(self, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            bounds = np.array([(-1.1, 1.1), (0.0, 0.0)]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def get_paper_pi_x0(self, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            bounds = np.array([(-1.25, 1.25), (0.0, 0.0)]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def get_paper_plot_x0(self):
        return np.array(
            [
                [-0.95, 0.0],
                [-0.05, 0.0],
                [-0.4,  0.0],
                [0.6,   0.0],
                [-0.6,  0.0],
                [0.9,   0.0],
                [0.6,   0.0],
            ]
        )

    def plot_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-1.5, 1.5), (0.0, 0.0)]).T

    def get_plot_x0(self, setup_idx: int = 0) -> BState:
        with jax.ensure_compile_time_eval():
            n_pts, idxs = 12, (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.plot_bounds(), idxs, n_pts, n_pts, self.nominal_val_state())
            b_x0 = ei.rearrange(bb_x0, "nys nxs nx -> (nys nxs) nx")

            rng = np.random.default_rng(seed=123124)
            b_pos_noise = 0.05 * rng.standard_normal((b_x0.shape[0], 1))
            b_x0[:, 0:1] += b_pos_noise
            # Dummy state stays 0.

            b_in_ci = jax_vmap(self.in_ci_approx)(b_x0)
            b_x0 = b_x0[b_in_ci]

        return b_x0

    def get_plot_rng_x0(self) -> BState:
        return np.array([[-1.0, 0.0]])

    def in_ci_approx(self, state: State) -> BoolScalar:
        p, _ = self.chk_x(state)
        return jnp.abs(p) < self.pos_wall

    def nominal_val_state(self) -> State:
        return np.array([-1.0, 0.0])

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(self.NX)

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        # Use (P, _V_DUMMY) so the base class 2D mesh/contour machinery works unchanged.
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.P, self._V_DUMMY]))]

    def nom_pol_osc(self, state: State):
        self.chk_x(state)
        ref = 0.95
        # ref = 1.05
        error = ref - state
        K = np.array([[1.0, 0.0]])
        return jnp.clip(K @ error, -1.0, 1.0)

    def nom_pol_rng(self, state: State, key: PRNGKey = jr.PRNGKey(58123)):
        self.chk_x(state)
        state_fake = np.ones(self.nx)
        mlp = mlp_partial([16, 16, 16])
        ff = ft.partial(PosEmbed, mlp, embed_dim=8, scale=1.2)
        pol_def = PolDet(ff, self.nu)
        random_pol = TrainState.create_from_def(key, pol_def, (state_fake,), tx=None)
        return random_pol.apply(state)

    def nom_pol_rng2(self, state: State):
        return self.nom_pol_rng(state, jr.PRNGKey(511247))

    def nom_pol_rng3(self, state: State):
        coef = 1e0
        std = np.array([2.0, 1.0])
        normsq = jnp.sum((state - np.array([0.8, 0.0])) ** 2 / (std ** 2))
        weight = jnp.exp(-normsq)
        nominal_u = self.nom_pol_osc(state)
        return self.nom_pol_rng(state, jr.PRNGKey(511249)) + coef * weight * nominal_u

    def handcbf_B(self, state: State, alpha: float):
        p, _ = self.chk_x(state)
        h_p_ub = p - self.pos_wall
        h_p_lb = -(p + self.pos_wall)
        h_B = jnp.stack([h_p_ub, h_p_lb], axis=0)
        return h_B.max()

    def handcbf_pol(self, state: State, alpha: float, cbf_alpha: float = 5.0):
        h_B = self.handcbf_B(state, alpha)
        hx_Bx = jax.jacfwd(ft.partial(self.handcbf_B, alpha=alpha))(state)

        u_nom = self.nom_pol_rng2(state)
        u_lb, u_ub = self.u_min, self.u_max
        f, G = self.f(state), self.G(state)

        settings = JaxProxQP.Settings.default()
        u_qp, r, sol = min_norm_cbf(cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom, settings=settings)
        u_qp = self.chk_u(u_qp.clip(self.u_min, self.u_max))
        return u_qp

    def get_ci_points(self):
        # Return wall positions with dummy v=0, shaped like the double-integrator
        # version so any caller iterating ci_pts[:, 0] / ci_pts[:, 1] still works.
        ps = np.array([-self.pos_wall, self.pos_wall])
        vs = np.zeros_like(ps)
        return np.stack([ps, vs], axis=1)

    def plot_phase(self, ax: plt.Axes):
        """Render as a 1D number line; dummy axis is collapsed."""
        PLOT_XMIN, PLOT_XMAX = -1.5, 1.5
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(-0.1, 0.1))
        ax.set_xlabel(self.x_labels[0])
        ax.set_yticks([])
        ax.axhline(0, color="k", lw=0.5)

        for wall_x in [-self.pos_wall, self.pos_wall]:
            ax.axvline(wall_x, **PlotStyle.ci_line)

        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), PlotStyle.obs_region)

    def plot_phase_paper(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -1.25, 1.25
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(-0.1, 0.1))
        ax.set_xlabel(self.x_labels[0])
        ax.set_yticks([])

        for wall_x in [-self.pos_wall, self.pos_wall]:
            ax.axvline(wall_x, color="0.4", lw=1.0)

        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)
        obs_style = dict(facecolor="none", lw=1.0, edgecolor="0.4", alpha=0.8, zorder=3.4, hatch="/")
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)