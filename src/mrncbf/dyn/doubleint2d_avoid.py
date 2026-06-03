import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from mrncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, PolObs, State, TState, VObs
from mrncbf.dyn.odeint import rk4, tsit5
from mrncbf.dyn.task import Task
from mrncbf.utils.hocbf import hocbf
from mrncbf.utils.jax_types import Arr, BoolScalar, TFloat
from mrncbf.utils.none import get_or
from mrncbf.utils.sampling_utils import get_mesh_np


class DoubleInt2dAvoid(Task):
    """2D double integrator (quadruped) with circular obstacle avoidance.

    State: (px, vx, py, vy)  —  position and velocity in x and y.
    Control: (ux, uy)         —  acceleration inputs.
    Safety: h(x) = radius - ||p|| < 0 means safe (outside obstacle).

    Unlike Dubins3d the system can stop in front of the obstacle:
    at v=0, LGψ₁ = outward normal (never zero), so CBF is satisfied by u=0.
    """

    NX = 4
    NU = 2

    PX, VX, PY, VY = range(NX)
    UX, UY = range(NU)

    DT = 0.05

    def __init__(self):
        self.umax = 1.0
        self._dt = DoubleInt2dAvoid.DT

        self.pos_obs = jnp.array([0.0, 0.0])
        self.radius_obs = 0.5

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------

    @property
    def n_Vobs(self) -> int:
        return 4

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_labels(self) -> list[str]:
        return [r"$p_x$", r"$v_x$", r"$p_y$", r"$v_y$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$u_x$", r"$u_y$"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$obs$"]

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def max_ttc(self) -> float:
        return 5.0

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def f(self, state: State) -> State:
        px, vx, py, vy = self.chk_x(state)
        return jnp.array([vx, 0.0, vy, 0.0])

    def G(self, state: State):
        self.chk_x(state)
        GT = np.array([[0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
        G = GT.T  # (4, 2)
        return G * self.umax

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)

    # ------------------------------------------------------------------
    # Safety constraint h
    # ------------------------------------------------------------------

    def h_components(self, state: State) -> HFloat:
        px, vx, py, vy = self.chk_x(state)
        dist = jnp.sqrt((px - self.pos_obs[0]) ** 2 + (py - self.pos_obs[1]) ** 2)
        # h < 0  →  safe (outside obstacle);  h > 0  →  unsafe (inside)
        h_obs = self.radius_obs - dist
        return jnp.array([h_obs])

    def handcbf_B(self, state: State, alpha: float = 2.0) -> HFloat:
        """HOCBF ψ₁ = Lf h + α·h (relative degree 2).

        ψ₁ < 0  →  safe approach speed;  ψ₁ > 0  →  CBF filter must intervene.
        LGψ₁ = outward normal to obstacle — never zero outside obstacle center.
        """
        return hocbf(self.h_components, self.f, alpha0=alpha, state=state)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        obs = jnp.array(self.chk_x(state))
        return obs, obs

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(4)

    def nominal_val_state(self) -> State:
        return np.array([-2.0, 0.0, 0.0, 0.0])

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-3.0, 3.0), (-2.0, 2.0), (-3.0, 3.0), (-2.0, 2.0)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    # ------------------------------------------------------------------
    # Nominal policy: PD controller driving toward a goal
    # ------------------------------------------------------------------

    def nom_pol_goto(
        self, state: State, goal: jnp.ndarray = jnp.array([2.0, 0.0])
    ) -> Control:
        px, vx, py, vy = self.chk_x(state)
        p = jnp.array([px, py])
        v = jnp.array([vx, vy])
        u_raw    = -2.0 * (p - goal) - 1.5 * v
        norm_inf = jnp.max(jnp.abs(u_raw))
        u = jnp.where(norm_inf > self.umax, u_raw / norm_inf * self.umax, u_raw)
        return u

    def make_episode_pol(self, key, nom_pol):
        angle = jr.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        goal = 3.0 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        return ft.partial(nom_pol, goal=goal)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("xy", self.plot_phase, Task.mk_get2d([self.PX, self.PY]))]

    def plot_phase(self, ax: plt.Axes):
        """XY plane plot with circular obstacle."""
        PLOT_XMIN, PLOT_XMAX = -3.0, 3.0
        PLOT_YMIN, PLOT_YMAX = -3.0, 3.0
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[self.PX], ylabel=self.x_labels[self.PY])
        ax.set_aspect("equal")

        circle = plt.Circle(
            (float(self.pos_obs[0]), float(self.pos_obs[1])),
            self.radius_obs,
            facecolor="0.45",
            edgecolor="0.4",
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(circle)
