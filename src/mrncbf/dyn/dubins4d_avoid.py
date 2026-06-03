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


class Dubins4dAvoid(Task):
    """4D dynamic unicycle (quadruped) with circular obstacle avoidance.

    State: (x, y, theta, v)         — position, heading, forward speed.
    Control: (a, omega)             — forward acceleration, yaw rate.
    Safety: h(x) = radius - ||p|| < 0 means safe (outside obstacle).

    Relative degree of h with respect to control is 2: L_G h = 0, but
    L_G L_f h = [(radial dir on a),  v * (tangential dir on omega)],
    so the HOCBF psi_1 = L_f h + alpha h has both a and omega in psi_1_dot
    once v > 0. The QP can then steer the robot around the obstacle.
    """

    NX = 4
    NU = 2

    X, Y, THETA, V = range(NX)
    A, OMEGA = range(NU)

    DT = 0.05

    def __init__(self):
        self.amax = 1.0
        self.omega_max = 1.0
        self._dt = Dubins4dAvoid.DT

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
        return [r"$x$", r"$y$", r"$\theta$", r"$v$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$a$", r"$\omega$"]

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
        x, y, theta, v = self.chk_x(state)
        return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0.0, 0.0])

    def G(self, state: State):
        self.chk_x(state)
        # Columns: a -> v_dot (row V), omega -> theta_dot (row THETA).
        G = jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        return G

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
        x, y, theta, v = self.chk_x(state)
        dist = jnp.sqrt((x - self.pos_obs[0]) ** 2 + (y - self.pos_obs[1]) ** 2)
        # h < 0  ->  safe (outside obstacle);  h > 0  ->  unsafe (inside).
        h_obs = self.radius_obs - dist
        return jnp.array([h_obs])

    def handcbf_B(self, state: State, alpha: float = 2.0) -> HFloat:
        """HOCBF psi_1 = L_f h + alpha * h (relative degree 2 in (a, omega))."""
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
        return np.array([(-3.0, 3.0), (-3.0, 3.0), (-np.pi, np.pi), (0.0, 1.5)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    # ------------------------------------------------------------------
    # Nominal policy: PD controller steering toward a goal
    # ------------------------------------------------------------------

    def nom_pol_goto(
        self, state: State, goal: jnp.ndarray = jnp.array([2.0, 0.0])
    ) -> Control:
        x, y, theta, v = self.chk_x(state)
        dx, dy = goal[0] - x, goal[1] - y
        dist = jnp.sqrt(dx ** 2 + dy ** 2)

        theta_des = jnp.arctan2(dy, dx)
        err = (theta_des - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        omega = jnp.clip(2.0 * err, -self.omega_max, self.omega_max)

        # Slow as we approach the goal, cap forward speed at 1 m/s.
        v_des = jnp.minimum(1.0, 1.0 * dist)
        a = jnp.clip(2.0 * (v_des - v), -self.amax, self.amax)
        return jnp.array([a, omega])

    def make_episode_pol(self, key, nom_pol):
        angle = jr.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        goal = 3.0 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        return ft.partial(nom_pol, goal=goal)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("xy", self.plot_phase, Task.mk_get2d([self.X, self.Y]))]

    def plot_phase(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -3.0, 3.0
        PLOT_YMIN, PLOT_YMAX = -3.0, 3.0
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[self.X], ylabel=self.x_labels[self.Y])
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
