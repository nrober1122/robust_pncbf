import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.jax_types import Arr, BoolScalar, TFloat
from pncbf.utils.none import get_or
from pncbf.utils.sampling_utils import get_mesh_np


class Dubins3DAvoid(Task):
    NX = 3
    NU = 1

    X, Y, THETA = range(NX)
    (OMEGA,) = range(NU)

    DT = 0.1
    VEL = 1.0

    def __init__(self):
        self.umax = np.pi / 4
        self._dt = Dubins3DAvoid.DT
        self._vel = Dubins3DAvoid.VEL

        self.pos_obs = jnp.array([0.0, 0.0])
        self.radius_obs = 0.5
        self.has_episode_pol_val = False

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------

    @property
    def n_Vobs(self) -> int:
        return 3

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_labels(self) -> list[str]:
        return [r"$x$", r"$y$", r"$\theta$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$\omega$"]

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
        x, y, theta = self.chk_x(state)
        return jnp.array([self._vel * jnp.cos(theta), self._vel * jnp.sin(theta), 0.0])

    def G(self, state: State):
        self.chk_x(state)
        GT = np.array([[0.0, 0.0, 1.0]])
        G = GT.T
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
        x, y, theta = self.chk_x(state)
        # Negative means unsafe (inside or on the obstacle).
        # h = dist_to_center - radius; h < 0 => inside obstacle.
        h_obs = -(
            jnp.sqrt((x - self.pos_obs[0]) ** 2 + (y - self.pos_obs[1]) ** 2)
            - self.radius_obs
        )
        # h <= 1
        hs = poly4_clip_max_flat(jnp.array([h_obs]))
        # clip h >= h_min
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        x, y, theta = self.chk_x(state)
        obs = jnp.array([x, y, theta])
        return obs, obs

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(3)

    def nominal_val_state(self) -> State:
        # Start to the left of the obstacle, heading right.
        return np.array([-1.5, 0.0, 0*np.pi/4])

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.5, 2.5), (-2.5, 2.5), (-np.pi, np.pi)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    # ------------------------------------------------------------------
    # Nominal policy: steer toward a goal
    # ------------------------------------------------------------------

    def nom_pol_goto(
        self, state: State, goal: jnp.ndarray = jnp.array([2.0, 0.0])
    ) -> Control:
        x, y, theta = self.chk_x(state)
        dx, dy = goal[0] - x, goal[1] - y
        theta_des = jnp.arctan2(dy, dx)
        err = theta_des - theta
        # Wrap to [-pi, pi].
        err = (err + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.array([jnp.clip(2.0 * err, -1.0, 1.0)])
    
    def nom_pol_zero(self, state: State, goal=None) -> Control:
        return jnp.array([0.0])

    def nom_pol_avoid(self, state: State, goal=None) -> Control:
        x, y, theta = self.chk_x(state)
        
        # Vector from obstacle to agent
        dx = x - self.pos_obs[0]
        dy = y - self.pos_obs[1]
        dist = jnp.sqrt(dx**2 + dy**2)
        
        # Desired heading: directly away from obstacle
        theta_away = jnp.arctan2(dy, dx)
        
        # Heading error to "away" direction
        err = theta_away - theta
        err = (err + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # Scale by proximity — only turn when close
        influence_radius = 1.5
        weight = jnp.clip(1.0 - dist / influence_radius, 0.0, 1.0)
        
        omega = jnp.clip(2.0 * weight * err, -1.0, 1.0)
        return jnp.array([omega])

    def has_episode_pol(self) -> bool:
        return self.has_episode_pol

    def make_episode_pol(self, key, nom_pol):
        angle = jr.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        goal = 3.0 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        return ft.partial(nom_pol, goal=goal)
    # def make_episode_pol(self, key, nom_pol):
    #     omega = jr.uniform(key, minval=-1.0, maxval=1.0, shape=(1,))
    #     return lambda state: omega

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.X, self.Y]))]

    def plot_phase(self, ax: plt.Axes):
        """XY plane plot with circular obstacle."""
        PLOT_XMIN, PLOT_XMAX = -2.5, 2.5
        PLOT_YMIN, PLOT_YMAX = -2.5, 2.5
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[0], ylabel=self.x_labels[1])
        ax.set_aspect("equal")

        # Draw the circular obstacle.
        circle = plt.Circle(
            (float(self.pos_obs[0]), float(self.pos_obs[1])),
            self.radius_obs,
            facecolor="0.45",
            edgecolor="0.4",
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(circle)