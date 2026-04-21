import functools as ft

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from jaxtyping import Float

from pncbf.dyn.heron import HeronSimplified
from pncbf.dyn.dyn_types import Control, Disturb, HFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.none import get_or
from pncbf.utils.jax_types import Arr, TFloat

from typing import List


class DHeronSimplified(Task):
    # ------------------------------------------------------------------
    # Dimension constants
    # ------------------------------------------------------------------
    NERR: int = 3
    NVEH: int = 6
    NX: int = NERR + NVEH
    NU: int = 2

    # Slices into the full 26-dim state vector
    _SL_ERR = slice(0,  3)
    _SL_LEADER_P = slice(3, 6)
    _SL_FOLLOWER_P = slice(6, 9)

    # Error-state
    EX, EY, THETAREL = range(NERR)
    VELREF, YAWREF = range(NU)

    DT = 0.01

    def __init__(self) -> None:
        self._dt = DHeronSimplified.DT

        # guidance law parameters
        self._u_min = jnp.array([0.0, -0.6])
        self._u_max = jnp.array([2.0, 0.6])
        self._lookahead: float = 10.0
        self._speed_control_gain: float = 30.0
        self._heading_rate_gain: float = 0.2
        self._mode = "circle"

        # constant geometric parameters
        self._mx: float = 10.0
        self._my: float = 0.0
        self._ml: float = 5.0

        # safety parameters
        self._a: float = 10.0
        self._b: float = 5.0
        self._s: float = 1 / ((self._a * self._b) ** 2)

        # Heron vehicle objects — used only for their dynamics matrices 
        self.leader: HeronSimplified = HeronSimplified()
        self.follower: HeronSimplified = HeronSimplified()

        # helper values
        self.DEG2RAD = jnp.pi / 180.0
        self.RAD2DEG = 180.0 / jnp.pi

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------
    @property
    def u_min(self):
        return self._u_min

    @property
    def u_max(self):
        return self._u_max
    
    @property
    def n_Vobs(self) -> int:
        return self.NX

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def nx(self) -> int:
        return self.NX 

    @property
    def x_labels(self) -> List[str]:
        # Labels for the full state vector (used for debug / logging)
        return [
            r"$e_x$", r"$e_y$", r"$\theta_{\text{rel}}$",
            r"$x^L$", r"$y^L$", r"$\psi^L$",
            r"$x^F$", r"$y^F$", r"$\psi^F$",
        ]

    @property
    def u_labels(self) -> List[str]:
        return [r"$u^F$", r"$r^C$"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$obs$"]

    @property
    def h_max(self) -> float:
        return 10.25

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def max_ttc(self) -> float:
        return 5.0
    
    # ------------------------------------------------------------------
    # State vector helpers
    # ------------------------------------------------------------------
    def _unpack(self, state: State):
        """Split full state into its sub-arrays."""
        err_state = state[self._SL_ERR]
        leader_p  = state[self._SL_LEADER_P]
        follower_p  = state[self._SL_FOLLOWER_P]
        return err_state, leader_p, follower_p

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        """
        Returns only the 12 error states.
        The model / policy never sees leader or follower vehicle states.
        """
        obs = state[self._SL_ERR]   # shape (12,)
        return state, state
        # return obs, obs

    # ------------------------------------------------------------------
    # Dynamics helper
    # ------------------------------------------------------------------
    def _global_to_leader(self, leader_p: State, follower_p: State) -> tuple:
        """Purely functional frame transform."""
        del_x = follower_p[HeronSimplified.X] - leader_p[HeronSimplified.X]
        del_y = follower_p[HeronSimplified.Y] - leader_p[HeronSimplified.Y]

        heading_rad = leader_p[HeronSimplified.THETA]
        fx_leader = del_x * jnp.sin(heading_rad) - del_y * jnp.cos(heading_rad)
        fy_leader = del_x * jnp.cos(heading_rad) + del_y * jnp.sin(heading_rad)
        return fx_leader, fy_leader

    # -------------------------------------------------------------------------------------
    # Error-state drift  f_err(err, leader_x, leader_p, follower_p)
    # -------------------------------------------------------------------------------------
    def _f_err(self, err_state: State, leader_p: State, follower_p: State, leader_control: Control) -> State:
        """
        Drift vector for the error states.
        """
        ex = err_state[DHeronSimplified.EX]
        ey = err_state[DHeronSimplified.EY]
        thetarel = err_state[DHeronSimplified.THETAREL]
        x_leader = leader_p[HeronSimplified.X]
        y_leader = leader_p[HeronSimplified.Y]
        theta_leader = leader_p[HeronSimplified.THETA]
        x_follower = follower_p[HeronSimplified.X]
        y_follower = follower_p[HeronSimplified.Y]
        theta_follower = follower_p[HeronSimplified.THETA]

        v_leader = leader_control[HeronSimplified.VELOCITY]
        r_leader = leader_control[HeronSimplified.YAWRATE]

        d = jnp.sqrt(ex**2 + ey**2)
        gamma = jnp.arctan2(ey, ex)

        ex_dot_partial = r_leader * (self._my + d * jnp.sin(gamma))
        ey_dot_partial = -r_leader * (self._mx + d * jnp.cos(gamma)) - v_leader
        thetarel_dot_partial = -r_leader

        return jnp.array([
            ex_dot_partial, #v^F * c(tr) - ml * r^F * s(tr) + r^L * (my + d * s(g))
            ey_dot_partial, #v^F * s(tr) + ml * r^F * c(tr) - r^L * (mx + d * c(g)) - v^L
            thetarel_dot_partial #r^F - r^L
        ])

    def _G_err(self, err_state: State, leader_p: State, follower_p: State, leader_control: Control) -> State:
        """
        Input matrix for the 3 error states.
        Shape: (NERR, NU) = (3, 2).
        """
        ex = err_state[DHeronSimplified.EX]
        ey = err_state[DHeronSimplified.EY]
        thetarel = err_state[DHeronSimplified.THETAREL]
        x_leader = leader_p[HeronSimplified.X]
        y_leader = leader_p[HeronSimplified.Y]
        theta_leader = leader_p[HeronSimplified.THETA]
        x_follower = follower_p[HeronSimplified.X]
        y_follower = follower_p[HeronSimplified.Y]
        theta_follower = follower_p[HeronSimplified.THETA]

        g = jnp.zeros((self.NERR, self.NU))
        g = g.at[0, 0].set(jnp.cos(thetarel))
        g = g.at[0, 1].set(-self._ml * jnp.sin(thetarel))
        g = g.at[1, 0].set(jnp.sin(thetarel))
        g = g.at[1, 1].set(self._ml * jnp.cos(thetarel))
        g = g.at[2, 1].set(1.0)
        return g
    
    def f(self, state: State) -> State:
        err_state, leader_p, follower_p = self._unpack(state)
        theta_leader = leader_p[HeronSimplified.THETA]

        leader_control = HeronSimplified.get_leader_control(self._mode)

        f_err = self._f_err(err_state, leader_p, follower_p, leader_control)
        f_leader_p = jnp.array([[jnp.cos(theta_leader), 0.0],
                                [jnp.sin(theta_leader), 0.0],
                                [0.0, 1.0]]) @ leader_control
        f_follower_p = jnp.array([0.0, 0.0, 0.0])
        f = jnp.concatenate([f_err, f_leader_p, f_follower_p])
        return f
    
    def G(self, state: State) -> State:
        err_state, leader_p, follower_p = self._unpack(state)
        theta_follower = follower_p[HeronSimplified.THETA]

        leader_control = HeronSimplified.get_leader_control(self._mode)

        g_err = self._G_err(err_state, leader_p, follower_p, leader_control)
        g_leader_p = jnp.zeros((3, 2), dtype=jnp.float32)
        g_follower_p = jnp.array([[jnp.cos(theta_follower), 0.0],
                                  [jnp.sin(theta_follower), 0.0],
                                  [0.0, 1.0]])
        g = jnp.vstack([g_err, g_leader_p, g_follower_p])
        
        return g
    
    # ------------------------------------------------------------------
    # xdot — the function diffeqsolve calls; integrates full state
    # ------------------------------------------------------------------
    def xdot(self, state: State, control: Control) -> State:
        """
        Full dimensional time derivative.
        Purely functional — safe inside jit / vmap / diffeqsolve.

        Layout of returned vector mirrors the state vector layout described
        in the class docstring.
        """
        self.chk_x(state)
        self.chk_u(control)
        control = control.clip(self._u_min, self._u_max)
        f, G = self.f(state), self.G(state)
        self.chk_x(f)
        Gu = G @ control
        self.chk_x(Gu)
        dx = f + Gu
        return self.chk_x(dx)

    # ------------------------------------------------------------------
    # Safety constraint h 
    # ------------------------------------------------------------------
    def h_components(self, state: State) -> HFloat:
        ex = state[DHeronSimplified.EX]
        ey = state[DHeronSimplified.EY]

        h_obs = -self._s * ((self._a * self._b)**2 - (self._b * ex)**2 - (self._a * ey)**2)

        hs = poly4_clip_max_flat(jnp.array([h_obs]), max_val=self.h_max)
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    # ------------------------------------------------------------------
    # step / step_plot  
    # ------------------------------------------------------------------
    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_fn = ft.partial(self.xdot, control=control)
        return rk4(self._dt, xdot_fn, state)

    def step_plot(
        self, state: State, control: Control,
        disturb: Disturb = None, dt: float = None,
    ) -> tuple[TState, TFloat]:
        xdot_fn = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self._dt)
        return tsit5(dt, 4, xdot_fn, state), np.linspace(0, dt, num=5)

    # ------------------------------------------------------------------
    # States / bounds
    # ------------------------------------------------------------------
    def has_eq_state(self) -> bool:
        return False

    def eq_state(self) -> State:
        return np.zeros(self.NX, dtype=np.float32)

    def nominal_val_state(self) -> State:
        """Returns the full state."""
        return np.array([0, 0, np.pi/2, # ex, ey, thetarel
                         0, 0, np.pi/2, # xL, yL, thetaL
                         self._mx, -5, np.pi/2], #xF, yF, thetaF
                         dtype=np.float32)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        """Bounds over the full state vector."""
        err_bounds = np.array([
            (-15,  15),          # ex
            (-15,  15),          # ey
            (-2*np.pi, 2*np.pi),     # thetarel
        ], dtype=np.float32)

        veh_bounds = np.array([
            (-100, 100), # leader px
            (-100, 100), # leader py
            (-2*np.pi, 2*np.pi),  # leader heading
            (-100, 100), # follower px
            (-100, 100), # follower py
            (-2*np.pi, 2*np.pi),  # follower heading
        ], dtype=np.float32)

        bounds = np.concatenate([err_bounds, veh_bounds], axis=0)
        return bounds.T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    # ------------------------------------------------------------------
    # Nominal policy
    # ------------------------------------------------------------------
    def nom_pol_goto(self, state: State, goal: State = None) -> Control:
        """
        Guidance law. Reads leader/follower state from the full state vector —
        never from self.leader or self.follower, so safe inside a JAX trace.
        """
        _, leader_p, follower_p = self._unpack(state)

        leader_theta = leader_p[HeronSimplified.THETA]
        follower_theta = follower_p[HeronSimplified.THETA]

        fx_leader, fy_leader = self._global_to_leader(leader_p, follower_p)
        leader_control = HeronSimplified.get_leader_control(self._mode)
        v_leader = leader_control[HeronSimplified.VELOCITY]

        cross_track_error = fx_leader - self._mx
        along_track_error = fy_leader - self._my

        min_speed = self._u_min[0]
        max_speed = self._u_max[0]
        guidance_speed = (v_leader
                          - max_speed * (along_track_error
                                         / jnp.sqrt(along_track_error**2
                                                    + self._speed_control_gain**2)))
        desired_speed = jnp.clip(guidance_speed, min_speed, max_speed)

        chiR = jnp.arctan2(-cross_track_error, self._lookahead)
        desired_theta = leader_theta - chiR # negative since we are using theta instead of heading

        # now convert to a delta theta
        delta_theta = follower_theta - desired_theta
        guidance_yaw_rate = -delta_theta * self._heading_rate_gain # does not include filter from pTrajectTranslate. Also negative for correct sign
        
        min_yaw_rate = self._u_min[1]
        max_yaw_rate = self._u_max[1]
        desired_yaw_rate = jnp.clip(guidance_yaw_rate, min_yaw_rate, max_yaw_rate)

        return jnp.array([desired_speed, desired_yaw_rate], dtype=jnp.float32)
    
    def nom_pol_straight(self, state: State, goal: State = None) -> Control:
        """
        Guidance law. Reads leader/follower state from the full state vector —
        never from self.leader or self.follower, so safe inside a JAX trace.
        """
        desired_speed = 1.0
        desired_yaw_rate = 0.0

        return jnp.array([desired_speed, desired_yaw_rate], dtype=jnp.float32)

    def nom_pol_zero(self, state: State, goal: State = None) -> Control:
        return jnp.array([0.0, 0.0], dtype=jnp.float32)

    def has_episode_pol(self) -> bool:
        return False

    def make_episode_pol(self, key, nom_pol):
        return ft.partial(nom_pol)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase,
                                  Task.mk_get2d([self.EX, self.EY]))]

    def plot_phase(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -15, 15
        PLOT_YMIN, PLOT_YMAX = -15, 15
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[self.EX], ylabel=self.x_labels[self.EY])
        ax.set_aspect("equal")

    def plot_traj(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -2, 12
        PLOT_YMIN, PLOT_YMAX = -2, 12
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel="X", ylabel="Y")
        ax.set_aspect("equal")