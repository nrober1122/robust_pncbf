import functools as ft

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from jaxtyping import Float

from pncbf.dyn.heron import Leader, Follower
from pncbf.dyn.dyn_types import Control, Disturb, HFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.none import get_or
from pncbf.utils.jax_types import Arr, TFloat

from typing import List


class Error(Task):
    """
    Full state vector layout (NX = 26):

        [0:11] — error states  (the only thing the model/policy ever sees)
                  [ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot,
                   e_ui, e_ri, thetarel, uF, rC]
        [11:13] — leader internal dynamics   (2)
                  [surge, yaw_rate]
        [13:16] — leader position            (3)
                  [px, py, heading]
        [16:19] — follower position          (3)
                  [px, py, heading]
    """

    # ------------------------------------------------------------------
    # Dimension constants
    # ------------------------------------------------------------------
    NX_ERR: int = 11   # error states
    NX_VEH: int = 8   # vehicle states  (2 + 3 + 3)
    NX: int = NX_ERR + NX_VEH 

    NU: int = 2

    # Slices into the full 26-dim state vector
    _SL_ERR = slice(0,  11)
    _SL_LEADER_X = slice(11, 13)
    _SL_LEADER_P = slice(13, 16)
    _SL_follower_P = slice(16, 19)

    # Error-state indices (within the first 12 elements)
    EX, EXDOT, EXDDOT, EY, EYDOT, EYDDOT, EUI, ERI, THETAREL, UF, RC, UL, RL, XL, YL, THETAL, XF, YF, THETAF = range(NX)
    VELREF, YAWREF = range(NU)

    DT = 0.01

    def __init__(self) -> None:
        self._dt = Error.DT

        # guidance law parameters
        self._u_min_guidance = jnp.array([0.0, -0.6])
        self._u_max_guidance = jnp.array([2.0, 0.6])
        self._lookahead: float = 10.0
        self._speed_control_gain: float = 30.0
        self._heading_rate_gain: float = 0.2
        self._mode = "circle"

        # constant geometric parameters
        self._mx: float = 10.0
        self._my: float = 0.0
        self._ml: float = 5.0

        # dynamics coefficients
        self._AuF21: float = -1.0
        self._AuF22: float = -24.048562
        self._ArF21: float = -6.3246
        self._ArF22: float = -61.78262

        # safety parameters
        self._a: float = 10.0
        self._b: float = 5.0
        self._s: float = 1 / ((self._a * self._b) ** 2)
        self._u_min_filter = jnp.array([0.0, -0.6])
        self._u_max_filter = jnp.array([2.0, 0.6])

        # Heron vehicle objects — used only for their dynamics matrices 
        self.leader: Leader = Leader()
        self.follower: Follower = Follower()

        # helper values
        self.DEG2RAD = jnp.pi / 180.0
        self.RAD2DEG = 180.0 / jnp.pi

    # ------------------------------------------------------------------
    # State vector helpers
    # ------------------------------------------------------------------
    def _unpack(self, state: State):
        """Split full state into its sub-arrays."""
        err_state = state[self._SL_ERR]
        leader_x  = state[self._SL_LEADER_X]
        leader_p  = state[self._SL_LEADER_P]
        follower_p  = state[self._SL_follower_P]
        return err_state, leader_x, leader_p, follower_p

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------
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
            r"$e_x$", r"$\dot{e}_x$", r"$\ddot{e}_x$",
            r"$e_y$", r"$\dot{e}_y$", r"$\ddot{e}_y$",
            r"$e_{u_I}$", r"$e_{r_I}$",
            r"$\theta_{\text{rel}}$",
            r"$u^F$", r"$r^C$",
            # leader internal
            r"$u^L$", r"$r^L$",
            # leader position
            r"$x^L$", r"$y^L$", r"$\psi^L$",
            # follower position
            r"$x^F$", r"$y^F$", r"$\psi^F$",
        ]

    @property
    def u_labels(self) -> List[str]:
        return [r"$u_{\text{cmd}}^F$", r"$r_{\text{cmd}}^C$"]

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
        del_x = follower_p[Follower.X] - leader_p[Leader.X]
        del_y = follower_p[Follower.Y] - leader_p[Leader.Y]

        heading_rad = leader_p[Leader.THETA]
        fx_leader = del_x * jnp.sin(heading_rad) - del_y * jnp.cos(heading_rad)
        fy_leader = del_x * jnp.cos(heading_rad) + del_y * jnp.sin(heading_rad)
        return fx_leader, fy_leader

    # -------------------------------------------------------------------------------------
    # Error-state drift  f_err(err, leader_x, leader_p, follower_p)
    # -------------------------------------------------------------------------------------
    def _f_err(self, err_state: State, leader_x: State, leader_p: State, follower_p: State) -> State:
        """
        Drift vector for the error states.
        """
        leader_u = leader_x[Leader.SURGE]
        leader_r = leader_x[Leader.YAWRATE]  

        ex = err_state[Error.EX]
        ey = err_state[Error.EY]
        e_ui = err_state[Error.EUI]
        e_ri = err_state[Error.ERI]
        uF = err_state[Error.UF]
        rC = err_state[Error.RC] 

        thetarel = err_state[Error.THETAREL]
        thetarel_dot = rC - leader_r

        d = jnp.sqrt(ex**2 + ey**2 + 1e-6)
        gamma = jnp.arctan2(ex, ey)

        u_dotF = self._AuF21 * e_ui + self._AuF22 * uF
        r_dotC = self._ArF21 * e_ri + self._ArF22 * rC

        s_tr = jnp.sin(thetarel)
        c_tr = jnp.cos(thetarel)
        s_g = jnp.sin(gamma)
        c_g = jnp.cos(gamma)

        ex_dot = uF * c_tr - self._ml * rC * s_tr + leader_r * (self._my + d * s_g)
        ey_dot = uF * s_tr + self._ml * rC * c_tr - leader_r * (self._mx + d * c_g) - leader_u

        d_dot = (ex * ex_dot + ey * ey_dot) / d
        gamma_dot = (-ey * ex_dot + ex * ey_dot) / (d ** 2)

        ex_ddot = (u_dotF * c_tr - uF * s_tr * thetarel_dot
                   - self._ml * (r_dotC * s_tr + rC * c_tr * thetarel_dot)
                   + leader_r * (d_dot * s_g + d * c_g * gamma_dot))
        ey_ddot = (u_dotF * s_tr + uF * c_tr * thetarel_dot
                   + self._ml * (r_dotC * c_tr - rC * s_tr * thetarel_dot)
                   - leader_r * (d_dot * c_g - d * s_g * gamma_dot))

        d_ddot = (((ex_dot**2 + ex*ex_ddot + ey_dot**2 + ey*ey_ddot) * d**2)
                  - (ex*ex_dot + ey*ey_dot)**2) / (d**3)
        gamma_ddot = (((-ey*ex_ddot + ex*ey_ddot) * d**2)
                      - 2 * (-ey*ex_dot + ex*ey_dot) * (ex*ex_dot + ey*ey_dot)) / (d**4)

        ex_dddot = ((self._AuF21*uF + self._AuF22*u_dotF) * c_tr
                    - 2*u_dotF*s_tr*thetarel_dot
                    - uF*(c_tr*thetarel_dot**2 + s_tr*r_dotC)
                    - self._ml * ((self._ArF21*rC + self._ArF22*r_dotC)*s_tr
                                  + 2*r_dotC*c_tr*thetarel_dot
                                  + rC*(c_tr*r_dotC - s_tr*thetarel_dot**2))
                    + leader_r * (d_ddot*s_g + 2*d_dot*c_g*gamma_dot
                                  + d*(c_g*gamma_ddot - s_g*gamma_dot**2)))
        ey_dddot = ((self._AuF21*uF + self._AuF22*u_dotF) * s_tr
                    + 2*u_dotF*c_tr*thetarel_dot
                    + uF*(c_tr*r_dotC - s_tr*thetarel_dot**2)
                    + self._ml * ((self._ArF21*rC + self._ArF22*r_dotC)*c_tr
                                  - 2*r_dotC*s_tr*thetarel_dot
                                  - rC*(c_tr*thetarel_dot**2 + s_tr*r_dotC))
                    - leader_r * (d_ddot*c_g - 2*d_dot*s_g*gamma_dot
                                  - d*(c_g*gamma_dot**2 + s_g*gamma_ddot)))

        return jnp.array([
            ex_dot,
            ex_ddot,
            ex_dddot,
            ey_dot,
            ey_ddot,
            ey_dddot,
            uF, # \dot{e}_ui = uF - u^F_cmd. 
            rC, # \dot{e}_ri = rC - r^C_cmd.
            thetarel_dot,
            u_dotF,
            r_dotC,
        ])

    def _G_err(self, err_state: State) -> State:
        """
        Input matrix for the 11 error states.
        Shape: (NX_ERR, NU) = (11, 2).
        """
        thetarel = err_state[Error.THETAREL]

        g = jnp.zeros((self.NX_ERR, self.NU))
        g = g.at[2, 0].set(-self._AuF21 * jnp.cos(thetarel))
        g = g.at[2, 1].set( self._ml * self._ArF21 * jnp.sin(thetarel))
        g = g.at[5, 0].set(-self._AuF21 * jnp.sin(thetarel))
        g = g.at[5, 1].set(-self._ml * self._ArF21 * jnp.cos(thetarel))
        g = g.at[6, 0].set(-1.0)
        g = g.at[7, 1].set(-1.0)
        return g
    
    def f(self, state: State) -> State:
        err_state, leader_x, leader_p, follower_p = self._unpack(state)

        leader_surge = leader_x[Leader.SURGE]
        leader_yawrate = leader_x[Leader.YAWRATE]
        leader_theta = leader_p[Leader.THETA]

        follower_surge = err_state[Error.UF]
        follower_yawrate = err_state[Error.RC]
        follower_theta = follower_p[Follower.THETA]

        leader_control = Leader.get_leader_control(self._mode)

        f_err = self._f_err(err_state, leader_x, leader_p, follower_p)
        f_leader_x = Leader.A @ leader_x + Leader.B @ leader_control # contains Bu term
        f_leader_p = jnp.array([leader_surge * jnp.cos(leader_theta),
                                leader_surge * jnp.sin(leader_theta),
                                leader_yawrate])
        f_follower_p = jnp.array([follower_surge * jnp.cos(follower_theta),
                                  follower_surge * jnp.sin(follower_theta),
                                  follower_yawrate])
        f = jnp.concatenate([f_err, f_leader_x, f_leader_p, f_follower_p])
        return f
    
    def G(self, state: State) -> State:
        err_state, leader_x, leader_p, follower_p = self._unpack(state)

        g_err = self._G_err(err_state)
        g_leader_x = jnp.zeros((Leader.NX, 2), dtype=jnp.float32) # absorbed into f function
        g_leader_p = jnp.zeros((Leader.NP, 2), dtype=jnp.float32) # absorbed into f function
        g_follower_p = jnp.zeros((Follower.NP, 2), dtype=jnp.float32) # absorbed into f function
        g = jnp.vstack([g_err, g_leader_x, g_leader_p, g_follower_p])
        
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
        control = control.clip(self._u_min_guidance, self._u_max_guidance)
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
        ex = state[Error.EX]
        ey = state[Error.EY]

        h_obs = -self._s * ((self._a * self._b)**2
                            - (self._b * ex)**2
                            - (self._a * ey)**2)

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
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 
                         0, 0, np.pi/2, 
                         self._mx, 0, np.pi/2],
                        dtype=np.float32)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        """Bounds over the full state vector."""
        err_bounds = np.array([
            (-15,  15),          # ex
            ( -2,   2),          # ex_dot
            (-140, 140),         # ex_ddot
            (-15,  15),          # ey
            ( -4,   4),          # ey_dot
            (-60,  60),          # ey_ddot
            (-60,  60),          # eui
            (-40,  40),          # eri
            (-np.pi, np.pi),     # thetarel
            (  0,   4),          # uF
            ( -1,   1),          # rC
        ], dtype=np.float32)

        veh_bounds = np.array([
            (  0,  4),   # leader surge
            ( -1,  1),   # leader yaw_rate
            (-100, 100), # leader px
            (-100, 100), # leader py
            (-np.pi, np.pi),  # leader heading
            (-100, 100), # follower px
            (-100, 100), # follower py
            (-np.pi, np.pi),  # follower heading
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
        _, leader_x, leader_p, follower_p = self._unpack(state)
        # jax.debug.print("leader_x: {}", leader_x)
        # jax.debug.print("leader_p: {}", leader_p)
        # jax.debug.print("follower_p: {}", follower_p)

        leader_theta = leader_p[Leader.THETA]
        leader_surge   = leader_x[Leader.SURGE]
        follower_theta = follower_p[Follower.THETA]

        fx_leader, fy_leader = self._global_to_leader(leader_p, follower_p)

        cross_track_error = fx_leader - self._mx
        along_track_error = fy_leader - self._my

        min_speed = self._u_min_guidance[0]
        max_speed = self._u_max_guidance[0]
        guidance_speed = (leader_surge
                          - max_speed * (along_track_error
                                         / jnp.sqrt(along_track_error**2
                                                    + self._speed_control_gain**2)))
        desired_speed = jnp.clip(guidance_speed, min_speed, max_speed)

        chiR = jnp.arctan2(-cross_track_error, self._lookahead)
        desired_theta = leader_theta - chiR # negative since we are using theta instead of heading

        # now convert to a delta theta
        delta_theta = follower_theta - desired_theta
        guidance_yaw_rate = -delta_theta * self._heading_rate_gain # does not include filter from pTrajectTranslate. Also negative for correct sign
        
        min_yaw_rate = self._u_min_guidance[1]
        max_yaw_rate = self._u_max_guidance[1]
        desired_yaw_rate = jnp.clip(guidance_yaw_rate, min_yaw_rate, max_yaw_rate)
        # jax.debug.print("Desired speed: {}", desired_speed)
        # jax.debug.print("Desired yaw rate: {}", desired_yaw_rate)

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