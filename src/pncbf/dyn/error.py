import functools as ft

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from jaxtyping import Float

from pncbf.dyn.heron import Heron
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

        [ 0:12] — error states  (the only thing the model/policy ever sees)
                  [ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot,
                   e_ui, e_ri, thetarel, thetarel_dot, uF, rC]
        [12:16] — leader internal dynamics  (Heron.NX = 4)
                  [surge_int_err, surge, yaw_rate_int_err, yaw_rate]
        [16:19] — leader position            (Heron.NP = 3)
                  [px, py, heading]
        [19:23] — follower internal dynamics (Heron.NX = 4)
        [23:26] — follower position          (Heron.NP = 3)
    """

    # ------------------------------------------------------------------
    # Dimension constants
    # ------------------------------------------------------------------
    NX_ERR:  int = 12   # error states — what the model sees
    NX_VEH:  int = 14   # vehicle states  (4 + 3 + 4 + 3)
    NX:      int = NX_ERR + NX_VEH   # 26  — full integrated state

    NU: int = 2

    # Slices into the full 26-dim state vector
    _SL_ERR       = slice(0,  12)
    _SL_LEADER_X  = slice(12, 16)
    _SL_LEADER_P  = slice(16, 19)
    _SL_follower_X  = slice(19, 23)
    _SL_follower_P  = slice(23, 26)

    # Error-state indices (within the first 12 elements)
    EX, EXDOT, EXDDOT, EY, EYDOT, EYDDOT, EUI, EUR, THETAREL, THETARELDOT, UF, RC = range(NX_ERR)
    VELREF, YAWREF = range(NU)

    DT = 0.05

    def __init__(self) -> None:
        self._dt = Error.DT

        # guidance law parameters
        self._u_min_guidance = jnp.array([0.0, -0.6])
        self._u_max_guidance = jnp.array([2.0, 0.6])
        self._lookahead: float = 10.0
        self._speed_control_gain: float = 30.0
        self._heading_rate_gain: float = 0.2
        self._mode = "straight"

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

        # Heron vehicle objects — used only for their dynamics matrices and
        # to supply initial conditions.  Vehicle state lives in the full
        # state vector after initialisation; these objects are never mutated.
        leader_start_position   = jnp.array([0.0,       0.0, 0.0], dtype=jnp.float32)
        follower_start_position = jnp.array([self._a,   0.0, 0.0], dtype=jnp.float32)
        self.leader:   Heron = Heron(leader_start_position)
        self.follower: Heron = Heron(follower_start_position)

        # helper values
        self.DEG2RAD = jnp.pi / 180.0
        self.RAD2DEG = 180.0 / jnp.pi

    # ------------------------------------------------------------------
    # State vector helpers
    # ------------------------------------------------------------------
    def _unpack(self, state: State):
        """Split full 26-dim state into its five named sub-arrays."""
        err_state = state[self._SL_ERR]
        leader_x  = state[self._SL_LEADER_X]
        leader_p  = state[self._SL_LEADER_P]
        follower_x  = state[self._SL_follower_X]
        follower_p  = state[self._SL_follower_P]
        return err_state, leader_x, leader_p, follower_x, follower_p

    def initial_state(self, err_state: State = None) -> State:
        """
        Build the initial full 26-dim state vector.
        err_state defaults to nominal_val_state() if not provided.
        """
        if err_state is None:
            err_state = self.nominal_val_state()
        return jnp.concatenate([
            jnp.asarray(err_state,                        dtype=jnp.float32),
            self.leader.initial_x,
            self.leader.initial_position,
            self.follower.initial_x,
            self.follower.initial_position,
        ])

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------
    @property
    def n_Vobs(self) -> int:
        return self.NX
        # return self.NX_ERR   # model sees only error states

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def nx(self) -> int:
        return self.NX        # full 26-dim state for the integrator

    @property
    def x_labels(self) -> List[str]:
        # Labels for the full state vector (used for debug / logging)
        return [
            r"$e_x$", r"$\dot{e}_x$", r"$\ddot{e}_x$",
            r"$e_y$", r"$\dot{e}_y$", r"$\ddot{e}_y$",
            r"$e_{u_I}^F$", r"$e_{r_I}^F$",
            r"$\theta_{\text{rel}}$", r"$\dot{\theta}_{\text{rel}}$",
            r"$u^F$", r"$r^C$",
            # leader internal
            r"$e^L_{u_I}$", r"$u^L$", r"$e^L_{r_I}$", r"$r^L$",
            # leader position
            r"$x^L$", r"$y^L$", r"$\psi^L$",
            # follower internal
            r"$e^F_{u_I}$", r"$u^F$", r"$e^F_{r_I}$", r"$r^F$",
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
    # Observations — this is the firewall between full state and model
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
        """Purely functional frame transform. Safe inside any JAX trace."""
        del_x = follower_p[self.follower.X] - leader_p[self.leader.X]
        del_y = follower_p[self.follower.Y] - leader_p[self.follower.Y]

        heading_rad = leader_p[2] * self.DEG2RAD
        fx_leader =  del_x * jnp.cos(heading_rad) - del_y * jnp.sin(heading_rad)
        fy_leader =  del_x * jnp.sin(heading_rad) + del_y * jnp.cos(heading_rad)
        return fx_leader, fy_leader

    # -------------------------------------------------------------------------------------
    # Error-state drift  f_err(err, leader_x, leader_p, follower_x, follower_p)
    # -------------------------------------------------------------------------------------
    def _f_err(self, err_state: State,
               leader_x: State, leader_p: State, follower_x: State,
               follower_p: State) -> State:
        """
        Drift vector for the 12 error states.
        All arguments are explicit — no reads from self.leader / self.follower.
        """
        leader_u = leader_x[Heron.SURGE]    # index 1
        leader_r = leader_x[Heron.YAWRATE]  # index 3

        leader_heading   = leader_p[self.leader.HEADING]
        follower_heading = follower_p[self.follower.HEADING]

        thetarel     = -(follower_heading - leader_heading)
        thetarel_dot = -(follower_x[Heron.YAWRATE] - leader_x[Heron.YAWRATE])

        fx_leader, fy_leader = self._global_to_leader(leader_p, follower_p)

        control_point_x = fx_leader + self._ml * jnp.cos(thetarel)
        control_point_y = fy_leader + self._ml * jnp.sin(thetarel)

        ex   = control_point_x - self._mx
        ey   = control_point_y - self._my
        e_ui = follower_x[Heron.SURGE_INTEGRAL_ERROR]    # index 0
        e_ri = follower_x[Heron.YAWRATE_INTEGRAL_ERROR]  # index 2
        uF   = follower_x[Heron.SURGE]                   # index 1
        rC   = follower_x[Heron.YAWRATE]                 # index 3

        d     = jnp.sqrt(control_point_x ** 2 + control_point_y ** 2)
        gamma = jnp.arctan2(ex, ey)

        u_dotF = self._AuF21 * e_ui + self._AuF22 * uF
        r_dotC = self._ArF21 * e_ri + self._ArF22 * rC

        s_tr = jnp.sin(thetarel);  c_tr = jnp.cos(thetarel)
        s_g  = jnp.sin(gamma);     c_g  = jnp.cos(gamma)

        ex_dot = uF * c_tr - self._ml * rC * s_tr + leader_r * (self._my + d * s_g)
        ey_dot = uF * s_tr + self._ml * rC * c_tr - leader_r * (self._mx + d * c_g) - leader_u

        d_dot     = (ex * ex_dot + ey * ey_dot) / d
        gamma_dot = (-ey * ex_dot + ex * ey_dot) / d ** 2

        ex_ddot = (u_dotF * c_tr - uF * s_tr * thetarel_dot
                   - self._ml * (r_dotC * s_tr + rC * c_tr * thetarel_dot)
                   + leader_r * (d_dot * s_g + d * c_g * gamma_dot))
        ey_ddot = (u_dotF * s_tr + uF * c_tr * thetarel_dot
                   + self._ml * (r_dotC * c_tr - rC * s_tr * thetarel_dot)
                   - leader_r * (d_dot * c_g - d * s_g * gamma_dot))

        d_ddot = (((ex_dot**2 + ex*ex_ddot + ey_dot**2 + ey*ey_ddot) * d**2)
                  - (ex*ex_dot + ey*ey_dot)**2) / d**3
        gamma_ddot = (((-ey*ex_ddot + ex*ey_ddot) * d**2)
                      - 2 * (-ey*ex_dot + ex*ey_dot) * (ex*ex_dot + ey*ey_dot)) / d**4

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
            uF,        # f component of e_ui
            rC,        # f component of e_ri
            thetarel_dot,
            r_dotC,    # thetarel_ddot (excludes leader yaw acceleration)
            u_dotF,
            r_dotC,
        ])

    def _G_err(self, leader_p: State, follower_p: State) -> State:
        """
        Input matrix for the 12 error states.
        Shape: (NX_ERR, NU) = (12, 2).
        """
        leader_heading   = leader_p[2]
        follower_heading = follower_p[2]
        thetarel = -(follower_heading - leader_heading)

        g = jnp.zeros((self.NX_ERR, self.NU))
        g = g.at[2, 0].set(-self._AuF21 * jnp.cos(thetarel))
        g = g.at[2, 1].set( self._ml * self._ArF21 * jnp.sin(thetarel))
        g = g.at[5, 0].set(-self._AuF21 * jnp.sin(thetarel))
        g = g.at[5, 1].set(-self._ml * self._ArF21 * jnp.cos(thetarel))
        g = g.at[6, 0].set(-1.0)
        g = g.at[7, 1].set(-1.0)
        return g
    
    def f(self, state: State) -> State:
        err_state, leader_x, leader_p, follower_x, follower_p = self._unpack(state)

        leader_surge = leader_x[self.leader.SURGE]
        leader_yawrate = leader_x[self.leader.YAWRATE]
        leader_heading = leader_p[self.leader.HEADING]

        follower_surge = follower_x[self.follower.SURGE]
        follower_yawrate = follower_x[self.follower.YAWRATE]
        follower_heading = follower_p[self.follower.HEADING]

        leader_control = self.leader.get_leader_control(self._mode)

        f_err = self._f_err(err_state, leader_x, leader_p, follower_x, follower_p)
        f_leader_x = self.leader.A @ leader_x + self.leader.B @ leader_control # contains Bu term
        f_leader_p = jnp.array([leader_surge * jnp.sin(leader_heading),
                                leader_surge * jnp.cos(leader_heading),
                                leader_yawrate])
        f_follower_x = self.follower.A @ follower_x # Bu term in G function
        f_follower_p = jnp.array([follower_surge * jnp.sin(follower_heading),
                                follower_surge * jnp.cos(follower_heading),
                                follower_yawrate])
        f = jnp.concatenate([f_err, f_leader_x, f_leader_p, f_follower_x, f_follower_p])
        return f
    
    def G(self, state: State) -> State:
        err_state, leader_x, leader_p, follower_x, follower_p = self._unpack(state)

        g_err = self._G_err(leader_p, follower_p)
        g_leader_x = jnp.zeros((4, 2), dtype=jnp.float32) # absorbed into f function
        g_leader_p = jnp.zeros((3, 2), dtype=jnp.float32) # absorbed into f function
        g_follower_x = self.follower.B
        g_follower_p = jnp.zeros((3, 2), dtype=jnp.float32) # absorbed into f function
        g = jnp.vstack([g_err, g_leader_x, g_leader_p, g_follower_x, g_follower_p])
        
        return g
    
    # ------------------------------------------------------------------
    # xdot — the function diffeqsolve calls; integrates all 26 dims
    # ------------------------------------------------------------------
    def xdot(self, state: State, control: Control) -> State:
        """
        Full 26-dim time derivative.
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
        ex = state[self.EX]
        ey = state[self.EY]

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
        return self.initial_state(
            np.zeros(self.NX_ERR, dtype=np.float32))

    def nominal_val_state(self) -> State:
        """Returns the full 26-dim state."""
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 
                         0, 0, 0, 
                         0, 0, 0, 0, 
                         self._ml, 0, 0],
                        dtype=np.float32)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        """Bounds over the full 26-dim state vector."""
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
            (-1.5,  1.5),        # thetarel_dot
            (  0,   4),          # uF
            ( -1,   1),          # rC
        ], dtype=np.float32)

        veh_bounds = np.array([
            (-60, 60),   # leader surge_int_err
            (  0,  4),   # leader surge
            (-40, 40),   # leader yaw_rate_int_err
            ( -1,  1),   # leader yaw_rate
            (-100, 100), # leader px
            (-100, 100), # leader py
            (-np.pi, np.pi),  # leader heading
            (-60, 60),   # follower surge_int_err
            (  0,  4),   # follower surge
            (-40, 40),   # follower yaw_rate_int_err
            ( -1,  1),   # follower yaw_rate
            (-100, 100), # follower px
            (-100, 100), # follower py
            (-np.pi, np.pi),  # follower heading
        ], dtype=np.float32)

        bounds = np.concatenate([err_bounds, veh_bounds], axis=0)
        return bounds.T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    # ------------------------------------------------------------------
    # Nominal policy  (reads from state vector — no self.leader access)
    # ------------------------------------------------------------------
    def nom_pol_goto(self, state: State, goal: State = None) -> Control:
        """
        Guidance law. Reads leader/follower state from the full state vector —
        never from self.leader or self.follower, so safe inside a JAX trace.
        """
        _, leader_x, leader_p, follower_x, follower_p = self._unpack(state)

        leader_heading = leader_p[self.leader.HEADING]
        leader_surge   = leader_x[Heron.SURGE]
        follower_heading = follower_p[self.follower.HEADING]

        fx_leader, fy_leader = self._global_to_leader(leader_p, follower_p)

        cross_track_error = fx_leader - self._mx
        along_track_error = fy_leader - self._my

        max_speed = self._u_max_guidance[0]
        min_speed = self._u_min_guidance[0]
        guidance_speed = (leader_surge
                          - max_speed * (along_track_error
                                         / jnp.sqrt(along_track_error**2
                                                    + self._speed_control_gain**2)))
        desired_speed = jnp.clip(guidance_speed, min_speed, max_speed)

        chiR          = jnp.arctan2(-cross_track_error, self._lookahead)
        desired_heading  = leader_heading + chiR * self.RAD2DEG

        # now convert to a delta theta
        delta_theta = -(follower_heading - desired_heading)
        desired_yaw_rate = delta_theta * self._heading_rate_gain # does not include filter from pTrajectTranslate

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