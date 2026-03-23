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
    NX: int = 14
    NU: int = 2

    EX, EXDOT, EXDDOT, EY, EYDOT, EYDDOT, EUI, EUR, THETAREL, THETARELDOT, UF, RC, D, GAMMA = range(NX)
    VELREF, YAWREF = range(NU)

    DT = 0.04

    def __init__(self) -> None:
        self._dt = Error.DT

        # guidance law parameters
        self._u_min_guidance = jnp.array([0.0, -0.6])
        self._u_max_guidance = jnp.array([2.0, 0.6])
        self._lookahead: float = 10.0
        self._speed_control_gain: float = 30.0
        self._yaw_rate_gain: float = 0.2
        self._mode = "straight"
    
        # constant geometric parameters
        self._mx: float = 10.0 # horizontal safe region offset
        self._my: float = 0.0 # vertical safe region offset
        self._ml: float = 5.0 # control point offset

        # dynamics coefficients
        self._AuF21: float = -1.0
        self._AuF22: float = -24.048562
        self._ArF21: float = -6.3246
        self._ArF22: float = -61.78262

        # safety parameters
        self._a: float = 10.0
        self._b: float = 5.0
        self._s: float = 1/((self._a * self._b)**2)
        self._u_min_filter = jnp.array([0.0, -0.6])
        self._u_max_filter = jnp.array([2.0, 0.6])

        # Heron Vehicles
        leader_start_position = jnp.array([[0.0], [0.0], [0.0]], dtype=jnp.float32)
        follower_start_position = jnp.array([[self._a], [0.0], [0.0]], dtype=jnp.float32)
        self.leader: Heron = Heron(leader_start_position)
        self.follower: Heron = Heron(follower_start_position)

        # helper values
        self.DEG2RAD = jnp.pi / 180.0
        self.RAD2DEG = 180.0 / jnp.pi

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
    def x_labels(self) -> List[str]:
        return [r"$e_x$", r"$\dot{e}_x$", r"$\ddot{e}_x$",
                r"$e_y$", r"$\dot{e}_y$", r"$\ddot{e}_y$",
                r"$e_{u_I}^F$", r"$\e_{r_I}^F$",
                r"$\theta_{\text{rel}}$", r"$\dot{\theta}_{\text{rel}}$", 
                r"$u^F$", r"$u^L$", r"$r^C$", r"$r^L$",
                r"$d$", r"$\dot{d}$", r"$\ddot{d}$",
                r"$\gamma$", r"$\dot{\gamma}$", r"$\ddot{\gamma}$"]
    
    @property
    def u_labels(self) -> List[str]:
        return [r"$u_{\text{cmd}}^F$", r"$r_{\text{cmd}}^C$"]
    
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
    @staticmethod
    def heading_to_theta(heading):
        yaw = jnp.fmod(heading + jnp.pi/2, 2*jnp.pi)
        return yaw
    
    def global_to_leader(self):
        # calculate relative position using globalToLeader function from pUNREPCBF
        del_x = self.follower.position[0][0] - self.leader.position[0][0]
        del_y = self.follower.position[1][0] - self.follower.position[1][0]

        leader_heading = self.leader.position[2][0]
        leader_heading_rad = leader_heading * self.DEG2RAD
        follower_x_leader = del_x * jnp.cos(leader_heading_rad) - del_y * jnp.sin(leader_heading_rad)
        follower_y_leader = del_x * jnp.sin(leader_heading_rad) + del_y * jnp.sin(leader_heading_rad)
        return (follower_x_leader, follower_y_leader)
    
    def f(self, state: State) -> State:
        # ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)

        # extract leader relevant terms
        leader_u = self.leader.surge_state[1][0]
        leader_r = self.leader.yaw_rate_state[1][0]

        thetarel = -(self.follower.position[2][0] - self.leader.position[2][0]) # delta_theta = -delta_yaw
        thetarel_dot = -(self.follower.x[3][0] - self.leader.x[3][0]) # thetarel_dot = -yawrel_dot
        
        follower_x_leader, follower_y_leader = self.global_to_leader()
        control_point_x_leader = follower_x_leader + self._ml * jnp.cos(thetarel)
        control_point_y_leader = follower_y_leader + self._ml * jnp.sin(thetarel)
        
        # calculate state values from leader and follower states
        ex = control_point_x_leader - self._mx
        ey = control_point_y_leader - self._my
        e_ui = self.follower.x[0][0]
        e_ri = self.follower.x[2][0]
        
        uF = self.follower.x[1][0]
        rC = self.follower.x[3][0]
        d = (control_point_x_leader**2 + control_point_y_leader**2) ** 0.5
        gamma = jnp.atan2(ex, ey)

        u_dotF = self._AuF21*e_ui + self._AuF22*uF
        r_dotC = self._ArF21*e_ri + self._ArF22*rC
        s_thetarel = jnp.sin(thetarel)
        c_thetarel = jnp.cos(thetarel)
        s_gamma = jnp.sin(gamma)
        c_gamma = jnp.cos(gamma) 

        ex_dot = uF*c_thetarel - self._ml*rC*s_thetarel + leader_r*(self._my + d*s_gamma)
        ey_dot = uF*s_thetarel + self._ml*rC*c_thetarel - leader_r*(self._mx + d*c_gamma) - leader_u

        d_dot = (ex*ex_dot + ey*ey_dot)/d
        gamma_dot = (-ey*ex_dot + ex*ey_dot)/(d**2)
        
        ex_ddot = u_dotF*c_thetarel - uF*s_thetarel*thetarel_dot - self._ml*(r_dotC*s_thetarel + rC*c_thetarel*thetarel_dot) + leader_r*(d_dot*s_gamma + d*c_gamma*gamma_dot)
        ey_ddot = u_dotF*s_thetarel + uF*c_thetarel*thetarel_dot + self._ml*(r_dotC*c_thetarel - rC*s_thetarel*thetarel_dot) - leader_r*(d_dot*c_gamma - d*s_gamma*gamma_dot)

        d_ddot = ((ex_dot**2 + ex*ex_ddot + ey_dot**2 + ey*ey_ddot)*(d**2) - (ex*ex_dot + ey*ey_dot)**2)/(d**3)
        gamma_ddot = ((-ey*ex_ddot + ex*ey_ddot)*(d**2) - 2*(-ey*ex_dot + ex*ey_dot)*(ex*ex_dot + ey*ey_dot))/(d**4)
        # simplified from what's in the reference document
        
        ex_dddot = (self._AuF21*uF + self._AuF22*u_dotF)*c_thetarel - 2*u_dotF*s_thetarel*thetarel_dot - uF*(c_thetarel*thetarel_dot**2 + s_thetarel*r_dotC) - self._ml*((self._ArF21*rC + self._ArF22*r_dotC)*s_thetarel + 2*r_dotC*c_thetarel*thetarel_dot + rC*(c_thetarel*r_dotC - s_thetarel*thetarel_dot**2)) + leader_r*(d_ddot*s_gamma + 2*d_dot*c_gamma*gamma_dot + d*(c_gamma*gamma_ddot - s_gamma*gamma_dot**2))
        ey_dddot = (self._AuF21*uF + self._AuF22*u_dotF)*s_thetarel + 2*u_dotF*c_thetarel*thetarel_dot + uF*(c_thetarel*r_dotC - s_thetarel * thetarel_dot**2) + self._ml*((self._ArF21*rC + self._ArF22*r_dotC)*c_thetarel - 2*r_dotC*s_thetarel*thetarel_dot - rC*(c_thetarel*thetarel_dot**2 + s_thetarel*r_dotC)) - leader_r*(d_ddot*c_gamma - 2*d_dot*s_gamma*gamma_dot - d*(c_gamma*gamma_dot**2 + s_gamma*gamma_ddot))

        output = jnp.array([
            ex_dot, 
            ex_ddot,
            ex_dddot, # f component of ex_dddot
            ey_dot,
            ey_ddot,
            ey_dddot, # f component of ey_dddot
            uF, # f component of e_ui
            rC, # f component of e_ri
            thetarel_dot, # f component of thetarel_dot
            r_dotC - 0, # thetarel_ddot, not including leader yaw acceleration
            u_dotF,
            r_dotC,
            d_dot, # d_dot
            gamma_dot, # gamma_dot
            ])
        return output
    
    def G(self, state: State) -> State:
        # ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)
        thetarel = -(self.follower.position[2][0] - self.leader.position[2][0]) # delta_theta = -delta_yaw

        g = jnp.zeros((self.NX, self.NU))
        g = g.at[2, 0].set(-self._AuF21 * jnp.cos(thetarel))
        g = g.at[2, 1].set(self._ml * self._ArF21 * jnp.sin(thetarel))

        g = g.at[5, 0].set(-self._AuF21 * jnp.sin(thetarel))
        g = g.at[5, 1].set(-self._ml * self._ArF21 * jnp.cos(thetarel))

        g = g.at[6, 0].set(-1)
        g = g.at[7, 1].set(-1)

        return g
    
    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        leader_control: Control = self.leader.get_leader_control(self.mode)
        self.leader.step(leader_control)
        self.follower.step(control)

        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)
    
    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None) -> tuple[TState, TFloat]:
        leader_control: Control = self.leader.get_leader_control(self.mode)
        self.leader.step(leader_control)
        self.follower.step(control)

        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)
    
    # ------------------------------------------------------------------
    # Safety constraint h
    # ------------------------------------------------------------------

    def h_components(self, state: State) -> HFloat:
        ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)

        # Negative means unsafe (outside or on the obstacle).
        h_obs = self._s*((self._a*self._b)**2 - (self._b * ex)**2 - (self._a * ey)**2)

        # h <= 1
        hs = poly4_clip_max_flat(jnp.array([h_obs]))
        # clip h >= h_min
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)
        obs = jnp.array([ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma])
        return obs, obs
    
    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    # TODO: Check with Max
    def has_eq_state(self) -> bool:
        return False

    def eq_state(self) -> State:
        # TODO: Check with Max
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self._mx, 0], dtype=np.float32)

    # TODO: Define starting conditions
    def nominal_val_state(self) -> State:
        # Start in the safe region
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # TODO: check with MAX
    def train_bounds(self) -> Float[Arr, "2 nx"]:
        # there are set by taking the largest absolute value in sim and doubling it (at minimum)
        return np.array([(-4, 4), # ex
                         (-2, 2), # ex_dot
                         (-140, 140), #ex_ddot (big spikes when hitting the safe region boundary)
                         (-16, 16), # ey
                         (-4, 4), # ey_dot
                         (-60, 60), # ey_ddot
                         (-60, 60), # eui
                         (-40, 40), # eri
                         (-jnp.pi, jnp.pi), # thetarel
                         (-1.5, 1.5), # thetarel_dot
                         (0, 4), # uF
                         (-1, 1), # rC
                         (-20, 20), # d
                         (-jnp.pi, jnp.pi) # gamma
                         ], dtype=np.float32).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()
    
    # ------------------------------------------------------------------
    # Nominal policy: steer toward the center of the safe region
    # ------------------------------------------------------------------
    def nom_pol_goto(self, state: State, goal: State = None) -> Control:
        leader_heading: float = self.leader.position[2][0]
        follower_x_leader, follower_y_leader = self.global_to_leader()
        
        # set relative positions as done in BHV_UNREP
        cross_track_error: float = follower_x_leader - self._mx
        along_track_error: float = follower_y_leader - self._my

        # speed control
        max_speed: float = self._u_max_guidance[0]
        min_speed: float = self._u_min_guidance[0]
        guidance_speed: float = self.leader.surge_state[1][0] - max_speed * (along_track_error / (along_track_error**2 + self._speed_control_gain**2)**0.5)
        desired_speed = jnp.clip(guidance_speed, min_speed, max_speed)

        # yaw rate control
        chiR: float = jnp.arctan2(-cross_track_error, self._lookahead)
        desired_heading: float = leader_heading + chiR * self.RAD2DEG
        desired_yaw_rate: float = desired_heading * self._yaw_rate_gain # gain from pTrajectTranslate

        control: Control = jnp.array([desired_speed, desired_yaw_rate], dtype=jnp.float32)
        return control

    def nom_pol_zero(self, state: State, goal: State = None) -> Control:
        zero: Control = jnp.array([0.0, 0.0], dtype=jnp.float32)
        return zero

    def has_episode_pol(self) -> bool:
        return self.has_episode_pol

    def make_episode_pol(self, key, nom_pol):
        return ft.partial(nom_pol)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.EX, self.EY]))]

    def plot_phase(self, ax: plt.Axes):
        """XY plane plot with Heron obstacle."""
        PLOT_XMIN, PLOT_XMAX = -15, 15
        PLOT_YMIN, PLOT_YMAX = -15, 15
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[0], ylabel=self.x_labels[4])
        ax.set_aspect("equal")