import functools as ft

import jax.numpy as jnp
import numpy as np

from jaxtyping import Float

from pncbf.dyn.heron import Heron
from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.none import get_or
from pncbf.utils.jax_types import Arr, TFloat

from typing import List

class HERON(Task):
    NX: int = 14
    NU: int = 2

    EX, EXDOT, EXDDOT, EY, EYDOT, EYDDOT, EUI, EUR, THETAREL, THETARELDOT, UF, RC, D, GAMMA = range(NX)
    VELREF, YAWREF = range(NU)

    def __init__(self) -> None:
        self._dt = 0.04

        # guidance law parameters
        self._u_min_guidance: np.ndarray[float] = np.array([0, -0.6])
        self._u_max_guidance: np.ndarray[float] = np.array([2, 0.6])
        self._lookahead: float = 10.0
        self._yaw_rate_gain: float = 0.2
    
        # constant geometric parameters
        self._mx: float = 10.0 # horizontal safe region offset
        self._my: float = 0.0 # vertical safe region offset
        self._ml: float = 5.0 # control point offset

        # constant leader properties
        # TODO: uL and rL??
        self._uL: float = 0.0
        self._rL: float = 0.0

        # dynamics coefficients
        self._AuF21: float = -1.0
        self._AuF22: float = -24.048562
        self._ArF21: float = -6.3246
        self._ArF22: float = -61.78262

        # safety parameters
        self._a: float = 10.0
        self._b: float = 5.0
        self._s: float = 1/((self._a * self._b)**2)
        self._u_min_filter: np.ndarray[float] = np.array([0, -0.6])
        self._u_max_filter: np.ndarray[float] = np.array([2, 0.6])

        # Heron Vehicles
        leader_start_position: np.ndarray[float] = np.array([[0], [0], [0]])
        follower_start_position: np.ndarray[float] = np.array([[self._a], [0], [0]])
        self.leader: Heron = Heron(leader_start_position)
        self.follower: Heron = Heron(follower_start_position)

        # helper values
        self.DEG2RAD = np.pi / 180.0
        self.RAD2DEG = 180.0 / np.pi

    # ------------------------------------------------------------------
    # Required Task interface
    # ------------------------------------------------------------------

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
    
    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    # def f(self, state: State) -> State:
    #     ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)

    #     u_dotF = self._AuF21*e_ui + self._AuF22*uF
    #     r_dotC = self._ArF21*e_ri + self._ArF22*rC
    #     s_thetarel = jnp.sin(thetarel)
    #     c_thetarel = jnp.cos(thetarel)
    #     s_gamma = jnp.sin(gamma)
    #     c_gamma = jnp.cos(gamma) 
    #     d_dot = (ex*ex_dot + ey*ey_dot)/d
    #     d_ddot = ((ex_dot**2 + ex*ex_ddot + ey_dot**2 + ey*ey_ddot)*d**2 - (ex*ex_dot + ey*ey_dot)**2)/(d**3)
    #     gamma_dot = (-ey*ex_dot + ex*ey_dot)/(d**2)
    #     gamma_ddot = ((-ey_dot*ex_dot - ey*ex_ddot + ex_dot*ey_dot + ex*ey_ddot)*d**2 - 2*(-ey*ex_dot + ex*ey_dot)*(ex*ex_dot + ey*ey_dot))/(d**4)

    #     output = jnp.array([
    #         uF*c_thetarel - self._ml*rC*s_thetarel + self._rL*(self._my + d*s_gamma), #ex_dot

    #         u_dotF*c_thetarel - uF*s_thetarel*thetarel_dot - self._ml*(r_dotC*s_thetarel + rC*c_thetarel*thetarel_dot) + self._rL*(d_dot*s_gamma + d*c_gamma*gamma_dot), #ex_ddot

    #         (self._AuF21*uF + self._AuF22*u_dotF)*c_thetarel - 2*u_dotF*s_thetarel*thetarel_dot - uF*(c_thetarel*thetarel_dot**2 + s_thetarel*r_dotC) - self._ml*((self._ArF21*rC + self._ArF22*r_dotC)*s_thetarel + 2*r_dotC*c_thetarel*thetarel_dot + rC*(c_thetarel*r_dotC - s_thetarel*thetarel_dot**2)) + self._rL*(d_ddot*s_gamma + 2*d_dot*c_gamma*gamma_dot + d*(c_gamma*gamma_ddot - s_gamma*gamma_dot**2)), #ex_dddot

    #         uF*s_thetarel + self._ml*rC*c_thetarel - self._rL*(self._mx + d*c_gamma) - self._uL, #ey_dot

    #         u_dotF*s_thetarel + uF*c_thetarel*thetarel_dot + self._ml*(r_dotC*c_thetarel - rC*s_thetarel*thetarel_dot) - self._rL*(d_dot*c_gamma - d*s_gamma*gamma_dot) # ey_ddot

    #         (self._AuF21*uF + self._AuF22*u_dotF)*s_thetarel + 2*u_dotF*c_thetarel*thetarel_dot + uF*(c_thetarel*r_dotC - s_thetarel * thetarel_dot**2) + self._ml*((self._ArF21*rC + self._ArF22*r_dotC)*c_thetarel - 2*r_dotC*s_thetarel*thetarel_dot - rC*(c_thetarel*thetarel_dot**2 + s_thetarel*r_dotC)) - self._rL*(d_ddot*c_gamma - 2*d_dot*s_gamma*gamma_dot - d*(c_gamma*gamma_dot**2 + s_gamma*gamma_ddot)), # ey_dddot

    #         uF, # e_ui

    #         rC, # e_ri

    #         rC - self._rL, # thetarel_dot

    #         r_dotC - 0, # thetarel_ddot,

    #         u_dotF, # u_dotF

    #         r_dotC, # r_dotC

    #         d_dot, # d_dot

    #         gamma_dot, # gamma_dot
    #         ])
    #     return output
    
    # def G(self, state: State) -> State:
    #     ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)

    #     g = np.zeros((self.NX, self.NU))
    #     g[2, 0] = -self._AuF21*jnp.cos(thetarel)
    #     g[2, 1] = self._ml*self._ArF21*jnp.sin(thetarel)

    #     g[5, 0] = -self._AuF21*jnp.sin(thetarel)
    #     g[5, 1] = -self._ml*self._ArF21*jnp.cos(thetarel)

    #     g[6, 0] = -1
    #     g[7, 1] = -1

    #     return g
    
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
        ex, ex_dot, ex_ddot, ey, ey_dot, ey_ddot, e_ui, e_ri, thetarel, thetarel_dot, uF, rC, d, gamma = self.chk_x(state)

        # Negative means unsafe (outside or on the obstacle).
        # h = distance metric from center of safe region
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

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        # TODO: Check with Max
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self._uL, self._rL, self._mx, 0])

    # TODO: Define starting conditions
    def nominal_val_state(self) -> State:
        # Start in the safe region
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # TODO: consider what the proper range is for UNREP
    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.5, 2.5), 
                         (-2.5, 2.5), 
                         (-np.pi, np.pi)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()
    
    # ------------------------------------------------------------------
    # Nominal policy: steer toward the center of the safe region
    # ------------------------------------------------------------------
    def nom_pol_goto(self, state: State, goal: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])) -> Control:
        # calculate relative position using globalToLeader function from pUNREPCBF
        del_x: float = self.follower.position[0][0] - self.leader.position[0][0]
        del_y: float = self.follower.position[1][0] - self.follower.position[1][0]

        leader_heading: float = self.leader.position[2][0]
        leader_heading_rad: float = leader_heading * self.DEG2RAD
        follower_x_leader: float = del_x * np.cos(leader_heading_rad) - del_y * np.sin(leader_heading_rad)
        follower_y_leader: float = del_x * np.sin(leader_heading_rad) + del_y * np.sin(leader_heading_rad)

        # set relative positions as done in BHV_UNREP
        cross_track_error: float = follower_x_leader - self._mx
        along_track_error: float = follower_y_leader - self._my

        # speed control
        max_speed: float = self._u_max_guidance[0]
        min_speed: float = self._u_min_guidance[0]
        guidance_speed: float = self._uL - max_speed * (along_track_error / (along_track_error**2 + self._speed_control_gain**2)**0.5)
        desired_speed: float = min(max(min_speed, guidance_speed), max_speed)

        # yaw rate control
        chiR: float = np.arctan2(-cross_track_error, self._lookahead)
        desired_heading: float = leader_heading + chiR * self.RAD2DEG
        desired_yaw_rate: float = desired_heading * self._yaw_rate_gain # gain from pTrajectTranslate

        control: Control = np.array([desired_speed, desired_yaw_rate])
        return control

    def nom_pol_zero(self, state: State, goal=None) -> Control:
        zero: Control = jnp.array([0.0, 0.0])
        return zero

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
