"""3D quadrotor with 12-dimensional state space.

State (12):  [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
Control (4): [uT, u_tau_x, u_tau_y, u_tau_z], each in [-1, 1]

  T_total = T_HOVER + T_DELTA * uT         (thrust, [0, 2*m*g])
  tau_x   = TAU_MAX * u_tau_x
  tau_y   = TAU_MAX * u_tau_y
  tau_z   = TAU_Z_MAX * u_tau_z

Base safety constraints (6):
  floor:    pz >= Z_MIN
  ceil:     pz <= Z_MAX
  roll_lb:  phi >= -PHI_MAX
  roll_ub:  phi <=  PHI_MAX
  pitch_lb: theta >= -THETA_MAX
  pitch_ub: theta <=  THETA_MAX

Optional sphere obstacles (1 constraint each):
  h_obs_i = 1 - ||p - p_obs_i||^2 / r_i^2   (> 0 when inside sphere)
  Pass obstacles=[(cx, cy, cz, r), ...] to __init__.

Hand-designed CBF:
  Altitude floor/ceil: stopping-distance barrier (relative degree 1 after extension)
    B_floor = (pz - Z_MIN) - min(vz, 0)^2 / (2 * g)
  Attitude + obstacles: HOCBF extension via hocbf() utility (relative degree 2 -> 1)
  Convention: h < 0 => safe, matching h_components.
"""

import functools as ft

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from mrncbf.dyn.dyn_types import Control, HFloat, PolObs, State, VObs
from mrncbf.dyn.task import Task
from mrncbf.utils.costconstr_utils import poly4_clip_max_flat
from mrncbf.utils.hocbf import hocbf
from mrncbf.utils.jax_types import Arr
from mrncbf.utils.sampling_utils import get_mesh_np


class Quad3D(Task):
    NX = 12
    NU = 4
    DT = 0.01   # 100 Hz control

    PX, PY, PZ, VX, VY, VZ, PHI, THETA, PSI, P, Q, R = range(NX)
    UT, UTAUX, UTAUY, UTAUZ = range(NU)

    # Physical parameters
    MASS = 1.0
    GRAV = 9.81                # gravitational acceleration [m/s²]  (G is reserved for input matrix)
    IXX  = 0.01
    IYY  = 0.01
    IZZ  = 0.02
    T_HOVER   = MASS * GRAV    # hover thrust [N]
    T_DELTA   = MASS * GRAV    # per-unit thrust deviation → T ∈ [0, 2mg]
    TAU_MAX   = 1.0            # roll / pitch torque scale [N·m]
    TAU_Z_MAX = 0.5            # yaw torque scale [N·m]

    # Safety limits
    # Z_MIN     = 0.1            # floor [m]
    Z_MIN     = 0.1            # floor [m]
    Z_MAX     = 3.5            # ceiling [m]
    PHI_MAX   = np.pi / 2      # roll  limit [rad]
    THETA_MAX = np.pi / 2      # pitch limit [rad]

    # Hover target
    Z_HOVER = 1.5

    def __init__(self, obstacles=None):
        """
        Args:
            obstacles: list of (cx, cy, cz, r) tuples defining sphere obstacles.
                       Each adds one h_component and one handcbf_B constraint.
        """
        self._dt = self.DT
        self.obstacles = obstacles or []

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n_Vobs(self) -> int:
        return self.NX

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def x_labels(self) -> list[str]:
        return ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'φ', 'θ', 'ψ', 'p', 'q', 'r']

    @property
    def u_labels(self) -> list[str]:
        return ['uT', 'uτx', 'uτy', 'uτz']

    @property
    def h_labels(self) -> list[str]:
        base = ['floor', 'ceil', 'roll_lb', 'roll_ub', 'pitch_lb', 'pitch_ub']
        obs  = [f'obs_{i}' for i in range(len(self.obstacles))]
        return base + obs

    # ── Observation ─────────────────────────────────────────────────────────────

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        return state, state

    # ── Dynamics ────────────────────────────────────────────────────────────────

    def _thrust_dir(self, state: State) -> jnp.ndarray:
        """Body z-axis in world frame (ZYX Euler convention, 3rd column of R)."""
        phi, theta, psi = state[self.PHI], state[self.THETA], state[self.PSI]
        cp, sp   = jnp.cos(phi),   jnp.sin(phi)
        ct, st   = jnp.cos(theta), jnp.sin(theta)
        cps, sps = jnp.cos(psi),   jnp.sin(psi)
        return jnp.array([
            cps * st * cp + sps * sp,
            sps * st * cp - cps * sp,
            ct  * cp,
        ])

    def f(self, state: State) -> State:
        """Drift vector (hover thrust + gravity + kinematics, no control)."""
        px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = state

        cp, sp = jnp.cos(phi),   jnp.sin(phi)
        ct     = jnp.cos(theta)
        tt     = jnp.tan(theta)
        ct_safe = jnp.where(jnp.abs(ct) < 1e-3, jnp.sign(ct + 1e-10) * 1e-3, ct)

        Rz = self._thrust_dir(state)
        a_hover = (self.T_HOVER / self.MASS) * Rz - jnp.array([0.0, 0.0, self.GRAV])

        return jnp.array([
            vx, vy, vz,
            a_hover[0], a_hover[1], a_hover[2],
            p + (q * sp + r * cp) * tt,    # phi-dot
            q * cp - r * sp,                # theta-dot
            (q * sp + r * cp) / ct_safe,   # psi-dot
            (self.IYY - self.IZZ) / self.IXX * q * r,
            (self.IZZ - self.IXX) / self.IYY * p * r,
            (self.IXX - self.IYY) / self.IZZ * p * q,
        ])

    def G(self, state: State) -> jnp.ndarray:
        """Input matrix (12 × 4)."""
        Rz = self._thrust_dir(state)
        G = jnp.zeros((self.NX, self.NU))
        scale_T = self.T_DELTA / self.MASS
        G = G.at[self.VX,    self.UT   ].set(scale_T * Rz[0])
        G = G.at[self.VY,    self.UT   ].set(scale_T * Rz[1])
        G = G.at[self.VZ,    self.UT   ].set(scale_T * Rz[2])
        G = G.at[self.P,     self.UTAUX].set(self.TAU_MAX   / self.IXX)
        G = G.at[self.Q,     self.UTAUY].set(self.TAU_MAX   / self.IYY)
        G = G.at[self.R,     self.UTAUZ].set(self.TAU_Z_MAX / self.IZZ)
        return G

    # ── Safety constraints ───────────────────────────────────────────────────────

    def h_components(self, state: State) -> HFloat:
        """h > 0 means unsafe.

        Base 6 constraints: altitude floor/ceil + roll/pitch limits.
        Optional extra: one sphere obstacle constraint per obstacle in self.obstacles.
        Obstacle constraint: h_obs = 1 - ||p - p_obs||^2 / r^2
          (positive = inside sphere = unsafe, normalised so h=1 at center, h=0 at surface)
        """
        pz    = state[self.PZ]
        phi   = state[self.PHI]
        theta = state[self.THETA]

        h_floor    = -(pz - self.Z_MIN)
        h_ceil     = -(self.Z_MAX - pz)
        h_roll_lb  = -(phi   + self.PHI_MAX)
        h_roll_ub  = -(self.PHI_MAX   - phi)
        h_pitch_lb = -(theta + self.THETA_MAX)
        h_pitch_ub = -(self.THETA_MAX - theta)

        hs = jnp.array([h_floor, h_ceil, h_roll_lb, h_roll_ub, h_pitch_lb, h_pitch_ub])

        if self.obstacles:
            px, py = state[self.PX], state[self.PY]
            h_obs_list = []
            for cx, cy, cz, r in self.obstacles:
                d2 = (px - cx)**2 + (py - cy)**2 + (pz - cz)**2
                h_obs_list.append(1.0 - d2 / (r ** 2))
            hs = jnp.concatenate([hs, jnp.array(h_obs_list)])

        hs = poly4_clip_max_flat(hs, max_val=self.h_max)
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    # ── Hand-designed CBF ────────────────────────────────────────────────────────

    def handcbf_B(self, state: State, alpha_obs: float = 2.0, alpha_att: float = 2.0) -> HFloat:
        """Valid CBF barriers for all constraints, returned as h < 0 = safe.

        Altitude floor/ceil:
          Stopping-distance barrier (manually constructed, relative degree 1).
          B_floor = (pz - Z_MIN) - min(vz, 0)^2 / (2*g)

        Attitude (roll/pitch limits):
          HOCBF extension psi1 = dh/dt|_f + alpha_att * h (relative degree 2 -> 1).
          Control appears in dpsi1/dt via angular accelerations.

        Obstacles:
          HOCBF extension psi1 = dh/dt|_f + alpha_obs * h (relative degree 2 -> 1).
          Control appears in dpsi1/dt via translational acceleration (thrust).

        All constraints share the h < 0 = safe output convention.
        """
        a_max = self.GRAV   # max upward net accel (conservative: T_max/m - g = g)
        pz, vz = state[self.PZ], state[self.VZ]

        # ── Altitude: stopping-distance (relative degree 1) ───────────────────
        B_floor = (pz - self.Z_MIN) - jnp.minimum(vz, 0.0) ** 2 / (2 * a_max)
        B_ceil  = (self.Z_MAX - pz) - jnp.maximum(vz, 0.0) ** 2 / (2 * a_max)

        # ── Attitude: HOCBF (relative degree 2, h >= 0 = safe convention) ────
        # psi1 = d/dt(h_att)|_f + alpha_att * h_att  >= 0 = safe
        # LG psi1 != 0 because torques drive p,q,r which appear in phi-dot, theta-dot
        def h_att_safe(x):
            return jnp.array([
                x[self.PHI]   + self.PHI_MAX,
                self.PHI_MAX  - x[self.PHI],
                x[self.THETA] + self.THETA_MAX,
                self.THETA_MAX - x[self.THETA],
            ])

        psi1_att = hocbf(h_att_safe, self.f, alpha0=alpha_att, state=state)

        hs = jnp.concatenate([
            jnp.array([-B_floor, -B_ceil]),
            -psi1_att,
        ])

        # ── Obstacles: HOCBF (relative degree 2, thrust drives velocity) ──────
        # Use linear surface distance d - r (unit gradient).  Lower alpha_obs
        # keeps the CBF feasible at higher speeds within bounded inputs.
        if self.obstacles:
            def h_obs_safe(x):
                vals = []
                for cx, cy, cz, r in self.obstacles:
                    d = jnp.sqrt((x[self.PX]-cx)**2 + (x[self.PY]-cy)**2 + (x[self.PZ]-cz)**2)
                    vals.append(d - r)   # >= 0 outside sphere = safe
                return jnp.array(vals)

            psi1_obs = hocbf(h_obs_safe, self.f, alpha0=alpha_obs, state=state)
            hs = jnp.concatenate([hs, -psi1_obs])

        return hs

    # ── Nominal policy (cascade PD hover) ────────────────────────────────────────

    def nom_pol_hover(self, state: State) -> Control:
        """Cascade PD controller hovering at (0, 0, Z_HOVER)."""
        px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = state

        kp_z, kd_z = 4.0, 3.0
        u_T = jnp.clip(-kp_z * (pz - self.Z_HOVER) - kd_z * vz, -1.0, 1.0)

        kp_xy, kd_xy = 0.4, 1.2
        theta_des = jnp.clip(-(kp_xy * px + kd_xy * vx) / self.GRAV, -0.3, 0.3)
        phi_des   = jnp.clip( (kp_xy * py + kd_xy * vy) / self.GRAV, -0.3, 0.3)

        # Gains chosen for discrete-time stability at DT=0.05 s.
        # Stability requires DT * (TAU_MAX/IXX) * kd_att < 2, i.e. kd_att < 0.4.
        # kp_att < kd_att / DT = 8 keeps the determinant condition |det| < 1.
        kp_att, kd_att = 4.0, 0.3
        kp_psi, kd_psi = 1.0, 0.2
        u_tau_x = jnp.clip(kp_att * (phi_des   - phi)   - kd_att * p, -1.0, 1.0)
        u_tau_y = jnp.clip(kp_att * (theta_des - theta) - kd_att * q, -1.0, 1.0)
        u_tau_z = jnp.clip(-kp_psi * psi - kd_psi * r,               -1.0, 1.0)

        return jnp.array([u_T, u_tau_x, u_tau_y, u_tau_z])

    @property
    def nom_pol_pid(self):
        return self.nom_pol_hover

    # ── Equilibrium / bounds ──────────────────────────────────────────────────────

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        x = np.zeros(self.NX)
        x[self.PZ] = self.Z_HOVER
        return x

    def nominal_val_state(self) -> State:
        return self.eq_state()

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        lb = np.array([-2.5, -2.5,  0.0, -3.0, -3.0, -3.0,
                       -np.pi / 2, -np.pi / 2, -np.pi, -4.0, -4.0, -2.0])
        ub = np.array([ 2.5,  2.5,  4.0,  3.0,  3.0,  3.0,
                        np.pi / 2,  np.pi / 2,  np.pi,  4.0,  4.0,  2.0])
        return np.stack([lb, ub], axis=0)

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        bounds = self.train_bounds().copy()
        bounds[:, self.PZ] = np.array([-0.1, 4.1])
        bounds[:, self.VZ] = np.array([-3.3, 3.3])
        return bounds

    # ── Phase2D setups ────────────────────────────────────────────────────────────

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [
            Task.Phase2DSetup("pz_vz",     self.plot_pz_vz,    Task.mk_get2d([self.PZ, self.VZ])),
            Task.Phase2DSetup("phi_theta",  self.plot_phi_theta, Task.mk_get2d([self.PHI, self.THETA])),
            Task.Phase2DSetup("px_py",     self.plot_px_py,    Task.mk_get2d([self.PX, self.PY])),
        ]

    # ── Plot helpers ──────────────────────────────────────────────────────────────

    def plot_pz_vz(self, ax: plt.Axes):
        ax.set(xlim=(-0.1, 4.1), ylim=(-3.5, 3.5),
               xlabel='pz (m)', ylabel='vz (m/s)')
        ax.axvline(self.Z_MIN, color='C3', lw=1.0, ls='--', alpha=0.8, label='floor')
        ax.axvline(self.Z_MAX, color='C3', lw=1.0, ls='--', alpha=0.8, label='ceil')
        ax.fill_betweenx([-3.5, 3.5], -0.1,       self.Z_MIN, color='C3', alpha=0.15)
        ax.fill_betweenx([-3.5, 3.5], self.Z_MAX,  4.1,       color='C3', alpha=0.15)

    def plot_phi_theta(self, ax: plt.Axes):
        lim = np.pi / 2 + 0.1
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
               xlabel='φ (rad)', ylabel='θ (rad)')
        for v in [-self.PHI_MAX, self.PHI_MAX]:
            ax.axvline(v, color='C3', lw=1.0, ls='--', alpha=0.8)
        for v in [-self.THETA_MAX, self.THETA_MAX]:
            ax.axhline(v, color='C3', lw=1.0, ls='--', alpha=0.8)

    def plot_px_py(self, ax: plt.Axes):
        ax.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6),
               xlabel='px (m)', ylabel='py (m)')
        for i, (cx, cy, cz, r) in enumerate(self.obstacles):
            circle = plt.Circle((cx, cy), r, color='C3', alpha=0.25, zorder=3)
            ax.add_patch(circle)
            ax.plot(cx, cy, 'x', color='C3', ms=6, zorder=4)

    def plot_phase(self, ax: plt.Axes, setup_idx: int = 0):
        self.phase2d_setups()[setup_idx].plot_fn(ax)
