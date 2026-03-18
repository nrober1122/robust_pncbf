import jax.numpy as jnp
import functools as ft
from pncbf.dyn.odeint import rk4
from pncbf.dyn.task import Task
from pncbf.dyn.dyn_types import Control, State

class Heron(Task):
    NX: int = 4
    NU: int = 2

    SURGE, SURGE_INTEGRAL_ERROR, YAWRATE, YAWRATE_INTEGRAL_ERROR = range(NX)
    DES_SURGE, DES_YAWRATE = range(NU)

    def __init__(self, initial_position, dt: float = 0.01) -> None:
        self.dt: float = dt

        self.position = initial_position # x, y, heading
        self.surge_state = jnp.zeros((2, 1))
        self.yaw_rate_state = jnp.zeros((2, 1))

        # closed loop reference dynamics
        self.Au = jnp.array([[0.0, 1.0], [-1.0, -24.048562]])
        self.Bu = jnp.array([[-1.0], [0.0]])
        self.Ar = jnp.array([[0.0, 1.0], [-6.3246, -61.78262]])
        self.Br = jnp.array([[-1.0], [0.0]])

        # stacked dynamics for single system
        self.A = jnp.block([[self.Au, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), self.Ar]])
        self.B = jnp.block([[self.Bu, jnp.zeros((2, 1))], [jnp.zeros((2, 1)), self.Br]]) 
        self.x = jnp.vstack((self.surge_state, self.yaw_rate_state))  

        # control limits
        self.u_min = jnp.array([0, -0.6])
        self.u_max = jnp.array([2, 0.6]) 

    def f(self, state: State) -> State:
        self.chk_x(state)

        Ax: State = self.A @ state
        return Ax
    
    def G(self, state: State) -> State:
        self.chk_x(state)

        G: State = self.B
        return G

    def xdot(self, state: State, control: Control) -> State:
        self.chk_x(self.x)
        self.chk_u(control)
        control = control.clip(self.u_min, self.u_max)
        f, G = self.f(state), self.G(state)
        self.chk_x(f)
        Gu: State = G @ control
        self.chk_x(Gu)
        dx: State = f + Gu
        return self.chk_x(dx)

    def step(self, control: Control) -> None:
        xdot_with_u = ft.partial(self.xdot, control=control)
        x_new: State = rk4(self.dt, xdot_with_u, self.x)

        surge: float = self.surge_state[1][0]
        yaw_rate: float = self.yaw_rate_state[2][0]
        heading: float = self.position[2][0]
        position_dot = jnp.array([[surge*jnp.sin(heading)], [surge*jnp.cos(heading)], [yaw_rate]])
        new_position = rk4(self.dt, position_dot, self.position)

        # save new states
        self.x = x_new
        self.position = new_position

    def get_leader_control(self, mode: str) -> Control:
        if mode == "straight":
            return jnp.array([1.0, 0.0])
        else:
            raise ValueError(f"{mode} is not a valid mode for Heron.get_leader_control(mode)")



