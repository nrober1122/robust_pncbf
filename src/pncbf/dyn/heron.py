import jax.numpy as jnp
from pncbf.dyn.dyn_types import Control, State


class Heron():
    # Internal dynamics state layout: [surge_integral_err, surge, yaw_rate_integral_err, yaw_rate]
    NX: int = 4
    NP: int = 3   # position state: [px, py, heading]
    NU: int = 2

    SURGE_INTEGRAL_ERROR, SURGE, YAWRATE_INTEGRAL_ERROR, YAWRATE = range(NX)
    X, Y, HEADING = range(NP)
    DES_SURGE, DES_YAWRATE = range(NU)

    def __init__(self, initial_position: State, dt: float = 0.01) -> None:
        self._dt: float = dt

        # closed loop reference dynamics matrices
        Au = jnp.array([[0.0, 1.0], [-1.0, -24.048562]])
        Bu = jnp.array([[-1.0], [0.0]])
        Ar = jnp.array([[0.0, 1.0], [-6.3246, -61.78262]])
        Br = jnp.array([[-1.0], [0.0]])

        self.A = jnp.block([[Au, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), Ar]])
        self.B = jnp.block([[Bu, jnp.zeros((2, 1))], [jnp.zeros((2, 1)), Br]])

        # control limits
        self.u_min = jnp.array([0.0, -0.6])
        self.u_max = jnp.array([2.0, 0.6])

        # Initial conditions — used only to build the initial full state vector in Error.
        # After that, vehicle state lives entirely inside the Error full state vector.
        self.initial_x        = jnp.zeros(self.NX, dtype=jnp.float32)
        self.initial_position = initial_position

    # # ------------------------------------------------------------------
    # # Purely functional dynamics — safe inside any JAX trace
    # # ------------------------------------------------------------------
    # def xdot_x(self, x: State, control: Control) -> State:
    #     """Derivative of internal dynamics state."""
    #     control = control.clip(self.u_min, self.u_max)
    #     return self.A @ x + self.B @ control

    # @staticmethod
    # def xdot_position(position: State, x: State) -> State:
    #     """
    #     Derivative of position [px, py, heading].
    #     Driven by actual surge and yaw_rate read from internal state x,
    #     not the commanded control.
    #     """
    #     surge    = x[Heron.SURGE]    # index 1
    #     yaw_rate = x[Heron.YAWRATE]  # index 3
    #     heading  = position[2]
    #     return jnp.array([surge * jnp.sin(heading),
    #                        surge * jnp.cos(heading),
    #                        yaw_rate])

    # def xdot(self, x: State, position: State, control: Control) -> tuple[State, State]:
    #     """
    #     Returns (dx, dposition) — time derivatives of both sub-states.
    #     Purely functional. Safe inside jit / vmap / diffeqsolve.
    #     """
    #     dx        = self.xdot_x(x, control)
    #     dposition = self.xdot_position(position, x)
    #     return dx, dposition

    # ------------------------------------------------------------------
    # Leader control (returns a constant — safe anywhere)
    # ------------------------------------------------------------------
    @staticmethod
    def get_leader_control(mode: str) -> Control:
        if mode == "straight":
            return jnp.array([1.0, 0.0])
        elif mode == "circle":
            return jnp.array([1.0, -0.01])
        else:
            raise ValueError(f"{mode} is not a valid mode for Heron.get_leader_control(mode)")