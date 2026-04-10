import jax.numpy as jnp
from pncbf.dyn.dyn_types import Control


class Leader():
    # Internal dynamics state layout: [surge, yaw_rate]
    NX: int = 2
    NP: int = 3
    NU: int = 2

    SURGE, YAWRATE = range(NX)
    X, Y, THETA = range(NP)
    DES_SURGE, DES_YAWRATE = range(NU)

    # simplified closed loop reference dynamics matrices
    A = jnp.array([[-1.0, 0.0],
                   [0.0, -1.0]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 1.0]], dtype=jnp.float32)

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Leader control (returns a constant — safe anywhere)
    # ------------------------------------------------------------------
    @staticmethod
    def get_leader_control(mode: str) -> Control:
        if mode == "straight":
            return jnp.array([1.0, 0.0], dtype=jnp.float32)
        elif mode == "circle":
            return jnp.array([1.0, 0.05], dtype=jnp.float32)
        elif mode == "zero":
            return jnp.array([0.0, 0.0], dtype=jnp.float32)
        else:
            raise ValueError(f"{mode} is not a valid mode for Heron.get_leader_control(mode)")    

class Follower():
    # Internal dynamics state layout: [surge_integral_err, surge, yaw_rate_integral_err, yaw_rate]
    NX: int = 4
    NP: int = 3
    NU: int = 2

    SURGE_INTEGRAL_ERROR, SURGE, YAWRATE_INTEGRAL_ERROR, YAWRATE = range(NX)
    X, Y, THETA = range(NP)
    DES_SURGE, DES_YAWRATE = range(NU)

    # closed loop reference dynamics matrices
    Au = jnp.array([[0.0, 1.0], 
                    [-1.0, -24.048562]])
    Bu = jnp.array([[-1.0], 
                    [0.0]])
    Ar = jnp.array([[0.0, 1.0],
                    [-6.3246, -61.78262]])
    Br = jnp.array([[-1.0], 
                    [0.0]])

    A = jnp.block([[Au, jnp.zeros((2, 2))], 
                   [jnp.zeros((2, 2)), Ar]])
    B = jnp.block([[Bu, jnp.zeros((2, 1))], 
                   [jnp.zeros((2, 1)), Br]])

    def __init__(self) -> None:
        pass