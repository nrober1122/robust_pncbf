import numpy as np
from pncbf.dyn.odeint import rk4
from typing import Tuple

class Heron:
    def __init__(self, initial_position: np.ndarray[float], dt = 0.01) -> None:
        self.NX = 3
        self.NU = 2
        self.dt = dt

        self.position = initial_position # x, y, heading
        self.surge_state = np.zeros((2, 1))
        self.yaw_rate_state = np.zeros((2, 1))

        # closed loop reference dynamics
        self.Au: np.ndarray[float] = np.array([[0.0, 1.0], [-1.0, -24.048562]])
        self.Bu: np.ndarray[float]= np.array([[-1.0], [0.0]])
        self.Ar: np.ndarray[float] = np.array([[0.0, 1.0], [-6.3246, -61.78262]])
        self.Br: np.ndarray[float]= np.array([[-1.0], [0.0]])

    def step(self, control: np.ndarray[float]) -> None:
        x_dot_u: np.ndarray[float] = self.Au @ self.surge_state + self.Bu * control[0]
        x_dot_r: np.ndarray[float] = self.Ar @ self.yaw_rate_state + self.Br * control[1]

        x_new_u = rk4(self.dt, x_dot_u, self.surge_state)
        x_new_r = rk4(self.dt, x_dot_r, self.yaw_rate_state)

        surge: float = self.surge_state[1][0]
        yaw_rate: float = self.yaw_rate_state[1][0]
        heading: float = self.position[2][0]
        position_dot: np.ndarray[float] = np.array([[surge*np.sin(heading)], [surge*np.cos(heading)], [yaw_rate]])
        new_position = rk4(self.dt, position_dot, self.position)

        # save new states
        self.surge_state = x_new_u
        self.yaw_rate_state = x_new_r
        self.position = new_position

    def get_leader_control(self, mode: str) -> np.ndarray[float]:
        if mode == "straight":
            return np.array([1.0, 0.0])
        else:
            raise ValueError(f"{mode} is not a valid mode for Heron.get_leader_control(mode)")



