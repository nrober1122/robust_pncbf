import functools as ft
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from diffrax import (
    Bosh3,
    ConstantStepSize,
    DirectAdjoint,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from loguru import logger

from mrncbf.dyn.dyn_types import State
from mrncbf.dyn.odeint import tsit5, tsit5_dense
from mrncbf.dyn.task import Task
from mrncbf.utils.jax_utils import concat_at_end, jax_vmap, merge01
from mrncbf.utils.none import get_or


def get_solver(solver: str):
    if solver == "bosh3":
        return Bosh3()
    if solver == "tsit5":
        return Tsit5()

    raise NotImplementedError("")


def sim_step(task: Task, policy, dt: float, x0: State, solver: str = "tsit5"):
    def body(t, state, args):
        control = policy(state)
        return task.xdot(state, control)

    term = ODETerm(body)
    solver = get_solver(solver)
    saveat = SaveAt(t1=True)
    stepsize_controller = ConstantStepSize()
    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=dt,
        dt0=dt / 2,
        y0=x0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=4,
    )
    assert solution.ys.shape == (1, task.nx)
    return solution.ys[0]


class SimCtsReal:
    def __init__(
        self,
        task: Task,
        policy,
        tf: float,
        result_dt: float,
        dt0: float = None,
        use_obs: bool = False,
        max_steps: int = 256,
        use_pid: bool = True,
        solver: str = "tsit5",
    ):
        self.task = task
        self.policy = policy
        self.result_dt = result_dt
        self.tf = tf
        self.use_obs = use_obs
        self.max_steps = max_steps
        self.use_pid = use_pid
        self.dt0 = get_or(dt0, result_dt)
        self.solver = solver

    def get_control(self, state):
        if self.use_obs:
            Vobs, polobs = self.task.get_obs(state)
            return self.policy(polobs)

        return self.policy(state)

    def rollout_plot(self, x0: State, noise_scale: float = 0.0, rng_key=None):
        nx = self.task.nx
        use_noise = noise_scale > 0.0

        def body(t, state, args):
            # x_true = state[:nx]
            
            if use_noise:
                key = jax.random.fold_in(rng_key, jnp.int32(t * 1e4))
                # noise = jax.random.normal(key, shape=(nx,)) * noise_scale
                noise = -jnp.ones(nx) * noise_scale
                control = self.get_control(state + noise)
            else:
                control = self.get_control(state)
            
            return self.task.xdot(state, control)

        term = ODETerm(body)
        solver = get_solver(self.solver)

        T = int(round(self.tf / self.result_dt))
        T_ts = np.linspace(0, self.tf, num=T + 1)
        saveat = SaveAt(ts=T_ts)
        # saveat = SaveAt(dense=True)
        # adjoint = DirectAdjoint()
        adjoint = RecursiveCheckpointAdjoint()
        if self.use_pid:
            stepsize_controller = PIDController(pcoeff=0.1, icoeff=0.4, rtol=1e-5, atol=1e-5)
        else:
            stepsize_controller = ConstantStepSize()
        solution = diffeqsolve(
            term,
            solver,
            t0=0,
            t1=self.tf,
            dt0=self.dt0,
            y0=x0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=self.max_steps,
        )

        if use_noise:
            # re-derive noisy observations at each saved timestep for plotting
            # since x_noisy = x_true + noise, we just re-sample at the saved ts
            keys = jax.vmap(lambda t: jax.random.fold_in(rng_key, jnp.int32(t * 1e4)))(solution.ts)
            # noise = jax.vmap(lambda k: jax.random.normal(k, shape=(nx,)) * noise_scale)(keys)
            noise = -jnp.ones((len(solution.ts), nx)) * noise_scale
            T_states_noisy = solution.ys + noise
            T_controls = jax.vmap(lambda s, n: self.get_control(s + n))(solution.ys, noise)
            return solution.ys, solution.ts, solution.stats, T_states_noisy, T_controls

        T_states = solution.ys
        T_controls = jax.vmap(self.get_control)(solution.ys)
        return solution.ys, solution.ts, solution.stats, None, T_controls