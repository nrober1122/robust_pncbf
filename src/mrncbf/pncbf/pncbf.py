from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger

import mrncbf.utils.typed_ft as ft
from mrncbf.dyn.dyn_types import BBControl, BBHFloat, BControl, BHFloat, BState, BTControl, BTState, HFloat, State
from mrncbf.dyn.sim_cts import SimCtsReal
from mrncbf.dyn.task import Task
from mrncbf.networks.block import TmpNet
from mrncbf.networks.ensemble import Ensemble, subsample_ensemble
from mrncbf.networks.mlp import MLP
from mrncbf.networks.temporal_residual import ResidualCorrection
from mrncbf.networks.ncbf import MultiNormValueFn, MultiValueFn, Rescale
from mrncbf.networks.network_utils import HidSizes, get_act_from_str
from mrncbf.networks.optim import get_default_tx
from mrncbf.networks.train_state import TrainState
from mrncbf.pncbf.compute_disc_avoid import AllDiscAvoidTerms, compute_all_disc_avoid_terms
from mrncbf.qp.min_norm_cbf import (
    min_norm_cbf, min_norm_cbf_qp_mats,
    rcbf, rcbf_qp_mats,
    rcbf_qp_linear, rcbf_qp_linear_mats,
    compute_rcbf_gammas,
    rcbf_qp_learned, rcbf_qp_learned_mats,
)
from mrncbf.utils.grad_utils import compute_norm, empty_grad_tx
from mrncbf.utils.jax_types import BBFloat, BBool, BFloat, BHBool, BTHBool, FloatScalar, MetricsDict
from mrncbf.utils.jax_utils import jax_vmap, rep_vmap, tree_copy
from mrncbf.utils.loss_utils import weighted_sum_dict
from mrncbf.utils.none import get_or
from mrncbf.utils.rng import PRNGKey
from mrncbf.utils.schedules import Schedule, as_schedule

def _resolve_alpha(h_V, alpha_safe, alpha_unsafe):
    """Returns per-constraint alpha matching the existing sloped logic."""
    if isinstance(alpha_safe, float) or jnp.asarray(alpha_safe).ndim == 0:
        return jnp.where(jnp.all(h_V < 0), alpha_safe, alpha_unsafe)
    else:
        return jnp.where(h_V < 0,
                         jnp.asarray(alpha_safe),
                         jnp.asarray(alpha_unsafe))   # (nh,)

# def _resolve_alpha_local(h_V, alpha_safe, alpha_unsafe):
#         """Same logic as _resolve_alpha in the main codebase."""
#         if isinstance(alpha_safe, float) or jnp.asarray(alpha_safe).ndim == 0:
#             return jnp.where(jnp.all(h_V < 0), alpha_safe, alpha_unsafe)
#         else:
#             return jnp.where(h_V < 0,
#                             jnp.asarray(alpha_safe),
#                             jnp.asarray(alpha_unsafe))

@define
class PNCBFTrainCfg:
    # How many trajectories to collect
    collect_size: int
    # What should the spacing be in dt in the trajectory we return?
    # This is different from the dt used in the solver.
    rollout_dt: float
    # How many of the rollout_dt we use.
    rollout_T: int

    # Batch size during training.
    batch_size: int
    # 0: No discount. Infinity: V = h.
    lam: Schedule

    # Target V smoothing. Small values = more smoothing.
    tau: float

    # Instead of discount, we can just "discount" the target. This is biased but so is discount.
    tgt_rhs: Schedule


@define
class PNCBFEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class PNCBFCfg:
    act: str
    lr: Schedule
    wd: Schedule

    hids: HidSizes
    train_cfg: PNCBFTrainCfg
    eval_cfg: PNCBFEvalCfg

    # Num in ensemble.
    n_Vs: int
    # Num to use as target.
    n_min_tgt: int

    @property
    def alg_name(self) -> str:
        return "PNCBF"


class PNCBF(struct.PyTreeNode):
    collect_idx: int
    update_idx: int
    key: PRNGKey
    Vh: TrainState[HFloat]
    Vh_tgt: TrainState[HFloat]

    nom_pol: Callable = struct.field(pytree_node=False)
    cfg: PNCBFCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    _lam: FloatScalar
    lam_sched: optax.Schedule = struct.field(pytree_node=False)
    tgt_rhs_sched: optax.Schedule = struct.field(pytree_node=False)

    class CollectData(NamedTuple):
        """Data collected here has a spacing of train_cfg.rollout_dt."""

        bT_x: BTState
        bT_u: BTControl
        # b_vterms: DiscAvoidTerms
        b_vterms: AllDiscAvoidTerms
        # If true, then V = h. We can enforce derivative conditions on this.
        # This can be modified depending on the value of V(x_T).
        bTh_iseqh: BTHBool

    class Batch(NamedTuple):
        b_x0: BState
        b_u0: BTControl
        b_xT: BState
        bh_iseqh: BBool
        bh_lhs: BHFloat
        bh_int_rhs: BHFloat
        b_discount_rhs: BFloat

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bbh_V: BBHFloat
        bbh_Vdot: BBHFloat
        bbh_Vdot_disc: BBHFloat
        # bbh_hmV: BBHFloat
        # bbh_compl: BBHFloat
        # bbh_eq_h: BBHBool
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: PNCBFCfg, nom_pol: Callable) -> "PNCBF":
        key, key_Vh = jr.split(jr.PRNGKey(seed), 2)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        # Define V network.
        Vh_cls = ft.partial(MLP, cfg.hids, act)
        Vh_cls = ft.partial(MultiValueFn, Vh_cls, task.nh)
        Vh_def = Ensemble(Vh_cls, cfg.n_Vs)
        Vh_tx = get_default_tx(cfg.lr.make(), cfg.wd.make())
        Vh = TrainState.create_from_def(key_Vh, Vh_def, (pol_obs,), Vh_tx)

        # Target network for stability.
        Vh_tgt_def = Ensemble(Vh_cls, num=cfg.n_min_tgt)
        Vh_params_copy = tree_copy(Vh.params)  # Copy so that we can donate buffer.
        Vh_tgt = TrainState.create(apply_fn=Vh_tgt_def.apply, params=Vh_params_copy, tx=empty_grad_tx())

        lam_sched = as_schedule(cfg.train_cfg.lam)
        logger.info("Discount lam: {}".format(lam_sched))

        col_idx, upd_idx = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
        lam = jnp.array(0.0)
        return PNCBF(
            col_idx, upd_idx, key, Vh, Vh_tgt, nom_pol, cfg, task, lam, lam_sched.make(), cfg.train_cfg.tgt_rhs.make()
        )

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def lam(self) -> FloatScalar:
        # return self.lam_sched(self.update_idx)
        return self._lam

    @property
    def V_shift(self):
        return -(1 - np.exp(-self.lam * self.task.max_ttc)) * self.task.h_min

    def update_lam(self):
        new_lam = self.lam_sched(self.update_idx)
        return self.replace(_lam=new_lam), new_lam != self._lam

    @property
    def tgt_rhs_coeff(self) -> FloatScalar:
        return self.tgt_rhs_sched(self.update_idx)

    def get_V(self, state: State, params=None):
        return self.get_Vh(state, params).max(-1)

    def get_eh_Vh(self, state: State, params=None):
        params = get_or(params, self.Vh.params)
        V_obs, _ = self.task.get_obs(state)
        eh_Vh = self.Vh.apply_fn(params, V_obs)
        assert eh_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        return eh_Vh

    def get_Vh(self, state: State, params=None):
        eh_Vh = self.get_eh_Vh(state, params)
        h_Vh = eh_Vh.mean(axis=0)
        assert h_Vh.shape == (self.task.nh,)
        return h_Vh

    def get_Vh_tgt(self, state: State):
        return self.get_Vh(state, self.Vh_tgt.params)

    def get_eVh_tgt(self, state: State):
        V_obs, _ = self.task.get_obs(state)
        eh_Vh_tgt = self.Vh.apply_fn(self.Vh_tgt.params, V_obs)
        assert eh_Vh_tgt.shape == (self.cfg.n_Vs, self.task.nh)
        return eh_Vh_tgt

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_dset(self):
        key = jr.fold_in(self.key, self.collect_idx)

        # Sample initial states.
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.collect_size)

        rollout_dt = self.task.dt
        tf = rollout_dt * (self.train_cfg.rollout_T + 0.001)
        # sim = SimCtsReal(self.task, self.nom_pol, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
        # bT_x, _, _ = jax_vmap(sim.rollout_plot)(b_x0)

        if self.task.has_episode_pol():
            def rollout_one(x0, key):
                pol = self.task.make_episode_pol(key, self.nom_pol)
                sim = SimCtsReal(self.task, pol, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
                return sim.rollout_plot(x0)
            b_keys = jr.split(key, self.train_cfg.collect_size)
            bT_x, _, _, _, _ = jax.vmap(rollout_one)(b_x0, b_keys)
        else:
            sim = SimCtsReal(self.task, self.nom_pol, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
            bT_x, _, _, _, _ = jax_vmap(sim.rollout_plot)(b_x0)
        assert bT_x.shape == (self.train_cfg.collect_size, self.train_cfg.rollout_T + 1, self.task.nx)

        # Compute nominal control (again) here, especially if it's expensive (QP).
        bT_u = rep_vmap(self.nom_pol, rep=2)(bT_x)

        bTh_h = rep_vmap(self.task.h_components, rep=2)(bT_x)

        # Compute value function terms for each trajectory.
        b_vterms: AllDiscAvoidTerms = jax_vmap(ft.partial(compute_all_disc_avoid_terms, self.lam, rollout_dt))(bTh_h)
        bTh_iseqh = None

        return self.replace(collect_idx=self.collect_idx + 1), self.CollectData(bT_x, bT_u, b_vterms, bTh_iseqh)

    @jax.jit
    def get_vterms(self, bT_x: BTState) -> AllDiscAvoidTerms:
        bTh_h = rep_vmap(self.task.h_components, rep=2)(bT_x)
        return jax_vmap(ft.partial(compute_all_disc_avoid_terms, self.lam, self.task.dt))(bTh_h)

    def compute_loss(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat, params):
        eh_V_apply = ft.partial(self.get_eh_Vh, params=params)
        beh_V_pred = jax_vmap(eh_V_apply)(b_x)

        # 1: Value function loss. This applies to all states in the trajectory.
        eh_loss_Vh = jnp.mean((bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        assert eh_loss_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        loss_Vh = jnp.mean(eh_loss_Vh)

        info_dict = {}

        loss_dict = {
            "Loss/Vh_mse": loss_Vh,
        }
        loss = weighted_sum_dict(loss_dict, loss_weights)
        return loss, loss_dict | info_dict

    def update_Vh(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat):
        loss_fn = ft.partial(self.compute_loss, loss_weights, b_x, b_u, bh_iseqh, bh_V_tgt)
        grads, info = jax.grad(loss_fn, has_aux=True)(self.Vh.params)
        info["V_grad"] = compute_norm(grads)
        Vh_new = self.Vh.apply_gradients(grads)

        return self.replace(Vh=Vh_new), info

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, data: Batch, loss_weights: MetricsDict):
        b_size = len(data.b_xT)
        b_Vobs_T, _ = jax_vmap(self.task.get_obs)(data.b_xT)

        # 1: Compute value function at all states for each trajectory.
        # 1.1: Eval V tgt at xT.
        # bh_VT = self.get_Vh(b_xT)
        key_update = jr.fold_in(self.key, self.update_idx)
        key_tgt = key_update
        target_params = subsample_ensemble(key_tgt, self.Vh_tgt.params, self.cfg.n_min_tgt, self.cfg.n_min_tgt)
        ebh_VhT = self.Vh_tgt.apply_fn(target_params, b_Vobs_T)
        # bh_VhT = jnp.mean(ebh_VhT, axis=0)
        bh_VhT = jnp.min(ebh_VhT, axis=0)
        # bh_VhT = jnp.full((b_size, self.task.nh), -1e12)

        # Speed up tgt 1: clip it between hmin and hmax.
        bh_VhT = jnp.clip(bh_VhT, self.task.h_min, self.task.h_max)
        # Speed up tgt 2: If its smaller than h, then make it h.
        bh_tgt = jax_vmap(self.task.h_components)(data.b_xT)
        bh_VhT = jnp.maximum(bh_VhT, bh_tgt)

        # 1.2: Compute (target) V at T_x using b_vterms
        bh_rhs = data.bh_int_rhs + data.b_discount_rhs[:, None] * bh_VhT
        bh_V_tgt = data.bh_lhs + self.tgt_rhs_coeff * jnp.maximum(0, bh_rhs - data.bh_lhs)
        assert bh_V_tgt.shape == (b_size, self.task.nh)

        new_self, info_mean = self.update_Vh(loss_weights, data.b_x0, data.b_u0, data.bh_iseqh, bh_V_tgt)

        info_mean["anneal/lam"] = self.lam
        gamma = jnp.exp(-self.lam * self.task.dt)
        info_mean["anneal/eff_horizon"] = 1 / (1 - gamma)
        info_mean["anneal/tgt_rhs_coeff"] = self.tgt_rhs_coeff
        info_mean["collect_idx"] = self.collect_idx
        info_mean["update_idx"] = self.update_idx

        Vh_tgt_params = optax.incremental_update(new_self.Vh.params, self.Vh_tgt.params, self.train_cfg.tau)
        Vh_tgt = self.Vh_tgt.replace(params=Vh_tgt_params)

        return new_self.replace(Vh_tgt=Vh_tgt, update_idx=self.update_idx + 1), info_mean
    
    def _get_cbf_ingredients(self, state: State, V_shift: float, nom_pol):
        """Compute all ingredients shared across filter variants."""
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        nom_pol  = get_or(nom_pol, self.nom_pol)

        h_V   = Vh_apply(state) + V_shift          # (nh,)
        hx_Vx = jax.jacobian(Vh_apply)(state)      # (nh, nx)
        f     = self.task.f(state)                 # (nx,)
        G     = self.task.G(state)                 # (nx, nu)
        u_nom = nom_pol(state)                     # (nu,)
        return h_V, hx_Vx, f, G, u_nom

    def get_cbf_qpmats(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        hx_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        penalty, relax_eps1, relax_eps2 = 10.0, 5e-1, 20.0
        qp = min_norm_cbf_qp_mats(alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
        return qp

    def get_cbf_control_sloped_all(
        self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None
    ):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)

        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        # u, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        u, r, sol = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom, penalty=10.0, relax_eps1=5e-1, relax_eps2=0.1)
        return self.task.chk_u(u), (r, sol)

    def get_cbf_control_sloped(
        self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None
    ):
        return self.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state, V_shift, nom_pol)[0]

    def get_cbf_control(self, alpha: float, state: State):
        return self.get_cbf_control_sloped_all(alpha, alpha, state)[0]
    
    # ── r-cbf ─────────────────────────────────────────────────────────────────────
    def get_rcbf_control_all(
        self,
        alpha_safe: float,
        alpha_unsafe: float,
        state: State,
        V_shift: float = 1e-3,
        rho_scale: float = 0.25,
        nom_pol=None,
    ):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        h_V, hx_Vx, f, G, u_nom = self._get_cbf_ingredients(state, V_shift, nom_pol)
        alpha = _resolve_alpha(h_V, alpha_safe, alpha_unsafe)

        u_opt, r, sol = rcbf(
            alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
            rho_scale=rho_scale,
        )
        return self.task.chk_u(u_opt), (r, sol)

    def get_rcbf_control(self, alpha_safe, alpha_unsafe, state, V_shift=1e-3,
                        rho_scale=0.25, nom_pol=None):
        return self.get_rcbf_control_all(
            alpha_safe, alpha_unsafe, state, V_shift, rho_scale, nom_pol
        )[0]

    # ── r-cbf-qp ──────────────────────────────────────────────────────────────────
    def get_rcbf_qp_control_all(
        self,
        alpha_safe: float,
        alpha_unsafe: float,
        state: State,
        nnv_filter = None,              # Box from nnv_state_bounds
        epsilon: float = 1e-4,             # Regularization for QP
        V_shift: float = 1e-3,
        n_samples: int = 100,
        rng_key = None,
        nom_pol=None,
    ):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        h_V, hx_Vx, f, G, u_nom = self._get_cbf_ingredients(state, V_shift, nom_pol)
        alpha = _resolve_alpha(h_V, alpha_safe, alpha_unsafe)
        nom_pol = get_or(nom_pol, self.nom_pol)
        
        if nnv_filter is None:
            state_bounds = type('obj', (object,), {
                'lo': state - epsilon * np.ones(state.shape),
                'hi': state + epsilon * np.ones(state.shape)
            })()
        else:
            state_bounds = nnv_filter.state_bounds(epsilon=epsilon)
        

        # Sample states within bounds to estimate max control deviation.
        rng_key = get_or(rng_key, jax.random.PRNGKey(0))
        x_samples = jax.random.uniform(
            rng_key,
            shape=(n_samples, self.task.nx),
            minval=state_bounds.lo,
            maxval=state_bounds.hi,
        )
        u_hat = nom_pol(state)
        u_samples = jax.vmap(nom_pol)(x_samples)
        sigma_hat = jnp.max(jnp.linalg.norm(u_samples - u_hat, axis=-1)).astype(jnp.float32)

        gamma1, gamma2 = compute_rcbf_gammas(sigma_hat)

        u_opt, r, sol = rcbf_qp_linear(
            alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
            gamma1=gamma1, gamma2=gamma2,
            relax_eps2=0.1,
        )
        return self.task.chk_u(u_opt), (r, sol)

    def get_rcbf_qp_control(self, alpha_safe, alpha_unsafe, state, nnv_filter=None, epsilon=1e-4,
                            V_shift=1e-3, num_samples=1000, nom_pol=None):
        return self.get_rcbf_qp_control_all(
            alpha_safe, alpha_unsafe, state, nnv_filter, epsilon, V_shift, num_samples, nom_pol
        )[0]
    
    # ── mrncbf ────────────────────────────────────────────────────────────────────
    def get_learned_rcbf_control_all(
        self,                          # PNCBF instance
        alpha_safe: float,
        alpha_unsafe: float,
        state: State,                  # x̂ (the noisy estimate)
        Delta_net: ResidualCorrection,
        Delta_params,
        epsilon: float,
        history_seq: Optional[jnp.ndarray],  # (window_len, feat_dim) or None
        gamma1: float | jnp.ndarray,
        gamma2: float | jnp.ndarray,
        R_floor: float = 0.0,
        V_shift: float = 1e-3,
        nom_pol=None,
    ):
        """
        CBF-QP with analytical + learned robustification.
    
        Mirrors get_rcbf_qp_control_all but adds Δ_θ to the constraint.
        """
        from mrncbf.utils.none import get_or
    
        u_lb, u_ub = self.task.u_min, self.task.u_max
        h_V, hx_Vx, f, G, u_nom = self._get_cbf_ingredients(state, V_shift, nom_pol)
    
        # Resolve alpha (safe vs unsafe region)
        alpha = _resolve_alpha_local(h_V, alpha_safe, alpha_unsafe)
    
        # Per-constraint Lie derivative norms (features for Δ_θ)
        h_LG_V = hx_Vx @ G                          # (nh, nu)
        h_Lgh_norm = jnp.linalg.norm(h_LG_V, axis=-1)  # (nh,)
    
        # Evaluate learned residual
        h_Delta = Delta_net.apply(
            Delta_params,
            state,          # x̂
            h_Lgh_norm,     # (nh,)
            h_V,            # (nh,)
            epsilon,        # scalar
            history_seq,    # (window_len, feat_dim) or None
        )
    
        u_opt, r, sol = rcbf_qp_learned(
            alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom,
            gamma1, gamma2, h_Delta, R_floor,
            relax_eps2=0.1,
        )
        return self.task.chk_u(u_opt), (r, sol, h_Delta)
    

    @ft.partial(jax.jit, static_argnames=["T", "setup_idx", "use_pid"])
    def get_bb_V_nom(self, T: int = None, setup_idx: int = 0, use_pid: bool = False):
        return self.V_for_pol(self.nom_pol, T, setup_idx, use_pid)

    # @ft.partial(jax.jit, static_argnames=["T", "setup_idx"])
    # def get_bb_V_opt(self, T: int = None, setup_idx: int = 0):
    #     return self.V_for_pol(self.get_opt_u, T, setup_idx)

    @ft.partial(jax.jit, static_argnames=["T", "setup_idx"])
    def get_bb_V_sloped(self, T: int = None, setup_idx: int = 0):
        nom_pol = ft.partial(self.get_cbf_control_sloped, 2.0, 100.0)
        return self.V_for_pol(nom_pol, T, setup_idx, use_pid=False)

    def V_for_pol(self, pol, T: int = None, setup_idx: int = 0, use_pid: bool = True):
        T = get_or(T, self.cfg.eval_cfg.eval_rollout_T)
        tf = self.task.dt * (T + 0.001)
        sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False, use_pid=use_pid, max_steps=T + 3)
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0(setup_idx)
        bbT_x, _, _, _, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x)
        bbT_h = rep_vmap(self.task.h, rep=3)(bbT_x)
        return bbT_h.max(-1)

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0(0)
        b_x0_metric = self.task.get_metric_x0()
        # b_x0_loss = self.task.get_loss_x0()

        # Rollout using the min norm controller.
        alpha_safe, alpha_unsafe = 2.0, 100.0
        pol = ft.partial(self.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
        # sim = SimNCLF(self.task, pol, self.cfg.eval_cfg.eval_rollout_T)
        eval_rollout_T = self.cfg.eval_cfg.eval_rollout_T
        tf = self.task.dt * eval_rollout_T

        # sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False)
        # bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        # bT_x_metric, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        # Don't use PID stepsize controller, since the QP controller is probably super nonsmooth.
        sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False, use_pid=False, max_steps=eval_rollout_T + 1)
        bT_x_plot, _, _, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _, _, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        def get_V_info(state):
            Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
            h_V = Vh_apply(state)
            h_Vx = jax.jacobian(Vh_apply)(state)
            f, G = self.task.f(state), self.task.G(state)
            h_h = self.task.h_components(state)

            u_nom = self.nom_pol(state)
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
            # u, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            u, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            xdot = f + jnp.sum(G * u, axis=-1)
            h_Vdot = jnp.sum(h_Vx * xdot, axis=-1)

            h_Vdot_disc = h_Vdot - self.lam * (h_V - h_h)

            return h_V, h_Vdot, h_Vdot_disc, u

        # def get_loss_info(state):
        #     Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        #     h_V = Vh_apply(state)
        #     h_Vx = jax.jacobian(Vh_apply)(state)
        #     h_h = self.task.h_components(state)
        #
        #     # Check descent condition along pi.
        #     f, G = self.task.f(state), self.task.G(state)
        #     u_nom = self.nom_pol(state)
        #     # is_safe = jnp.all(h_V < 0)
        #     # alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        #     # u_qp, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #     # u_qp, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #
        #     xdot_nom = f + jnp.sum(G * u_nom, axis=-1)
        #     # xdot_qp = f + jnp.sum(G * u_qp, axis=-1)
        #
        #     h_Vdot_nom = jnp.sum(h_Vx * xdot_nom, axis=-1)
        #     # h_Vdot_qp = jnp.sum(h_Vx * xdot_qp, axis=-1)
        #
        #     h_Vdot_disc_nom = h_Vdot_nom - self.lam * (h_V - h_h)
        #
        #     # h_h > 0  =>  h_V > 0
        #     # equiv as  ~( h_h > 0 ) or (h_V > 0)
        #     h_clsfy = (h_h <= 0) | (h_V > 0)
        #
        #     return h_clsfy, h_V - h_h, h_Vdot_nom, h_Vdot_disc_nom

        u_lb, u_ub = self.task.u_min, self.task.u_max
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        # bh_clsfy, bh_hmV, bh_Vdot_nom, bh_Vdot_disc_nom = jax_vmap(get_loss_info)(b_x0_loss)
        # h_desc_nom = jnp.mean(bh_Vdot_nom <= 0, axis=0)
        # h_desc_disc_nom = jnp.mean(bh_Vdot_disc_nom <= 0, axis=0)
        # # h_desc_qp = jnp.mean(bh_Vdot_qp <= 0, axis=0)
        # desc_nom_max = (bh_Vdot_nom.max(axis=1) <= 0).mean()
        # desc_nom_disc_max = (bh_Vdot_disc_nom.max(axis=1) <= 0).mean()
        # # desc_qp_max = (bh_Vdot_qp.max(axis=1) <= 0).mean()
        # h_hmV_neg = jnp.mean(bh_hmV <= 0, axis=0)
        # hmV_neg_max = jnp.mean(bh_hmV.max(axis=1) <= 0, axis=0)
        #
        # h_clsfy = bh_clsfy.mean(axis=0)
        # h_clsfy_max = jnp.all(bh_clsfy, axis=1).mean()

        loss_info = {}
        # for ii, h_label in enumerate(self.task.h_labels_clean):
        #     loss_info[f"Descent/Nom/{h_label}"] = h_desc_nom[ii]
        #     loss_info[f"Descent/Nom Disc/{h_label}"] = h_desc_disc_nom[ii]
        #     # loss_info[f"Descent/QP/{h_label}"] = h_desc_qp[ii]
        #     loss_info[f"ClassifyDet/{h_label}"] = h_hmV_neg[ii]
        #     loss_info[f"Classify/{h_label}"] = h_clsfy[ii]
        # loss_info[f"Descent/Nom/Max"] = desc_nom_max
        # loss_info[f"Descent/Nom Disc/Max"] = desc_nom_disc_max
        # loss_info[f"Descent/QP/Max"] = desc_qp_max
        # loss_info[f"ClassifyDet/Max"] = hmV_neg_max
        # loss_info[f"Classify/Max"] = h_clsfy_max

        # bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)
        bTh_hs = rep_vmap(self.task.h_components, rep=2)(bT_x_metric)
        bT_hs = bTh_hs.max(axis=2)

        h_mean = bT_hs.mean(-1).mean(0)
        h_max = bT_hs.max(-1).mean(0)

        h_labels = self.task.h_labels_clean
        safe_fracs = {
            "Safe/{}".format(h_labels[ii]): jnp.all(bTh_hs[:, :, ii] < 0, axis=1).mean() for ii in range(self.task.nh)
        }

        eval_info = {
            "Constr Mean": h_mean,
            "Constr Max Mean": h_max,
            "Safe Frac": jnp.mean(jnp.all(bT_hs < 0, axis=-1)),
            **safe_fracs,
            **loss_info,
        }

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u, eval_info)
