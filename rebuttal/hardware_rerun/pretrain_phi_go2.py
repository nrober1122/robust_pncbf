"""Supervised pretraining of the Go2 NMR-CBF network, extracted from cell 18 of
robust_pncbf-go2/scripts/dbint/dbint2d_cbf_compare.ipynb (with proj_grad_max_box from cell 5).

All task parameters (R_OBS, UMAX, ALPHA_HOCBF, ALPHA_QP, EPS_MAX) come from that repo's
scripts/dbint/dbint2d_quad.py, whose QUADRUPED_PARAMS block is active. Pass --eps-max to
override EPS_MAX (px, vx, py, vy) without editing it.

    uv run python rebuttal/hardware_rerun/pretrain_phi_go2.py <go2_repo> <out.pkl> [--eps-max 0.5 0.25 0.5 0.25]

Fine-tuning afterwards:
    cd <go2_repo> && python scripts/dbint/finetune_phi_dbint2d.py --name <name> --init-ckpt <out.pkl>
(finetune_phi_dbint2d.py reads EPS_MAX from dbint2d_quad.py, so it must match.)
"""

import argparse
import functools as ft
import pickle
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("go2_repo", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--eps-max", type=float, nargs=4, default=None)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--phi-max", type=float, default=8.0)
args = p.parse_args()

sys.path.insert(0, str(args.go2_repo / "scripts" / "dbint"))
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import dbint2d_quad as Q
from dbint2d_quad import ALPHA_QP, R_OBS, B_hocbf, f, phi_net

jax.config.update("jax_enable_x64", False)
EPS_MAX = jnp.array(args.eps_max) if args.eps_max else Q.EPS_MAX
print(f"R_OBS {R_OBS}  UMAX {Q.UMAX}  ALPHA_HOCBF {Q.ALPHA_HOCBF}  ALPHA_QP {ALPHA_QP}  EPS_MAX {EPS_MAX}")
PHI_MAX = args.phi_max  # cap labels against outliers near the obstacle boundary (notebook: 8.0)


def proj_grad_max_box(scalar_fn, lo, hi, n_restarts, n_steps, lr, key):
    x0s = lo[None] + jr.uniform(key, (n_restarts, lo.shape[0])) * (hi - lo)

    def run_one(x0):
        def step(x, _):
            g = jax.grad(scalar_fn)(x)
            return jnp.clip(x + lr * g, lo, hi), None
        x_star, _ = jax.lax.scan(step, x0, None, length=n_steps)
        return scalar_fn(x_star), x_star

    vals, xs = jax.vmap(run_one)(x0s)
    best = jnp.argmax(vals)
    return vals[best], xs[best]


@ft.partial(jax.jit, static_argnames=("n_restarts", "n_steps"))
def phi_true_fn(xhat, eps, key, n_restarts=8, n_steps=60, lr=0.05):
    """phi(xhat, eps) = max_{x in [xhat-eps, xhat+eps]} (Lf psi1 + alpha psi1)(x) - (...)(xhat), clipped >= 0."""
    lo, hi = xhat - eps, xhat + eps
    h_nom = B_hocbf(xhat)
    Lf_nom = jax.jacobian(B_hocbf)(xhat) @ f(xhat)

    def worst_i(i, k):
        _, x_wc = proj_grad_max_box(
            lambda x: (jax.jacobian(B_hocbf)(x) @ f(x) + ALPHA_QP * B_hocbf(x))[i],
            lo, hi, n_restarts, n_steps, lr, k)
        h_wc = B_hocbf(x_wc)[i]
        Lf_wc = (jax.jacobian(B_hocbf)(x_wc) @ f(x_wc))[i]
        delta = (Lf_wc + ALPHA_QP * h_wc) - (Lf_nom[i] + ALPHA_QP * h_nom[i])
        return jnp.clip(delta, 0.0, PHI_MAX)

    keys = jr.split(key, 1)
    return jax.vmap(worst_i)(jnp.arange(1), keys)


params = phi_net.init(jr.PRNGKey(0), jnp.zeros(4), jnp.zeros(4))["params"]
x_lo = jnp.array([-3., -2., -3., -2.])
x_hi = jnp.array([3., 2., 3., 2.])
N_TRAIN, N_TEST = 20000, 5000
rng_phi = jr.PRNGKey(7)
rng_phi, rng_tr, rng_te, rng_eps_tr, rng_eps_te = jr.split(rng_phi, 5)
outside = lambda xh: jnp.sqrt(xh[0] ** 2 + xh[2] ** 2) > R_OBS + 0.1
cands_tr = x_lo + jr.uniform(rng_tr, (N_TRAIN * 2, 4)) * (x_hi - x_lo)
cands_te = x_lo + jr.uniform(rng_te, (N_TEST * 2, 4)) * (x_hi - x_lo)
xhat_train = cands_tr[jax.vmap(outside)(cands_tr)][:N_TRAIN]
xhat_test = cands_te[jax.vmap(outside)(cands_te)][:N_TEST]
eps_train = jr.uniform(rng_eps_tr, (N_TRAIN, 4)) * EPS_MAX
eps_test = jr.uniform(rng_eps_te, (N_TEST, 4)) * EPS_MAX
keys_tr = jr.split(jr.PRNGKey(1), N_TRAIN)
keys_te = jr.split(jr.PRNGKey(2), N_TEST)

t0 = time.time()
phi_train = jax.vmap(phi_true_fn)(xhat_train, eps_train, keys_tr)
phi_test = jax.vmap(phi_true_fn)(xhat_test, eps_test, keys_te)
print(f"labels in {time.time() - t0:.0f} s; phi mean {float(phi_train.mean()):.3f} max {float(phi_train.max()):.3f}, at cap {float((phi_train >= PHI_MAX - 1e-4).mean()):.2%}")

optimizer = optax.adam(3e-3)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, xhat_b, eps_b, phi_b):
    def loss_fn(p):
        pred = jax.vmap(lambda x, e: phi_net.apply({"params": p}, x, e))(xhat_b, eps_b)
        return jnp.mean((pred - phi_b) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt, loss


t0 = time.time()
rng_ep = jr.PRNGKey(42)
for epoch in range(args.epochs):
    rng_ep, rng_s = jr.split(rng_ep)
    perm = jr.permutation(rng_s, N_TRAIN)
    for b in range(0, N_TRAIN, 256):
        idx = perm[b:b + 256]
        params, opt_state, ep_loss = train_step(params, opt_state, xhat_train[idx], eps_train[idx], phi_train[idx])
    if (epoch + 1) % 100 == 0:
        pred = jax.vmap(lambda x, e: phi_net.apply({"params": params}, x, e))(xhat_test, eps_test)
        print(f"epoch {epoch + 1}: train {float(ep_loss):.3e}  val {float(jnp.mean((pred - phi_test) ** 2)):.3e}  ({time.time() - t0:.0f} s)")

args.out.parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "wb") as f_:
    pickle.dump(params, f_)
print("saved", args.out)
