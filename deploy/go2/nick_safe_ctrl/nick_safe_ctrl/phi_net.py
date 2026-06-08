"""Numpy port of the Flax ``PhiNet`` from ``dbint2d_quad.py``.

The notebook's NMR-CBF uses a small MLP ``phi(xhat, eps) -> R+`` that approximates
the PSHR-CBF oracle (worst-case barrier tightening over the eps-box).  This
module loads the pickled Flax parameter tree and evaluates it in pure numpy,
so the deployment package needs no jax/flax dependency.

Architecture (from PhiNet in ``dbint2d_quad.py``)::

    inputs:  xhat (4,)  and  eps (4,)   -- concatenated to (8,)
    Dense(64) -> tanh
    Dense(64) -> tanh
    Dense(1)  -> softplus               -- output: phi >= 0

This supersedes the older ``value_function.py`` (which used the wrong
``PhiMLP`` shape -- no softplus on the output -- and pulled in jax/flax
imports that don't install on this Jetson).  ``value_function.py`` is left in
place but is no longer imported.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np


def _install_jax_pickle_stubs() -> None:
    """Stub ``jax._src.array._reconstruct_array`` with a numpy passthrough.

    The pickled checkpoint reduces its array leaves via this jax-internal
    function; on a machine without jax, ``pickle.load`` raises
    ``ModuleNotFoundError``.  Since the array data itself is a numpy ndarray
    passed in as one of the args, we can intercept the reconstructor and just
    return that ndarray.  This keeps the deployment package free of any jax
    dependency while letting us load checkpoints produced on a jax machine.
    """
    if "jax" in sys.modules:
        return

    def _reconstruct_array(*args, **kwargs):
        # Format observed: args = (numpy_reconstructor, ctor_args, setstate, meta).
        # Use the numpy pickle protocol directly: empty array, then __setstate__.
        if len(args) >= 3 and callable(args[0]) and isinstance(args[1], tuple) \
                and isinstance(args[2], tuple):
            arr = args[0](*args[1])
            arr.__setstate__(args[2])
            return arr
        # Already-an-ndarray fallback (older jax format).
        for a in args:
            if isinstance(a, np.ndarray):
                return a
        for v in kwargs.values():
            if isinstance(v, np.ndarray):
                return v
        raise RuntimeError(
            f"unexpected jax._src.array._reconstruct_array call signature: "
            f"{len(args)} args, types={[type(a).__name__ for a in args]}")

    jax_mod = types.ModuleType("jax")
    jax_src_mod = types.ModuleType("jax._src")
    jax_src_array_mod = types.ModuleType("jax._src.array")
    jax_src_array_mod._reconstruct_array = _reconstruct_array
    sys.modules["jax"] = jax_mod
    sys.modules["jax._src"] = jax_src_mod
    sys.modules["jax._src.array"] = jax_src_array_mod


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus:  log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class PhiNet:
    """Forward-pass-only numpy MLP.  Matches the Flax PhiNet's architecture."""

    def __init__(self, params: dict, hidden_dims: Sequence[int] = (64, 64)):
        self.hidden_dims = tuple(hidden_dims)
        self._W: list = []
        self._b: list = []
        n_layers = len(self.hidden_dims) + 1
        for i in range(n_layers):
            key = f"Dense_{i}"
            if key not in params:
                raise ValueError(
                    f"missing layer '{key}' in params; got {list(params.keys())}")
            self._W.append(np.asarray(params[key]["kernel"], dtype=np.float32))
            self._b.append(np.asarray(params[key]["bias"], dtype=np.float32))

        # shape sanity
        if self._W[0].shape[0] != 8:
            raise ValueError(
                f"PhiNet expects input dim 8 (xhat 4 + eps 4); got {self._W[0].shape[0]}")
        for i, h in enumerate(self.hidden_dims):
            if self._W[i].shape[1] != h:
                raise ValueError(
                    f"layer {i} hidden dim mismatch: {self._W[i].shape[1]} vs {h}")
            if self._W[i + 1].shape[0] != h:
                raise ValueError(
                    f"layer {i+1} input dim mismatch: {self._W[i+1].shape[0]} vs {h}")
        if self._W[-1].shape[1] != 1:
            raise ValueError(f"PhiNet output dim {self._W[-1].shape[1]} != 1")

    def __call__(self, xhat: np.ndarray, eps: np.ndarray) -> float:
        x = np.concatenate([
            np.asarray(xhat, dtype=np.float32),
            np.asarray(eps, dtype=np.float32),
        ])
        for W, b in zip(self._W[:-1], self._b[:-1]):
            x = np.tanh(x @ W + b)
        y = _softplus(x @ self._W[-1] + self._b[-1])
        return float(y[0])


def load_phi_net(path) -> PhiNet:
    """Load a pickled Flax parameter tree and wrap it in a numpy PhiNet."""
    _install_jax_pickle_stubs()
    path = Path(path)
    with open(path, "rb") as f:
        params = pickle.load(f)
    # Flax pickles store nested dicts whose leaves may be jax arrays; convert.
    params_np = {
        k: {kk: np.asarray(vv) for kk, vv in v.items()}
        for k, v in params.items()
    }
    return PhiNet(params_np)


if __name__ == "__main__":
    # Smoke test: load the bundled checkpoint and run a few inputs.
    import sys
    ckpt = Path(__file__).resolve().parent.parent / "ckpts" / "phi_params.pkl"
    if not ckpt.exists():
        print(f"checkpoint not found at {ckpt}", file=sys.stderr)
        raise SystemExit(1)
    net = load_phi_net(ckpt)
    eps = np.array([0.10, 0.05, 0.10, 0.05])
    for xhat in [
        np.array([-1.5, 0.0,  0.10, 0.0]),
        np.array([-1.0, 0.5, -0.05, 0.0]),
        np.array([ 0.5, 0.0,  0.30, 0.0]),
        np.array([ 2.0, 0.0,  0.00, 0.0]),
    ]:
        phi = net(xhat, eps)
        print(f"  PhiNet({xhat.tolist()}, eps={eps.tolist()}) = {phi:.4f}")
    print(f"params: {sum(W.size + b.size for W, b in zip(net._W, net._b))} total weights loaded")
