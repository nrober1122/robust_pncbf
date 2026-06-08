"""Flax MLP value function `phi(x)` loaded from a pickled parameter tree.

Matches the architecture inferred in ``notebooks/00_investigate_model.ipynb``:
a plain Dense MLP with tanh (or relu) activations between hidden layers and
a linear output head.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class PhiMLP(nn.Module):
    hidden_sizes: Sequence[int]
    out_size: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.tanh if self.activation == "tanh" else nn.relu
        for h in self.hidden_sizes:
            x = act(nn.Dense(h)(x))
        return nn.Dense(self.out_size)(x)


@dataclass
class ValueFunction:
    """Callable wrapper around a Flax PhiMLP with loaded params."""

    module: PhiMLP
    params: dict
    input_dim: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        y = self.module.apply({"params": self.params}, jnp.asarray(x))
        y = np.asarray(y)
        return y[0] if single else y


def load_value_function(
    checkpoint_path: Path | str,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: str = "tanh",
) -> ValueFunction:
    path = Path(checkpoint_path)
    with open(path, "rb") as f:
        params = pickle.load(f)

    module = PhiMLP(
        hidden_sizes=tuple(hidden_sizes),
        out_size=output_dim,
        activation=activation,
    )

    # Smoke-test the apply so a mismatched checkpoint fails loudly at startup
    # instead of inside the control callback.
    _ = module.apply({"params": params}, jnp.zeros((1, input_dim)))

    return ValueFunction(module=module, params=params, input_dim=input_dim)
