from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from mrncbf.dyn.dyn_types import Obs
from mrncbf.networks.network_utils import default_nn_init
from mrncbf.utils.tfp import TanhTransformedDistribution, tfd


class PolDet(nn.Module):
    """Automatically clipped from -1 to 1."""

    base_cls: Type[nn.Module]
    _nu: int

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        out = nn.Dense(self.nu, kernel_init=default_nn_init(), name="Output")(x)
        out = jnp.tanh(out)
        return out

    @property
    def nu(self):
        return self._nu
