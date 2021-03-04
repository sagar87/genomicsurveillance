


import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax, random

from functools import lru_cache
from numpyro import optim
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO


class Posterior(object):
    """
    Caches a posterior.
    """
    def __init__(self, posterior: dict):
        self.data = posterior

    def _to_numpy(self, posterior):
        return {k: np.asarray(v) for k, v in posterior.items()}

    @lru_cache(maxsize=128)
    def median(self, param, which, *args, **kwargs):
        return jnp.median(
            self._select(which)[param][tuple([slice(None), *args])], axis=0
        )

    @lru_cache(maxsize=128)
    def mean(self, param, which, *args, **kwargs):
        return self._select(which)[param][tuple([slice(None), *args])].mean(0)

    @lru_cache(maxsize=128)
    def hpdi(self, param, which, *args, **kwargs):
        return hpdi(self._select(which)[param][tuple([slice(None), *args])], **kwargs)

    @lru_cache(maxsize=128)
    def quantiles(self, param, which, *args, **kwargs):
        return jnp.quantile(
            self._select(which)[param][tuple([slice(None), *args])],
            jnp.array([0.05, 0.95]),
            axis=0,
        )

    def qlower(self, param, which, *args, **kwargs):
        return self.quantiles(param, which, *args, **kwargs)[0]

    def qupper(self, param, which, *args, **kwargs):
        return self.quantiles(param, which, *args, **kwargs)[1]

    def lower(self, param, which, *args, **kwargs):
        return self.hpdi(param, which, *args, **kwargs)[0]

    def upper(self, param, which, *args, **kwargs):
        return self.hpdi(param, which, *args, **kwargs)[1]

    def ci(self, param, which, *args, **kwargs):
        return np.abs(
            self.mean(param, which, *args, **kwargs)
            - self.hpdi(param, which, *args, **kwargs)
        )
