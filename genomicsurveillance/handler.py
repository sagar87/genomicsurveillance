from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.experimental import host_callback
from numpyro import optim
from numpyro.diagnostics import hpdi
from numpyro.infer import SVI, Predictive, Trace_ELBO

from genomicsurveillance.types import Guide, Model


class Posterior(object):
    """
    Caches a posterior.

    :param posterior: the posterior distribution
    """

    def __init__(self, posterior: dict, to_numpy: bool = True):
        self.data = posterior

        if to_numpy:
            self.data = self._to_numpy(posterior)

    def _to_numpy(self, posterior):
        return {k: np.asarray(v) for k, v in posterior.items()}

    @lru_cache(maxsize=128)
    def median(self, param, *args, **kwargs):
        """Returns the median of param."""
        return jnp.median(self.data[param][tuple([slice(None), *args])], axis=0)

    @lru_cache(maxsize=128)
    def mean(self, param, *args, **kwargs):
        """Returns the mean of param."""
        return self.data[param][tuple([slice(None), *args])].mean(0)

    @lru_cache(maxsize=128)
    def hpdi(self, param, *args, **kwargs):
        """Returns the highest predictive density interval of param."""
        return hpdi(self.data[param][tuple([slice(None), *args])], **kwargs)

    @lru_cache(maxsize=128)
    def quantiles(self, param, *args, **kwargs):
        """Returns the quantiles of param."""
        return jnp.quantile(
            self.data[param][tuple([slice(None), *args])],
            jnp.array([0.025, 0.975]),
            axis=0,
        )

    def qlower(self, param, *args, **kwargs):
        """Returns the quantile lower bound of param."""
        return self.quantiles(param, *args, **kwargs)[0]

    def qupper(self, param, *args, **kwargs):
        """Returns the quantile upper bound of param."""
        return self.quantiles(param, *args, **kwargs)[1]

    def lower(self, param, *args, **kwargs):
        """Returns the HPDI lower bound of param."""
        return self.hpdi(param, *args, **kwargs)[0]

    def upper(self, param, *args, **kwargs):
        """Returns the HPDI upper bound of param."""
        return self.hpdi(param, *args, **kwargs)[1]

    def ci(self, param, *args, **kwargs):
        return np.abs(
            self.mean(param, *args, **kwargs) - self.hpdi(param, *args, **kwargs)
        )


class SVIHandler(object):
    """
    Helper object that abstracts some of numpyros complexities. Inspired
    by an implementation of Florian Wilhelm.

    :param model: A numpyro model.
    :param guide: A numpyro guide.
    :param loss: Loss function, defaults to Trace_ELBO.
    :param lr: Learning rate, defaults to 0.01.
    :param rng_key: Random seed, defaults to 254.
    :param num_epochs: Number of epochs to train the model, defaults to 5000.
    :param num_samples: Number of posterior samples.
    :param log_func: Logging function, defaults to print.
    :param log_freq: Frequency of logging, defaults to 0 (no logging).
    :param to_numpy: Convert the posterior distribution to numpy array(s),
        defaults to True.
    """

    def __init__(
        self,
        model: Model,
        guide: Guide,
        loss: Trace_ELBO = Trace_ELBO(num_particles=1),
        optimizer: optim.optimizers.optimizer = optim.Adam,
        lr: float = 0.01,
        rng_key: int = 254,
        num_epochs: int = 5000,
        num_samples: int = 1000,
        log_func=print,
        log_freq=0,
        to_numpy: bool = True,
    ):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optimizer = optimizer(step_size=lr)
        self.rng_key = random.PRNGKey(rng_key)

        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss)
        self.init_state = None

        self.log_func = log_func
        self.log_freq = log_freq
        self.num_epochs = num_epochs
        self.num_samples = num_samples

        self.loss = None
        self.to_numpy = to_numpy

    def _log(self, epoch, loss, n_digits=4):
        msg = f"epoch: {str(epoch).rjust(n_digits)} loss: {loss: 16.4f}"
        self.log_func(msg)

    def _fit(self, epochs, *args):

        return lax.scan(
            lambda state, i: (
                self.svi.update(
                    lax.cond(
                        i % self.log_freq == 0,
                        lambda _: host_callback.id_tap(
                            _print_consumer, (i, self.num_epochs), result=state
                        ),
                        lambda _: state,
                        operand=None,
                    ),
                    *args,
                )
            ),
            self.init_state,
            jnp.arange(epochs),
        )

    def _update_state(self, state, loss):
        self.state = state
        self.init_state = state
        self.loss = loss if self.loss is None else jnp.concatenate([self.loss, loss])

    def fit(self, *args, **kwargs):
        num_epochs = kwargs.get("num_epochs", self.num_epochs)

        if self.init_state is None:
            self.init_state = self.svi.init(self.rng_key, *args)

        state, loss = self._fit(num_epochs, *args)
        self._update_state(state, loss)
        self.params = self.svi.get_params(state)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=self.num_samples,
        )

        self.posterior = Posterior(predictive(self.rng_key, *args), self.to_numpy)

    def get_posterior_predictive(self, *args, **kwargs):
        """kwargs -> Predictive, args -> predictive"""
        num_samples = kwargs.pop("num_samples", self.num_samples)
        rng_key = kwargs.pop("rng_key", self.rng_key)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
            **kwargs,
        )

        self.predictive = Posterior(predictive(rng_key, *args), self.to_numpy)


class SVIModel(SVIHandler):
    """
    Abstract class of the SVI handler. To be used classes that implement
    a numpyro model and guide.
    """

    def __init__(self, **kwargs):
        super().__init__(self.model, self.guide, **kwargs)

    def model(self):
        raise NotImplementedError()

    def guide(self):
        raise NotImplementedError()


def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(
        f"SVI step {iter_num:,} / {num_samples:,} | {iter_num/num_samples * 100:.0f} %"
    )
