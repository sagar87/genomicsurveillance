import functools
import pickle

import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.experimental import host_callback
from numpyro import optim
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO

from genomicsurveillance.types import Guide, Model


def ignore_unhashable(func):
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ("cache_info", "cache_clear")

    @functools.wraps(func, assigned=attributes)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "unhashable type" in str(error):
                return uncached(*args, **kwargs)
            raise

    wrapper.__uncached__ = uncached
    return wrapper


def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(
        f"SVI step {iter_num:,} / {num_samples:,} | {iter_num/num_samples * 100:.0f} %"
    )


def make_array(arg):
    if isinstance(arg, int):
        arr = np.array([arg])
    elif isinstance(arg, list):
        arr = np.array(arg)
    else:
        arr = arg

    return arr


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

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def indices(self, shape, *args):
        """
        Creates indices for easier access to variables.
        """
        indices = [np.arange(shape[0])]
        for i, arg in enumerate(args):
            if arg is None:
                indices.append(np.arange(shape[i + 1]))
            else:
                indices.append(make_array(arg))
        return np.ix_(*indices)

    def dist(self, param, *args, **kwargs):
        indices = self.indices(self[param].shape, *args)
        return self[param][indices]

    @ignore_unhashable
    @functools.lru_cache(maxsize=128)
    def median(self, param, *args):
        """Returns the median of param."""
        return np.median(self.dist(param, *args), axis=0)

    @ignore_unhashable
    @functools.lru_cache(maxsize=128)
    def mean(self, param, *args):
        """Returns the mean of param."""
        return np.mean(self.dist(param, *args), axis=0)

    @ignore_unhashable
    @functools.lru_cache(maxsize=128)
    def hpdi(self, param, *args, **kwargs):
        """Returns the highest predictive density interval of param."""
        return hpdi(self.dist(param, *args), **kwargs)

    @ignore_unhashable
    @functools.lru_cache(maxsize=128)
    def quantiles(self, param, *args, **kwargs):
        """Returns the quantiles of param."""
        q = kwargs.pop("q", [0.025, 0.975])
        return np.quantile(
            self.dist(param, *args),
            q,
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
        return np.abs(self.mean(param, *args) - self.hpdi(param, *args, **kwargs))


class Handler(object):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def dump_posterior(self, file_name: str):
        assert self.posterior is not None, "'init_svi' needs to be called first"
        pickle.dump(self.posterior.data, open(file_name, "wb"))

    def load_posterior(self, file_name):
        self.posterior = Posterior(pickle.load(open(file_name, "rb")))


class SVIHandler(Handler):
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
        lr: float = 0.001,
        rng_key: int = 254,
        num_epochs: int = 30000,
        num_samples: int = 1000,
        log_func=_print_consumer,
        log_freq=1000,
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

    def _fit(self, *args):
        def _step(state, i, *args):
            state = lax.cond(
                i % self.log_freq == 0,
                lambda _: host_callback.id_tap(
                    self.log_func, (i, self.num_epochs), result=state
                ),
                lambda _: state,
                operand=None,
            )
            return self.svi.update(state, *args)

        return lax.scan(
            lambda state, i: _step(state, i, *args),
            self.init_state,
            jnp.arange(self.num_epochs),
        )

    def _update_state(self, state, loss):
        self.state = state
        self.init_state = state
        self.loss = loss if self.loss is None else jnp.concatenate([self.loss, loss])

    def fit(self, *args, **kwargs):
        self.num_epochs = kwargs.pop("num_epochs", self.num_epochs)
        predictive_kwargs = kwargs.pop("predictive_kwargs", {})

        if self.init_state is None:
            self.init_state = self.svi.init(self.rng_key, *args)

        state, loss = self._fit(*args)
        self._update_state(state, loss)
        self.params = self.svi.get_params(state)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=self.num_samples,
            **predictive_kwargs,
        )

        self.posterior = Posterior(predictive(self.rng_key, *args), self.to_numpy)

        return self

    def predict(self, *args, **kwargs):
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

    def dump_params(self, file_name: str):
        assert self.params is not None, "'init_svi' needs to be called first"
        pickle.dump(self.params, open(file_name, "wb"))

    def load_params(self, file_name):
        self.params = pickle.load(open(file_name, "rb"))


class NutsHandler(Handler):
    def __init__(
        self,
        model,
        num_warmup=2000,
        num_samples=10000,
        num_chains=1,
        rng_key=0,
        to_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.model = model
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.rng_key, self.rng_key_ = random.split(random.PRNGKey(rng_key))
        self.to_numpy = to_numpy
        self.kernel = NUTS(model, **kwargs)
        self.mcmc = MCMC(self.kernel, num_warmup, num_samples, num_chains=num_chains)

    def predict(self, *args, **kwargs):
        predictive = Predictive(self.model, self.posterior.data, **kwargs)
        self.predictive = Posterior(predictive(self.rng_key_, *args))

    def fit(self, *args, **kwargs):
        self.num_samples = kwargs.get("num_samples", self.num_samples)

        self.mcmc.run(self.rng_key_, *args, **kwargs)
        self.posterior = Posterior(self.mcmc.get_samples(), self.to_numpy)

    def summary(self, *args, **kwargs):
        self.mcmc.print_summary(*args, **kwargs)


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


class Model(SVIHandler, NutsHandler):
    _latent_variables = []

    def __init__(self, handler="SVI", *args, **kwargs):
        self.handler = handler
        if handler == "SVI":
            SVIHandler.__init__(self, self.model, self.guide, *args, **kwargs)
        elif handler == "NUTS":
            NutsHandler.__init__(self, self.model, *args, **kwargs)

    def model(self):
        raise NotImplementedError()

    def guide(self):
        raise NotImplementedError()

    def fit(
        self, predictive_kwargs: dict = {}, deterministic: bool = True, *args, **kwargs
    ):
        if self.handler == "SVI":
            if len(predictive_kwargs) == 0:
                predictive_kwargs["return_sites"] = tuple(self._latent_variables)
            SVIHandler.fit(self, predictive_kwargs=predictive_kwargs, *args, **kwargs)
        elif self.handler == "NUTS":
            NutsHandler.fit(self, *args, **kwargs)

        if deterministic:
            self.deterministic()

    def deterministic(self):
        pass
