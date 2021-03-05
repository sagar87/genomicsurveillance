import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax, random
from jax.scipy.special import gammaln, xlogy
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    is_prng_key,
    multinomial,
    promote_shapes,
    validate_sample,
)


class MultinomialProbs(dist.Distribution):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }
    is_discrete = True

    def __init__(self, probs, total_count=1, scale=1.0, validate_args=None):
        if jnp.ndim(probs) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        batch_shape = lax.broadcast_shapes(
            jnp.shape(probs)[:-1], jnp.shape(total_count)
        )
        self.probs = promote_shapes(probs, shape=batch_shape + jnp.shape(probs)[-1:])[0]
        self.total_count = promote_shapes(total_count, shape=batch_shape)[0]
        self.scale = scale
        super(MultinomialProbs, self).__init__(
            batch_shape=batch_shape,
            event_shape=jnp.shape(self.probs)[-1:],
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        print("in multinomial sample")
        return multinomial(
            key, self.probs, self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.scale * (
            gammaln(self.total_count + 1)
            + jnp.sum(xlogy(value, self.probs) - gammaln(value + 1), axis=-1)
        )

    @property
    def mean(self):
        return self.probs * jnp.expand_dims(self.total_count, -1)

    @property
    def variance(self):
        return jnp.expand_dims(self.total_count, -1) * self.probs * (1 - self.probs)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)


class NegativeBinomial(dist.Distribution):
    r"""
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The ``rate`` parameter for the
    :class:`~numpyro.distributions.Poisson` distribution is unknown and randomly
    drawn from a :class:`~numpyro.distributions.Gamma` distribution.

    :param numpy.ndarray concentration: shape parameter (alpha) of the Gamma distribution.
    :param numpy.ndarray rate: rate parameter (beta) for the Gamma distribution.
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, mu, tau, validate_args=None):
        self.mu, self.tau = promote_shapes(mu, tau)
        # converts mean var parametrisation to r and p
        self.r = tau
        self.var = mu + 1 / self.r * mu ** 2
        self.p = (self.var - mu) / self.var
        self._gamma = dist.Gamma(self.r, (1 - self.p) / self.p)
        super(NegativeBinomial, self).__init__(
            self._gamma.batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        key_gamma, key_poisson = random.split(key)
        rate = self._gamma.sample(key_gamma, sample_shape)
        return dist.Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):

        return (
            self.tau * jnp.log(self.tau)
            - gammaln(self.tau)
            + gammaln(value + self.tau)
            + value * jnp.log(self.mu)
            - jnp.log(self.mu + self.tau) * (self.tau + value)
            - gammaln(value + 1)
        )
