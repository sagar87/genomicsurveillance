import jax.numpy as jnp
import numpy as np
from jax.ops import index, index_update
from jax.scipy.special import logsumexp

from genomicsurveillance.handler import Posterior, make_array
from genomicsurveillance.utils import TruncatedKnots

from .sites import Sites


class Lineage(object):
    EPS = -1e6
    SCALE = 100.0
    LTLA_DIM = 1
    TIME_DIM = 2
    LIN_DIM = 3
    """
    Implements Lineage model helper.
    """

    def __init__(
        self,
        tau,
        cases,
        lineages,
        lineage_dates,
        population,
        basis=None,
        auto_correlation=0.5,
        posterior=None,
    ):
        self.tau = tau
        self.cases = cases
        self.lineages = lineages
        self.lineage_dates = lineage_dates
        self.population = population
        self.posterior = Posterior(posterior) if posterior is not None else posterior
        self.auto_correlation = auto_correlation

        if basis is None:
            knots = TruncatedKnots(cases.shape[-1], periods=14)
            self.B = knots.basis
            # _, self.B = create_spline_basis(
            #     np.arange(cases.shape[1]),
            #     num_knots=int(np.ceil(cases.shape[1] / 10)),
            #     add_intercept=False,
            # )
        else:
            self.B = basis

        self.num_ltla = self.cases.shape[0]
        self.num_time = self.cases.shape[-1]
        self.num_lin = self.lineages.shape[-1] - 1
        self.num_basis = self.B.shape[-1]
        self.num_ltla_lin = self.nan_idx.shape[0]

    # Private methods
    def _expand_dims(self, array, num_dim=4, dim=1):
        """Soft dim expansion."""
        if array.ndim < num_dim:
            array = np.expand_dims(array, dim)
        return array

    def _expand_array(self, array: jnp.ndarray, index, shape: tuple) -> jnp.ndarray:
        """Creates an a zero array with shape `shape` and fills it with `array` at index."""
        expanded_array = jnp.zeros(shape)
        expanded_array = index_update(expanded_array, index, array)
        return expanded_array

    def _pad_array(self, array: jnp.ndarray, func=jnp.zeros):
        """Adds an additional column to an three dimensional array."""
        return jnp.concatenate(
            [array, func((array.shape[0], *[1 for _ in range(array.ndim - 1)]))], -1
        )

    def _indices(self, shape, *args):
        """Creates indices for easier access to variables."""
        indices = []
        for i, arg in enumerate(args):
            if arg is None:
                indices.append(np.arange(shape[i]))
            else:
                indices.append(make_array(arg))

        return np.ix_(*indices)

    def _is_nan_row(self, array: np.ndarray) -> np.ndarray:
        """
        Helper function to extract the indices of rows (1st dimension) in an array that
        contains nan values

        :param array: a numpy array
        :returns: an array of indices
        """
        return np.where(np.isnan(array.sum(axis=tuple(range(1, array.ndim)))))[0]

    @property
    def arma(self):
        if not hasattr(self, "_arma"):
            Σ0 = jnp.eye(self.num_basis)
            for i in range(1, self.num_basis):
                Σ0 = index_update(Σ0, index[i, i - 1], jnp.array(self.auto_correlation))

            Π0 = jnp.linalg.inv(Σ0)

            for i in range(self.num_basis - 3, self.num_basis):
                Π0 = index_update(Π0, index[i, i - 2 : i], jnp.array([1, -2]))

            Π0 = index_update(
                Π0,
                index[self.num_basis - 3, self.num_basis - 5 : self.num_basis - 3],
                0.5 * jnp.array([1, -2]),
            )
            self._arma = jnp.linalg.inv(Π0)
        return self._arma

    @property
    def nan_idx(self):
        if not hasattr(self, "_nan_idx"):
            exclude = list(
                set(self._is_nan_row(self.lineages)) | set(self._is_nan_row(self.cases))
            )
            self._nan_idx = np.array(
                [i for i in range(self.cases.shape[0]) if i not in exclude]
            )
        return self._nan_idx

    @property
    def missing_lineages(self):
        if not hasattr(self, "_missing_lineages"):
            self._missing_lineages = (self.lineages[..., :-1].sum(1) != 0)[
                self.nan_idx
            ].astype(int)
        return self._missing_lineages

    def aggregate(self, region, func, *args, **kwargs):
        agg = []
        for i in np.sort(np.unique(region)):
            region_idx = np.where(region == i)[0]
            region_not_nan = np.isin(region_idx, self.nan_idx)
            region_idx = region_idx[region_not_nan]
            aggregate = func(int(region_idx[0]), *args, **kwargs)

            for r in region_idx[1:]:
                aggregate += func(int(r), *args, **kwargs)

            agg.append(aggregate)

        return np.concatenate(agg, 1)

    def get_logits(self, ltla=None, time=None, lineage=None):

        logits = self.posterior.dist(Sites.B1, ltla, None, lineage) * np.arange(
            0, self.num_time
        )[make_array(time)].reshape(1, -1, 1) + self.posterior.dist(
            Sites.C1, ltla, None, lineage
        )
        logits = self._expand_dims(logits)
        return logits

    def get_probabilities(self, ltla=None, time=None, lineage=None):
        logits = self.get_logits(ltla, time)
        p = np.exp(logits - logsumexp(logits, -1, keepdims=True))

        if lineage is not None:
            idx = make_array(lineage)
        else:
            idx = slice(None)

        return p[..., idx]

    def get_growth_rate(self, ltla=None, time=None):
        """
        Computes the fitted growth rate.

        :param ltla: indices of the the LTLAs
        :param time: time indices
        :returns: the posterior distribution of the incidence
            with shape (num_samples, num_ltla, num_time, 1)
        """
        beta = self._expand_dims(self.posterior.dist(Sites.BETA1, ltla), self.LIN_DIM)
        basis = self._expand_dims(
            self.B[self._indices(self.B.shape, 1, time)].T.squeeze(), self.TIME_DIM
        )
        gr = np.einsum("ijk,kl->ijl", beta, basis)
        gr = self._expand_dims(gr, dim=self.LIN_DIM)
        return gr

    def get_growth_rate_lineage(self, ltla, time=None, lineage=None):
        p = self.get_probabilities(ltla, time)
        b1 = self._expand_dims(self.posterior.dist(Sites.B1, ltla), dim=self.TIME_DIM)
        gr = self.get_growth_rate(ltla, time)
        gr_lin = gr - np.einsum("mijk,milk->mijl", p, b1) + b1

        if lineage is not None:
            idx = make_array(lineage)
        else:
            idx = slice(None)

        return gr_lin[..., idx]

    def get_lambda(self, ltla=None, time=None):
        """
        Returns the posterior distribution of the incidence.

        :param ltla: indices of the the LTLAs
        :param time: time indices
        :returns: the posterior distribution of the incidence
            with shape (num_samples, num_ltla, num_time, 1)
        """
        beta = self.posterior.dist(Sites.BETA1, ltla)
        beta = self._expand_dims(beta, self.LIN_DIM)
        basis = self._expand_dims(
            self.B[self._indices(self.B.shape, 0, time)].T.squeeze(), self.TIME_DIM
        )

        lamb = self.population[self._indices(self.population.shape, ltla)].reshape(
            1, -1, 1
        ) * np.exp(np.einsum("ijk,kl->ijl", beta, basis))
        lamb = self._expand_dims(lamb, dim=self.LIN_DIM)
        return lamb

    def get_lambda_lineage(self, ltla=None, time=None, lineage=None):
        return self.get_lambda(ltla, time) * self.get_probabilities(ltla, time, lineage)

    def get_transmissibility(self, rebase=None):
        b = self.posterior.dist(Sites.B0)
        b = np.concatenate([b, np.zeros((b.shape[0], 1))], -1)

        if rebase is not None:
            b = b - b[..., rebase].reshape(-1, 1)

        return np.exp(b * self.tau)

    def get_log_R(self, ltla=None, time=None):
        log_R = (
            self.posterior.dist(Sites.BETA1, ltla)
            @ self.B[self._indices(self.B.shape, 1, time)].T.squeeze()
        ) * self.tau

        if log_R.ndim == 2:
            if isinstance(time, int):
                log_R = self._expand_dims(log_R, num_dim=3, dim=self.TIME_DIM)
            if isinstance(ltla, int):
                log_R = self._expand_dims(log_R, num_dim=3, dim=self.LIN_DIM)

        return self._expand_dims(log_R, dim=self.LIN_DIM)

    def get_log_R_lineage(self, ltla=None, time=None, lineage=None):
        p = self.get_probabilities(ltla, time)
        # TODO: set this up
        # b = self.posterior.dist(Sites.B0, lineage)
        # b1 = np.concatenate([b, np.zeros((b.shape[0], 1))], -1)
        b1 = self._expand_dims(self.posterior.dist(Sites.B1, ltla), dim=self.TIME_DIM)
        log_R = self.get_log_R(ltla, time)

        log_R_lineage = (log_R - (np.einsum("mijk,milk->mijl", p, b1) * self.tau)) + (
            b1 * self.tau
        )

        if lineage is not None:
            idx = make_array(lineage)
        else:
            idx = slice(None)

        return log_R_lineage[..., idx]

    def get_other_log_R(self, exclude, ltla=None, time=None):
        p = self.get_probabilities(ltla, time, exclude)
        b = self.posterior.dist(Sites.B0)
        b = np.concatenate([b, np.zeros((b.shape[0], 1))], -1)
        log_R = self.get_log_R(ltla, time)
        log_R0 = log_R - (b[..., exclude].reshape(-1, 1, 1, 1) * p) * self.tau

        return self._expand_dims(log_R0, dim=self.LIN_DIM)

    def aggregate_lambda(self, region, time=None):
        return self.aggregate(region, self.get_lambda, time)

    def aggregate_growth_rate(self, region, time=None):
        return self.aggregate(region, self.get_growth_rate, time)

    def aggregate_lambda_lineage(self, region, time=None, lineage=None):
        """
        Aggregates lambda lineage by an indicator array.

        :param region: an indicator array, e.g. np.array([0, 0, 0, 1, 1])
        :param time: index array containing indices of time
        :param lineage: index array containing indices of lineage of interest
        :return: a numpy array containing aggregated incidence due to each lineage
        """
        return self.aggregate(region, self.get_lambda_lineage, time, lineage)

    def aggregate_probabilities(self, region, time=None, lineage=None):
        lambda_lin = self.aggregate_lambda_lineage(region, time=time, lineage=lineage)
        return lambda_lin / lambda_lin.sum(-1, keepdims=True)

    def aggregate_log_R(self, region, time=None):
        lambda_regions = self.aggregate_lambda(region, time)

        def weighted_log_R(ltla, time):
            return self.get_log_R(ltla, time) * self.get_lambda(ltla, time)

        agg = self.aggregate(region, weighted_log_R, time)
        return agg / lambda_regions

    def aggregate_log_R_lineage(self, region, time=None):
        lambda_regions = self.aggregate_lambda_lineage(region, time)

        def weighted_log_R(ltla, time):
            return self.get_log_R(ltla, time) * self.get_lambda_lineage(ltla, time)

        agg = self.aggregate(region, weighted_log_R, time)
        return agg / lambda_regions

    def aggregate_growth_rate_lineage(self, region, time=None):
        lambda_regions = self.aggregate_lambda_lineage(region, time)

        def weighted_growth_rate(ltla, time):
            return self.get_growth_rate(ltla, time) * self.get_lambda_lineage(
                ltla, time
            )

        agg = self.aggregate(region, weighted_growth_rate, time)
        return agg / lambda_regions
