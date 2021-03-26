import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax.ops import index, index_update
from jax.scipy.special import logsumexp

from genomicsurveillance.distributions import NegativeBinomial
from genomicsurveillance.handler import Model
from genomicsurveillance.utils import create_spline_basis, is_nan_row

from .sites import Sites


class Lineage(object):
    """
    Implements Lineage model helper.
    """

    def __init__(self):
        pass

    def _nan_idx(self, cases, lineages):
        exclude = list(set(is_nan_row(lineages)) | set(is_nan_row(cases)))
        return np.array([i for i in range(cases.shape[0]) if i not in exclude])

    def _missing_lineages(self, lineages):
        return lineages[..., :-1].sum(1) != 0

    def _is_int(self, array):
        if type(array) == int:
            return [array]
        return array

    def _expand_dims(self, array, num_dim=4, dim=1):
        if array.ndim < num_dim:
            array = np.expand_dims(array, dim)
        return array

    def expand(self, array: jnp.ndarray, index, shape: tuple) -> jnp.ndarray:
        """Creates an a zero array with shape `shape` and fills it with `array` at index."""
        expanded_array = jnp.zeros(shape)
        expanded_array = index_update(expanded_array, index, array)
        return expanded_array

    def pad(self, array: jnp.ndarray, func=jnp.zeros):
        """Adds an additional column to an three dimensional array."""
        return jnp.concatenate(
            [array, func((array.shape[0], *[1 for _ in range(array.ndim - 1)]))], -1
        )

    def get_logits(self, idx, time=Ellipsis):

        logits = self.posterior.dist(Sites.B1, idx) * np.arange(0, self.num_time)[
            self._is_int(time)
        ].reshape(1, -1, 1) + self.posterior.dist(Sites.C1, idx)
        logits = self._expand_dims(logits)
        return logits

    def get_probabilities(self, idx, time=Ellipsis):
        logits = self.get_logits(idx, time=time)
        p = np.exp(logits) / np.exp(logsumexp(logits, -1, keepdims=True))
        return p

    def get_lambda(self, idx, time=Ellipsis):
        beta = self.posterior.dist(Sites.BETA1, idx)
        beta = self._expand_dims(beta, 3)

        lamb = self.population[idx].reshape(1, -1, 1) * np.exp(
            np.einsum("ijk,kl->ijl", beta, self.B[0][self._is_int(time)].T)
        )
        lamb = self._expand_dims(lamb, dim=-1)
        return lamb

    def get_lambda_lineage(self, idx, time=Ellipsis):
        return self.get_lambda(idx, time=time) * self.get_probabilities(idx, time=time)

    def get_R(self, idx, time=Ellipsis):
        p = self.get_probabilities(idx, time=time)
        b1 = self._expand_dims(self.posterior.dist(Sites.B1, idx), dim=2)
        beta = self.posterior.dist(Sites.BETA1, idx)
        logR = self._expand_dims(
            self._expand_dims(beta @ self.B[1][self._is_int(time)].T, num_dim=3, dim=1),
            num_dim=4,
            dim=-1,
        )

        return jnp.exp(((logR - (np.einsum("mijk,milk->mijl", p, b1))) + b1) * self.tau)

    def get_transmissibility(self):
        return np.exp(self.posterior.dist(Sites.B0) * self.tau)

    def aggregate_lambda(self, region, time=Ellipsis):
        agg = []
        for i in np.sort(np.unique(region)):
            region_idx = np.where(region == i)[0]
            region_not_nan = np.isin(region_idx, self.nan_idx)
            agg.append(
                self.get_lambda_lineage(region_idx[region_not_nan], time=time).sum(1)
            )

        return np.stack(agg, 1)

    def aggregate_R(self, region, time=Ellipsis):
        lambda_regions = self.aggregate_lambda(region, time=time)

        agg = []
        for i in np.sort(np.unique(region)):
            region_idx = np.where(region == i)[0]
            region_not_nan = np.isin(region_idx, self.nan_idx)
            agg.append(
                (
                    np.log(self.get_R(region_idx[region_not_nan], time=time))
                    * self.get_lambda_lineage(region_idx[region_not_nan], time=time)
                ).sum(1)
            )

        return np.exp(np.stack(agg, 1) / lambda_regions)


class MultiLineageArma(Model, Lineage):
    """
    WARNING: EXPERIMENTAL - The interface may change in future

    :param cases: A two-dimensional array with cases counts (locations, time)
    :param lineages: A three dimensional array containing the lineage the counts
        for each lineages.shape = (location, lineage_time, lineages)
    :param lineages_date:
    :param population: An array indicating the population in each location.
    :param regions: A list of index arrays indicating higher order regions.
    :param basis: The spline basis function an its derivative (2, time, num_basis).
    :param tau: Generation time in days.
    :param init_scale: Scaling factor of variational distributions
    :param beta_loc: Mean of the spline regression coefficients.
    :param beta_scale: Standard deviation of the spline regression coefficents.
    :param b0_scale: Standard deviation of the lineage transmissibility parameter.
    :param b_scale: Standard deviation of location specific lineage transmissibility.
    :param c_scale: Standard deviation of the lineage/location specific offset parameter.
    :param fit_rho: Fit the overdispersion in each location or use a fixed value.
    :param rho_loc: Mean of the overdispersion parameter, defaults to np.log(10).
    :param rho_scale: Standard deviation of the overdispersion parameter.
    :param multinomial_scale: Weight of the multinomial log likelihood.
    :param time_scale: Parameter to scale the variance of mu_b.
    :param exclude: Exclude missing lineages during the analysis
    :kwargs: SVI Handler arguments.
    """

    _latent_variables = [
        Sites.BETA1,
        Sites.BC0,
        Sites.B1,
        Sites.C1,
    ]

    def __init__(
        self,
        cases: np.ndarray,
        lineages: np.ndarray,
        lineage_dates: np.ndarray,
        population: np.ndarray,
        basis=None,
        tau: float = 5.0,
        init_scale: float = 0.1,
        beta_loc: float = -10.0,
        beta_scale: float = 5.0,
        b0_scale: float = 0.2,
        c0_loc: float = -10.0,
        c0_scale: float = 5.0,
        c_scale: float = 10.0,
        fit_rho: bool = False,
        rho_loc: float = np.log(10.0),
        rho_scale: float = 1.0,
        time_scale: float = 100.0,
        auto_correlation: float = 0.5,
        sample_deterministic: bool = False,
        *args,
        **kwargs,
    ):
        """
        Constructor.
        """
        # TODO: More sanity checks
        assert (
            cases.shape[0] == lineages.shape[0]
        ), "cases and lineages must have the number of location"
        super().__init__(**kwargs)
        self.cases = cases
        self.lineages = lineages
        self.lineage_dates = lineage_dates
        self.population = population

        self.tau = tau
        self.init_scale = init_scale
        self.fit_rho = fit_rho

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.b0_loc = 0.0
        self.b0_scale = b0_scale
        self.c0_loc = c0_loc
        self.c0_scale = c0_scale

        self.c_loc = 0.0
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale

        self.time_scale = time_scale
        self.auto_correlation = auto_correlation

        self.nan_idx = self._nan_idx(cases, lineages)
        self.missing_lineages = self._missing_lineages(lineages)[self.nan_idx].astype(
            int
        )
        self.sample_deterministic = sample_deterministic

        if basis is None:
            _, self.B = create_spline_basis(
                np.arange(cases.shape[1]),
                num_knots=int(np.ceil(cases.shape[1] / 10)),
                add_intercept=False,
            )
        else:
            self.B = basis

        self.num_ltla = self.cases.shape[0]
        self.num_time = self.cases.shape[1]
        self.num_lin = self.lineages.shape[-1] - 1
        self.num_basis = self.B.shape[-1]
        self.num_ltla_lin = self.nan_idx.shape[0]

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

    def get_lambda(self, idx, time=Ellipsis):
        beta = self._expand_dims(self.posterior.dist(Sites.BETA1, idx), 3)

        lamb = self.population[idx].reshape(1, -1, 1) * np.exp(
            np.einsum("ijk,kl->ijl", beta, self.B[0][time].T) + self.beta_loc
        )
        lamb = self._expand_dims(lamb, dim=-1)
        return lamb

    def model(self):
        """The model."""

        plate_ltla = npy.plate("ltla", self.num_ltla, dim=-2)
        # dispersion parameter for lads
        if self.fit_rho:
            with plate_ltla:
                rho = npy.sample(Sites.RHO, dist.Normal(self.rho_loc, self.rho_scale))
        else:
            with plate_ltla:
                rho = self.rho_loc

        # Regression coefficients (num_location x self.num_basis)
        beta = npy.sample(
            Sites.BETA,
            dist.MultivariateNormal(
                0.0,  # self.beta_loc,
                jnp.tile(self.beta_scale, (self.num_ltla_lin, self.num_basis, 1))
                * jnp.eye(self.num_basis).reshape(1, self.num_basis, self.num_basis),
            ),
        )

        beta = npy.deterministic(
            Sites.BETA1,
            self.expand(beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis))
            @ self.arma,
        )

        # MVN prior for b and c parameter
        bc0_loc = jnp.concatenate(
            [
                jnp.repeat(self.b0_loc, self.num_lin),
                jnp.repeat(self.c0_loc, self.num_lin),
            ]
        )
        bc0_scale = jnp.diag(
            jnp.concatenate(
                [
                    self.time_scale * jnp.repeat(self.b0_scale, self.num_lin),
                    jnp.repeat(self.c0_scale, self.num_lin),
                ]
            )
        )
        bc0 = npy.sample(Sites.BC0, dist.MultivariateNormal(bc0_loc, bc0_scale))

        # split the array
        b = bc0[: self.num_lin] / self.time_scale

        # sample non-centered c
        c0 = bc0[self.num_lin :]
        c_offset = npy.sample(
            Sites.C,
            dist.Normal(
                jnp.tile(self.c_loc, (self.num_ltla_lin, self.num_lin)),
                jnp.tile(self.c_scale, (self.num_ltla_lin, self.num_lin)),
            ),
        )
        c = c_offset + c0

        b1 = npy.deterministic(
            Sites.B1,
            self.pad(
                self.expand(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self.pad(
                self.expand(
                    c.reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )

        # Lineage specific regression coefficients (self.num_ltla x self.num_basis x self.num_lin)
        logits = b1 * jnp.arange(self.num_time).reshape(1, -1, 1) + c1
        p = jnp.exp(logits) / jnp.exp(logsumexp(logits, -1, keepdims=True))

        mu = jnp.exp((beta @ self.B[0].T) + self.beta_loc)
        lamb = npy.deterministic(
            Sites.LAMBDA_LINEAGE, self.population.reshape(-1, 1) * mu
        )

        npy.sample(
            Sites.CASES,
            NegativeBinomial(lamb[self.nan_idx], jnp.exp(rho)),
            obs=self.cases[self.nan_idx],
        )

        # with lineage_context:
        npy.sample(
            Sites.LINEAGE,
            dist.MultinomialProbs(
                p[self.nan_idx][:, self.lineage_dates],
                total_count=self.lineages[self.nan_idx].sum(-1),
            ),
            obs=self.lineages[self.nan_idx],
        )
        if self.sample_deterministic:
            npy.deterministic(Sites.LAMBDA, lamb)
            npy.deterministic(Sites.P, p)
            npy.deterministic(
                Sites.LAMBDA_LINEAGE,
                self.population.reshape(-1, 1, 1) * mu[..., jnp.newaxis] * p,
            )
            npy.deterministic(
                Sites.R,
                jnp.exp(
                    (
                        (
                            beta @ self.B[1].T
                            - jnp.einsum("ijk,ik->ij", p, b1.squeeze())
                        )[..., jnp.newaxis]
                        + b1
                    )
                    * self.tau
                ),
            )

    def guide(self):
        if self.fit_rho:
            rho_loc = npy.param(
                Sites.RHO + Sites.LOC,
                self.rho_loc * jnp.ones((self.num_ltla, 1)),
            )
            rho_scale = npy.param(
                Sites.RHO + Sites.SCALE,
                self.init_scale * self.rho_scale * jnp.ones((self.num_ltla, 1)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.RHO, dist.Normal(rho_loc, rho_scale))

        # mean / sd for parameter s
        beta_loc = npy.param(
            Sites.BETA + Sites.LOC,
            jnp.tile(0.0, (self.num_ltla_lin, self.num_basis)),  # used to self.beta_loc
        )
        beta_scale = npy.param(
            Sites.BETA + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(self.num_ltla_lin * [jnp.eye(self.num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        npy.sample(
            Sites.BETA, dist.MultivariateNormal(beta_loc, scale_tril=beta_scale)
        )  # cov

        bc0_loc = npy.param(
            Sites.BC0 + Sites.LOC,
            jnp.concatenate(
                [
                    self.b0_loc * jnp.ones(self.num_lin),
                    self.c0_loc * jnp.ones(self.num_lin),
                ]
            ),
        )
        bc0_scale = npy.param(
            Sites.BC0 + Sites.SCALE,
            jnp.diag(
                jnp.concatenate(
                    [
                        self.init_scale
                        * self.b0_scale
                        * self.time_scale
                        * jnp.ones(self.num_lin),
                        self.init_scale * self.c0_scale * jnp.ones(self.num_lin),
                    ]
                )
            ).reshape(2 * self.num_lin, 2 * self.num_lin),
            constraint=dist.constraints.lower_cholesky,
        )
        npy.sample(Sites.BC0, dist.MultivariateNormal(bc0_loc, scale_tril=bc0_scale))

        c_loc = npy.param(
            Sites.C + Sites.LOC,
            self.c_loc * jnp.ones((self.num_ltla_lin, self.num_lin)),
        )
        c_scale = npy.param(
            Sites.C + Sites.SCALE,
            self.init_scale
            * self.c_scale
            * jnp.ones((self.num_ltla_lin, self.num_lin)),
            constraint=dist.constraints.positive,
        )
        npy.sample(Sites.C, dist.Normal(c_loc, c_scale))

    def deterministic(self):
        """
        Performs post processing steps
        """
        if Sites.BC0 in self.posterior.keys():
            self.posterior[Sites.B0] = (
                self.posterior[Sites.BC0][:, : self.num_lin] / self.time_scale
            )


class MultiLineage(Model, Lineage):
    """
    WARNING: EXPERIMENTAL - The interface may change in future

    :param cases: A two-dimensional array with cases counts (locations, time)
    :param lineages: A three dimensional array containing the lineage the counts
        for each lineages.shape = (location, lineage_time, lineages)
    :param lineages_date:
    :param population: An array indicating the population in each location.
    :param regions: A list of index arrays indicating higher order regions.
    :param basis: The spline basis function an its derivative (2, time, num_basis).
    :param tau: Generation time in days.
    :param init_scale: Scaling factor of variational distributions
    :param beta_loc: Mean of the spline regression coefficients.
    :param beta_scale: Standard deviation of the spline regression coefficents.
    :param b0_scale: Standard deviation of the lineage transmissibility parameter.
    :param b_scale: Standard deviation of location specific lineage transmissibility.
    :param c_scale: Standard deviation of the lineage/location specific offset parameter.
    :param fit_rho: Fit the overdispersion in each location or use a fixed value.
    :param rho_loc: Mean of the overdispersion parameter, defaults to np.log(10).
    :param rho_scale: Standard deviation of the overdispersion parameter.
    :param multinomial_scale: Weight of the multinomial log likelihood.
    :param time_scale: Parameter to scale the variance of mu_b.
    :param exclude: Exclude missing lineages during the analysis
    :kwargs: SVI Handler arguments.
    """

    _latent_variables = [Sites.BETA1, Sites.BC0, Sites.B1, Sites.C1]

    def __init__(
        self,
        cases: np.ndarray,
        lineages: np.ndarray,
        lineage_dates: np.ndarray,
        population: np.ndarray,
        basis=None,
        tau: float = 5.0,
        init_scale: float = 0.1,
        beta_loc: float = -10.0,
        beta_scale: float = 5.0,
        b0_scale: float = 0.2,
        c0_loc: float = -10.0,
        c0_scale: float = 5.0,
        c_scale: float = 10.0,
        fit_rho: bool = False,
        rho_loc: float = np.log(10.0),
        rho_scale: float = 1.0,
        time_scale: float = 100.0,
        sample_deterministic: bool = False,
        *args,
        **kwargs,
    ):
        """
        Constructor.
        """
        # TODO: More sanity checks
        assert (
            cases.shape[0] == lineages.shape[0]
        ), "cases and lineages must have the number of location"
        super().__init__(**kwargs)
        self.cases = cases
        self.lineages = lineages
        self.lineage_dates = lineage_dates
        self.population = population

        self.tau = tau
        self.init_scale = init_scale
        self.fit_rho = fit_rho

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.b0_loc = 0.0
        self.b0_scale = b0_scale
        self.c0_loc = c0_loc
        self.c0_scale = c0_scale

        self.c_loc = 0.0
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale

        self.time_scale = time_scale

        self.nan_idx = self._nan_idx(cases, lineages)
        self.missing_lineages = self._missing_lineages(lineages)[self.nan_idx].astype(
            int
        )
        self.sample_deterministic = sample_deterministic

        if basis is None:
            _, self.B = create_spline_basis(
                np.arange(cases.shape[1]),
                num_knots=int(np.ceil(cases.shape[1] / 10)),
                add_intercept=False,
            )
        else:
            self.B = basis

        self.num_ltla = self.cases.shape[0]
        self.num_time = self.cases.shape[1]
        self.num_lin = self.lineages.shape[-1] - 1
        self.num_basis = self.B.shape[-1]
        self.num_ltla_lin = self.nan_idx.shape[0]

    def model(self):
        """The model."""

        plate_ltla = npy.plate("ltla", self.num_ltla, dim=-2)
        # dispersion parameter for lads
        if self.fit_rho:
            with plate_ltla:
                rho = npy.sample(Sites.RHO, dist.Normal(self.rho_loc, self.rho_scale))
        else:
            with plate_ltla:
                rho = self.rho_loc

        # Regression coefficients (num_location x self.num_basis)
        beta = npy.sample(
            Sites.BETA,
            dist.MultivariateNormal(
                self.beta_loc,
                jnp.tile(self.beta_scale, (self.num_ltla_lin, self.num_basis, 1))
                * jnp.eye(self.num_basis).reshape(1, self.num_basis, self.num_basis),
            ),
        )

        beta = npy.deterministic(
            Sites.BETA1,
            self.expand(beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis)),
        )

        # MVN prior for b and c parameter
        bc0_loc = jnp.concatenate(
            [
                jnp.repeat(self.b0_loc, self.num_lin),
                jnp.repeat(self.c0_loc, self.num_lin),
            ]
        )
        bc0_scale = jnp.diag(
            jnp.concatenate(
                [
                    self.time_scale * jnp.repeat(self.b0_scale, self.num_lin),
                    jnp.repeat(self.c0_scale, self.num_lin),
                ]
            )
        )
        bc0 = npy.sample(Sites.BC0, dist.MultivariateNormal(bc0_loc, bc0_scale))

        # split the array
        b = bc0[: self.num_lin] / self.time_scale

        # sample non-centered c
        c0 = bc0[self.num_lin :]
        c_offset = npy.sample(
            Sites.C,
            dist.Normal(
                jnp.tile(self.c_loc, (self.num_ltla_lin, self.num_lin)),
                jnp.tile(self.c_scale, (self.num_ltla_lin, self.num_lin)),
            ),
        )
        c = c_offset + c0

        b1 = npy.deterministic(
            Sites.B1,
            self.pad(
                self.expand(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self.pad(
                self.expand(
                    c.reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )

        # Lineage specific regression coefficients (self.num_ltla x self.num_basis x self.num_lin)
        logits = b1 * jnp.arange(self.num_time).reshape(1, -1, 1) + c1
        p = jnp.exp(logits) / jnp.exp(logsumexp(logits, -1, keepdims=True))

        mu = jnp.exp(beta @ self.B[0].T)
        lamb = self.population.reshape(-1, 1) * mu

        npy.sample(
            Sites.CASES,
            NegativeBinomial(lamb[self.nan_idx], jnp.exp(rho)),
            obs=self.cases[self.nan_idx],
        )

        # with lineage_context:
        npy.sample(
            Sites.LINEAGE,
            dist.MultinomialProbs(
                p[self.nan_idx][:, self.lineage_dates],
                total_count=self.lineages[self.nan_idx].sum(-1),
            ),
            obs=self.lineages[self.nan_idx],
        )
        if self.sample_deterministic:
            npy.deterministic(Sites.LAMBDA, lamb)
            npy.deterministic(Sites.P, p)
            npy.deterministic(
                Sites.LAMBDA_LINEAGE,
                self.population.reshape(-1, 1, 1) * mu[..., jnp.newaxis] * p,
            )
            npy.deterministic(
                Sites.R,
                jnp.exp(
                    (
                        (
                            beta @ self.B[1].T
                            - jnp.einsum("ijk,ik->ij", p, b1.squeeze())
                        )[..., jnp.newaxis]
                        + b1
                    )
                    * self.tau
                ),
            )

    def guide(self):
        if self.fit_rho:
            rho_loc = npy.param(
                Sites.RHO + Sites.LOC,
                self.rho_loc * jnp.ones((self.num_ltla, 1)),
            )
            rho_scale = npy.param(
                Sites.RHO + Sites.SCALE,
                self.init_scale * self.rho_scale * jnp.ones((self.num_ltla, 1)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.RHO, dist.Normal(rho_loc, rho_scale))

        # mean / sd for parameter s
        beta_loc = npy.param(
            Sites.BETA + Sites.LOC,
            jnp.tile(self.beta_loc, (self.num_ltla_lin, self.num_basis)),
        )
        beta_scale = npy.param(
            Sites.BETA + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(self.num_ltla_lin * [jnp.eye(self.num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        npy.sample(
            Sites.BETA, dist.MultivariateNormal(beta_loc, scale_tril=beta_scale)
        )  # cov

        bc0_loc = npy.param(
            Sites.BC0 + Sites.LOC,
            jnp.concatenate(
                [
                    self.b0_loc * jnp.ones(self.num_lin),
                    self.c0_loc * jnp.ones(self.num_lin),
                ]
            ),
        )
        bc0_scale = npy.param(
            Sites.BC0 + Sites.SCALE,
            jnp.diag(
                jnp.concatenate(
                    [
                        self.init_scale
                        * self.b0_scale
                        * self.time_scale
                        * jnp.ones(self.num_lin),
                        self.init_scale * self.c0_scale * jnp.ones(self.num_lin),
                    ]
                )
            ).reshape(2 * self.num_lin, 2 * self.num_lin),
            constraint=dist.constraints.lower_cholesky,
        )
        npy.sample(Sites.BC0, dist.MultivariateNormal(bc0_loc, scale_tril=bc0_scale))

        c_loc = npy.param(
            Sites.C + Sites.LOC,
            self.c_loc * jnp.ones((self.num_ltla_lin, self.num_lin)),
        )
        c_scale = npy.param(
            Sites.C + Sites.SCALE,
            self.init_scale
            * self.c_scale
            * jnp.ones((self.num_ltla_lin, self.num_lin)),
            constraint=dist.constraints.positive,
        )
        npy.sample(Sites.C, dist.Normal(c_loc, c_scale))

    def deterministic(self):
        """
        Performs post processing steps
        """
        if Sites.BC0 in self.posterior.keys():
            self.posterior[Sites.B0] = (
                self.posterior[Sites.BC0][:, : self.num_lin] / self.time_scale
            )


class SimpleMultiLineage(Model, Lineage):
    """
    WARNING: EXPERIMENTAL - The interface may change in future

    :param cases: A two-dimensional array with cases counts (locations, time)
    :param lineages: A three dimensional array containing the lineage the counts
        for each lineages.shape = (location, lineage_time, lineages)
    :param lineages_date:
    :param population: An array indicating the population in each location.
    :param regions: A list of index arrays indicating higher order regions.
    :param basis: The spline basis function an its derivative (2, time, self.num_basis).
    :param tau: Generation time in days.
    :param init_scale: Scaling factor of variational distributions
    :param beta_loc: Mean of the spline regression coefficients.
    :param beta_scale: Standard deviation of the spline regression coefficents.
    :param b0_scale: Standard deviation of the lineage transmissibility parameter.
    :param b_scale: Standard deviation of location specific lineage transmissibility.
    :param c_scale: Standard deviation of the lineage/location specific offset parameter.
    :param fit_rho: Fit the overdispersion in each location or use a fixed value.
    :param rho_loc: Mean of the overdispersion parameter, defaults to np.log(10).
    :param rho_scale: Standard deviation of the overdispersion parameter.
    :param multinomial_scale: Weight of the multinomial log likelihood.
    :param time_scale: Parameter to scale the variance of b0.
    :param exclude: Exclude missing lineages during the analysis
    :kwargs: SVI Handler arguments.
    """

    _latent_variables = [Sites.B1, Sites.C1, Sites.BETA1]

    def __init__(
        self,
        cases: np.ndarray,
        lineages: np.ndarray,
        lineage_dates: np.ndarray,
        population: np.ndarray,
        basis=None,
        tau: float = 5.0,
        init_scale: float = 0.1,
        beta_loc: float = -10.0,
        beta_scale: float = 5.0,
        b0_scale: float = 0.2,
        c_loc: float = -15.0,
        c_scale: float = 10.0,
        fit_rho: bool = False,
        rho_loc: float = np.log(10.0),
        rho_scale: float = 1.0,
        time_scale: float = 100.0,
        sample_deterministic: bool = False,
        handler: str = "SVI",
        *args,
        **kwargs,
    ):
        """
        Constructor.
        """
        # TODO: More sanity checks
        assert (
            cases.shape[0] == lineages.shape[0]
        ), "cases and lineages must have the number of location"
        super().__init__(handler, *args, **kwargs)
        self.cases = cases
        self.lineages = lineages
        self.lineage_dates = lineage_dates
        self.population = population

        self.tau = tau
        self.init_scale = init_scale
        self.fit_rho = fit_rho

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.b0_loc = 0.0
        self.b0_scale = b0_scale

        self.c_loc = c_loc
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale
        self.time_scale = time_scale

        self.nan_idx = self._nan_idx(cases, lineages)
        self.missing_lineages = self._missing_lineages(lineages)[self.nan_idx].astype(
            int
        )

        self.sample_deterministic = sample_deterministic

        if basis is None:
            _, self.B = create_spline_basis(
                np.arange(cases.shape[1]),
                num_knots=int(np.ceil(cases.shape[1] / 10)),
                add_intercept=False,
            )
        else:
            self.B = basis

        # number
        self.num_ltla = self.cases.shape[0]
        self.num_time = self.cases.shape[1]
        self.num_lin = self.lineages.shape[-1] - 1
        self.num_basis = self.B.shape[-1]
        self.num_ltla_lin = self.nan_idx.shape[0]

    def model(self):
        """The model."""
        plate_ltla = npy.plate("ltla", self.num_ltla, dim=-2)
        # dispersion parameter for lads
        if self.fit_rho:
            with plate_ltla:
                rho = npy.sample(Sites.RHO, dist.Normal(self.rho_loc, self.rho_scale))
        else:
            with plate_ltla:
                rho = self.rho_loc

        # Regression coefficients (num_location x self.num_basis)
        beta = npy.sample(
            Sites.BETA,
            dist.MultivariateNormal(
                self.beta_loc,
                jnp.tile(self.beta_scale, (self.num_ltla_lin, self.num_basis, 1))
                * jnp.eye(self.num_basis).reshape(1, self.num_basis, self.num_basis),
            ),
        )

        beta = npy.deterministic(
            Sites.BETA1,
            self.expand(beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis)),
        )

        # MVN prior for b and c parameter
        b0_loc = jnp.concatenate(
            [
                jnp.repeat(self.b0_loc, self.num_lin),
            ]
        )
        b0_scale = jnp.diag(
            jnp.concatenate(
                [
                    jnp.repeat(self.time_scale * self.b0_scale, self.num_lin),
                ]
            )
        )
        b0 = npy.sample(Sites.B0, dist.MultivariateNormal(b0_loc, b0_scale))

        b = b0 / self.time_scale

        c = npy.sample(
            Sites.C,
            dist.Normal(
                jnp.tile(self.c_loc, (self.num_ltla_lin, self.num_lin)),
                jnp.tile(self.c_scale, (self.num_ltla_lin, self.num_lin)),
            ),
        )

        b1 = npy.deterministic(
            Sites.B1,
            self.pad(
                self.expand(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self.pad(
                self.expand(
                    (self.missing_lineages * c).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )

        # Lineage specific regression coefficients (self.num_ltla x self.num_basis x self.num_lin)
        logits = b1 * jnp.arange(self.num_time).reshape(1, -1, 1) + c1
        p = jnp.exp(logits) / jnp.exp(logsumexp(logits, -1, keepdims=True))

        mu = beta @ self.B[0].T
        lamb = self.population.reshape(-1, 1) * jnp.exp(mu)

        npy.sample(
            Sites.CASES,
            NegativeBinomial(lamb[self.nan_idx], jnp.exp(rho)),
            obs=self.cases[self.nan_idx],
        )

        # with lineage_context:
        npy.sample(
            Sites.LINEAGE,
            dist.MultinomialProbs(
                p[self.nan_idx][:, self.lineage_dates],
                total_count=self.lineages[self.nan_idx].sum(-1),
            ),
            obs=self.lineages[self.nan_idx],
        )

        if self.sample_deterministic:
            npy.deterministic(Sites.LAMBDA, lamb)
            npy.deterministic(Sites.P, p)
            npy.deterministic(
                Sites.LAMBDA_LINEAGE,
                self.population.reshape(-1, 1, 1) * mu[..., jnp.newaxis] * p,
            )
            npy.deterministic(
                Sites.R,
                jnp.exp(
                    (
                        (
                            beta @ self.B[1].T
                            - jnp.einsum("ijk,ik->ij", p, b1.squeeze())
                        )[..., jnp.newaxis]
                        + b1
                    )
                    * self.tau
                ),
            )

    def guide(self):
        if self.fit_rho:
            rho_loc = npy.param(
                Sites.RHO + Sites.LOC,
                jnp.tile(self.rho_loc, (self.num_ltla, 1)),
            )
            rho_scale = npy.param(
                Sites.RHO + Sites.SCALE,
                jnp.tile(self.init_scale * self.rho_scale, (self.num_ltla, 1)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.RHO, dist.Normal(rho_loc, rho_scale))

        # mean / sd for parameter s
        beta_loc = npy.param(
            Sites.BETA + Sites.LOC,
            jnp.tile(self.beta_loc, (self.num_ltla_lin, self.num_basis)),
        )
        beta_scale = npy.param(
            Sites.BETA + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(self.num_ltla_lin * [jnp.eye(self.num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        npy.sample(Sites.BETA, dist.MultivariateNormal(beta_loc, scale_tril=beta_scale))

        b0_loc = npy.param(
            Sites.BC0 + Sites.LOC,
            jnp.concatenate(
                [
                    jnp.repeat(self.b0_loc, self.num_lin),
                ]
            ),
        )
        b0_scale = npy.param(
            Sites.BC0 + Sites.SCALE,
            jnp.diag(
                jnp.concatenate(
                    [
                        jnp.repeat(
                            self.init_scale * self.b0_scale * self.time_scale,
                            self.num_lin,
                        ),
                    ]
                )
            ),
            constraint=dist.constraints.lower_cholesky,
        )
        npy.sample(Sites.B0, dist.MultivariateNormal(b0_loc, scale_tril=b0_scale))

        c_loc = npy.param(
            Sites.C + Sites.LOC, jnp.tile(self.c_loc, (self.num_ltla_lin, self.num_lin))
        )

        c_scale = npy.param(
            Sites.C + Sites.SCALE,
            jnp.tile(self.init_scale * self.c_scale, (self.num_ltla_lin, self.num_lin)),
        )
        npy.sample(Sites.C, dist.Normal(c_loc, c_scale))
