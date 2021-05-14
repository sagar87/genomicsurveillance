import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax.ops import index
from jax.scipy.special import logsumexp

from genomicsurveillance.distributions import NegativeBinomial
from genomicsurveillance.handler import Model
from genomicsurveillance.utils import create_spline_basis

from .base import Lineage
from .sites import Sites


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
        beta_scale: float = 1.0,
        b0_loc: float = 0.0,
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
        posterior=None,
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
        Model.__init__(self, **kwargs)
        Lineage.__init__(
            self, tau, cases, lineages, lineage_dates, population, basis, posterior
        )
        # super().__init__(**kwargs)

        self.init_scale = init_scale
        self.fit_rho = fit_rho

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.b0_loc = b0_loc
        self.b0_scale = b0_scale
        self.c0_loc = c0_loc
        self.c0_scale = c0_scale

        self.c_loc = 0.0
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale

        self.time_scale = time_scale

        self.sample_deterministic = sample_deterministic

        self.num_ltla = self.cases.shape[0]
        self.num_time = self.cases.shape[1]
        self.num_lin = self.lineages.shape[-1] - 1
        self.num_basis = self.B.shape[-1]
        self.num_ltla_lin = self.nan_idx.shape[0]

    def get_lambda(self, idx, time=Ellipsis):
        beta = self.expand_dims(self.posterior.dist(Sites.BETA1, idx), 3)

        lamb = self.population[idx].reshape(1, -1, 1) * np.exp(
            np.einsum("ijk,kl->ijl", beta, self.B[0][time].T) + self.beta_loc
        )
        lamb = self.expand_dims(lamb, dim=-1)
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
            self._expand_array(
                beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis)
            )
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
            self._pad_array(
                self._expand_array(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self._pad_array(
                self._expand_array(
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
            self._expand_array(
                beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis)
            ),
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
            self._pad_array(
                self._expand_array(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self._pad_array(
                self._expand_array(
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
            self._expand_array(
                beta, index[self.nan_idx, :], (self.num_ltla, self.num_basis)
            ),
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
            self._pad_array(
                self._expand_array(
                    (self.missing_lineages * b).reshape(self.num_ltla_lin, 1, -1),
                    index[self.nan_idx, :, :],
                    (self.num_ltla, 1, self.num_lin),
                )
            ),
        )
        c1 = npy.deterministic(
            Sites.C1,
            self._pad_array(
                self._expand_array(
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
