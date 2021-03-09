from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax.ops import index, index_update

from genomicsurveillance.distributions import MultinomialProbs, NegativeBinomial
from genomicsurveillance.handler import SVIModel
from genomicsurveillance.utils import create_spline_basis, is_nan_row


class Sites:
    """
    Helper class to consistently label each sampling site in the model.
    """

    RHO = "rho"

    MU_COUNTRY = "mu_country"
    MU_UTLA = "mu_utla"
    MU_BETA = "mu_beta"
    SIGMA_BETA = "sigma_beta"

    BETA_0 = "beta_0"
    BETA_1 = "beta_1"

    MU_A = "mu_a"
    MU_B = "mu_b"

    A = "a"
    B = "b"
    C = "c"

    A_1 = "a_1"
    B_1 = "b_1"
    C_1 = "c_1"

    MU_1 = "mu_lin"
    MU = "mu"

    LAMB_1 = "lamb_lin"
    LAMB = "lamb"

    R_1 = "R_1"
    P = "p"

    SPECIMEN = "specimen"
    LINEAGE = "lineage"

    LOC = "_loc"
    SCALE = "_scale"


class IndependentMultiLineage(SVIModel):
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
    :param mu_b_scale: Standard deviation of the lineage transmissibility parameter.
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

    def __init__(
        self,
        cases: np.ndarray,
        lineages: np.ndarray,
        lineage_dates: np.ndarray,
        population: np.ndarray,
        regions=None,
        basis=None,
        tau: float = 5.0,
        init_scale: float = 0.1,
        beta_loc: float = -10.0,
        beta_scale: float = 5.0,
        mu_b_scale: float = np.log(2) / 5,
        b_scale: float = 0.01,
        c_scale: float = 5.0,
        fit_rho: bool = False,
        rho_loc: float = np.log(10.0),
        rho_scale: float = 1.0,
        multinomial_scale: float = 1.0,
        time_scale: float = 100.0,
        exclude: bool = True,
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
        self.regions = regions

        self.tau = tau
        self.init_scale = init_scale
        self.fit_rho = fit_rho
        self.fit_mu_b_mvn = False

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.mu_b_loc = 0.0
        self.mu_b_scale = mu_b_scale

        self.b_loc = 0.0
        self.b_scale = b_scale
        self.c_loc = -10.0
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale
        self.exclude = exclude

        self.multinomial_scale = multinomial_scale
        self.time_scale = time_scale

        if basis is None:
            _, self.B = create_spline_basis(
                np.arange(cases.shape[1]),
                num_knots=int(np.ceil(cases.shape[1] / 10)),
                add_intercept=False,
            )
        else:
            self.B = basis

    @property
    @lru_cache()
    def _nan_idx(self):
        exclude = list(set(is_nan_row(self.lineages)) | set(is_nan_row(self.cases)))
        return np.array([i for i in range(self.cases.shape[0]) if i not in exclude])

    @property
    @lru_cache()
    def _missing_lineages(self):
        return (self.lineages[..., :-1].sum(1) != 0)[self._nan_idx].astype(int)

    @property
    @lru_cache()
    def _missing_lineage_obs(self):
        return jnp.repeat(
            (self.lineages.sum(1) != 0).reshape(
                self.lineages.shape[0], 1, self.lineages.shape[-1]
            ),
            self.cases.shape[1],
            1,
        ).astype(int)

    def _expand(self, array: jnp.ndarray, index, shape: tuple) -> jnp.ndarray:
        """Creates an a zero array with shape `shape` and fills it with `array` at index."""
        expanded_array = jnp.zeros(shape)
        expanded_array = index_update(expanded_array, index, array)
        return expanded_array

    def _pad(self, array: jnp.ndarray, func=jnp.zeros):
        """Adds an additional column to an three dimensional array."""
        return jnp.concatenate([array, func((array.shape[0], 1, 1))], -1)

    def aggregate_guide(self, i, array):
        num_regions = len(np.unique(array))
        num_basis = self.B.shape[-1]
        # mean / sd for parameter s
        beta_i_loc = npy.param(
            f"beta_{i}" + Sites.LOC, self.beta_loc * jnp.ones((num_regions, num_basis))
        )
        beta_i_scale = npy.param(
            f"beta_{i}" + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(num_regions * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        npy.sample(
            f"beta_{i}", dist.MultivariateNormal(beta_i_loc, scale_tril=beta_i_scale)
        )  # cov

    def aggregate_model(self, i, array):
        num_regions = len(np.unique(array))
        num_basis = self.B.shape[-1]
        rho = self.rho_loc
        # Regression coefficients (num_location x num_basis)
        beta_i = npy.sample(
            f"beta_{i}",
            dist.MultivariateNormal(
                self.beta_loc,
                jnp.tile(self.beta_scale, (num_regions, num_basis, 1))
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )

        N_i = jnp.array(
            [self.population[array == j].sum(0) for j in range(num_regions)]
        )
        f_i_lin = jnp.stack(
            [jnp.nansum(self.f_lin[array == j], 0) for j in range(num_regions)], 0
        )
        f_i = f_i_lin.sum(-1, keepdims=True)
        p_i = npy.deterministic(f"p_{i}", f_i_lin / f_i)
        l_i = jnp.stack(
            [jnp.nansum(self.lineages[array == j], 0) for j in range(num_regions)], 0
        )

        g_i = jnp.exp(beta_i @ self.B[0].T)
        mu_i = g_i * f_i.squeeze()

        npy.deterministic(
            f"lamb_{i}_lin", N_i.reshape(-1, 1, 1) * g_i[..., jnp.newaxis] * f_i_lin
        )
        lamb_i = npy.deterministic(f"lamb_{i}", N_i.reshape(-1, 1) * mu_i)

        npy.sample(
            f"lineage_{i}",
            MultinomialProbs(
                p_i[:, self.lineage_dates],
                total_count=l_i.sum(-1),
                scale=self.multinomial_scale,
            ),
            obs=l_i,
        )

        specimen_i = jnp.stack(
            [jnp.nansum(self.cases[array == j], 0) for j in range(num_regions)], 0
        )

        npy.sample(
            f"specimen_{i}",
            NegativeBinomial(lamb_i, jnp.exp(rho)),  # jnp.clip(, 1e-6, 1e6)
            obs=specimen_i,
        )

    def model(self):
        """The model."""
        num_ltla = self.cases.shape[0]
        num_time = self.cases.shape[1]
        num_lin = self.lineages.shape[-1] - 1
        num_basis = self.B.shape[-1]
        num_ltla_lin = self._nan_idx.shape[0]

        plate_ltla = npy.plate("ltla", num_ltla, dim=-2)
        plate_lin = npy.plate("lin", num_lin, dim=-1)
        # dispersion parameter for lads
        if self.fit_rho:
            with plate_ltla:
                rho = npy.sample(Sites.RHO, dist.Normal(self.rho_loc, self.rho_scale))
        else:
            with plate_ltla:
                rho = self.rho_loc

        # Regression coefficients (num_location x num_basis)
        beta_0 = npy.sample(
            Sites.BETA_0,
            dist.MultivariateNormal(
                self.beta_loc,
                jnp.tile(self.beta_scale, (num_ltla_lin, num_basis, 1))
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )

        beta_0 = self._expand(beta_0, index[self._nan_idx, :], (num_ltla, num_basis))

        # lineage priors
        with plate_lin:
            if self.fit_mu_b_mvn:
                mu_b = npy.sample(
                    Sites.MU_B,
                    dist.MultivariateNormal(
                        self.mu_b_loc, self.mu_b_scale * jnp.eye(num_lin)
                    ),
                )
            else:
                mu_b = npy.sample(
                    Sites.MU_B,
                    dist.Normal(
                        self.time_scale * self.mu_b_loc,
                        self.time_scale * self.mu_b_scale,
                    ),
                )

        mu_bc = jnp.concatenate(
            [
                self._missing_lineages
                * jnp.repeat(mu_b.reshape(1, -1), num_ltla_lin, 0),
                jnp.repeat(
                    jnp.repeat(self.c_loc, num_lin).reshape(1, -1), num_ltla_lin, 0
                ),
            ],
            -1,
        )

        sd_bc = jnp.repeat(
            jnp.diag(
                jnp.concatenate(
                    [
                        self.time_scale * jnp.repeat(self.b_scale, num_lin),
                        jnp.repeat(self.c_scale, num_lin),
                    ]
                )
            ).reshape(1, 2 * num_lin, 2 * num_lin),
            num_ltla_lin,
            0,
        )
        bc = npy.sample("bc", dist.MultivariateNormal(mu_bc, sd_bc)).reshape(
            num_ltla_lin, 2, num_lin
        )

        # pad lineage parameters b, c to match the full size array
        #         b_1 = self._pad(
        #             Sites.B_1,
        #             bc[:, 0].reshape(num_ltla_lin, 1, -1) / self.time_scale,
        #             index[self._nan_idx, :, :],
        #             (num_ltla, 1, num_lin),
        #             jnp.zeros,
        #         )
        #         c_1 = self._pad(
        #             Sites.C_1,
        #             bc[:, 1].reshape(num_ltla_lin, 1, -1),
        #             index[self._nan_idx, :, :],
        #             (num_ltla, 1, num_lin),
        #             jnp.zeros,
        #         )

        b_1 = self._pad(
            self._expand(
                bc[:, 0].reshape(num_ltla_lin, 1, -1) / self.time_scale,
                index[self._nan_idx, :, :],
                (num_ltla, 1, num_lin),
            )
        )
        c_1 = self._pad(
            self._expand(
                bc[:, 1].reshape(num_ltla_lin, 1, -1),
                index[self._nan_idx, :, :],
                (num_ltla, 1, num_lin),
            )
        )

        # Lineage specific regression coefficients (num_ltla x num_basis x num_lin)
        self.f_lin = jnp.exp(b_1 * jnp.arange(num_time).reshape(1, -1, 1) + c_1)
        f = self.f_lin.sum(-1, keepdims=True)

        if self.exclude:
            p = npy.deterministic(Sites.P, self._missing_lineage_obs * (self.f_lin / f))
        else:
            p = npy.deterministic(Sites.P, self.f_lin / f)

        g = jnp.exp(beta_0 @ self.B[0].T)
        mu = g * f.squeeze()

        lamb = npy.deterministic(Sites.LAMB, self.population.reshape(-1, 1) * mu)
        npy.sample(
            Sites.SPECIMEN,
            NegativeBinomial(
                lamb[self._nan_idx], jnp.exp(rho)
            ),  # jnp.clip(, 1e-6, 1e6)
            obs=self.cases[self._nan_idx],
        )

        # with lineage_context:
        npy.sample(
            Sites.LINEAGE,
            MultinomialProbs(
                p[self._nan_idx][:, self.lineage_dates],
                total_count=self.lineages[self._nan_idx].sum(-1),
                scale=self.multinomial_scale,
            ),
            obs=self.lineages[self._nan_idx],
        )

        if self.regions is not None:
            for i, array in enumerate(self.regions):
                self.aggregate_model(i + 1, array)

        # deterministic sites
        npy.deterministic(
            Sites.LAMB_1,
            self.population.reshape(-1, 1, 1) * g[..., jnp.newaxis] * self.f_lin,
        )
        npy.deterministic("G", beta_0 @ self.B[1].T)
        npy.deterministic(
            Sites.R_1,
            jnp.exp(((beta_0 @ self.B[1].T)[..., jnp.newaxis] + b_1) * self.tau),
        )
        npy.deterministic("sa", jnp.exp(mu_b / self.time_scale * self.tau))

    def guide(self):
        num_ltla = self.cases.shape[0]
        num_lin = self.lineages.shape[-1] - 1
        num_basis = self.B.shape[-1]
        num_ltla_lin = self._nan_idx.shape[0]

        if self.fit_rho:
            rho_loc = npy.param(
                Sites.RHO + Sites.LOC,
                self.rho_loc * jnp.ones((num_ltla, 1)),
            )
            rho_scale = npy.param(
                Sites.RHO + Sites.SCALE,
                self.init_scale * self.rho_scale * jnp.ones((num_ltla, 1)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.RHO, dist.Normal(rho_loc, rho_scale))

        # mean / sd for parameter s
        beta_0_loc = npy.param(
            Sites.BETA_0 + Sites.LOC,
            self.beta_loc * jnp.ones((num_ltla_lin, num_basis)),
        )
        beta_0_scale = npy.param(
            Sites.BETA_0 + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(num_ltla_lin * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        npy.sample(
            Sites.BETA_0, dist.MultivariateNormal(beta_0_loc, scale_tril=beta_0_scale)
        )  # cov

        if self.regions is not None:
            for i, array in enumerate(self.regions):
                self.aggregate_guide(i + 1, array)

        # mean / sd for parameter s
        #         beta_1_loc = npy.param(
        #             Sites.BETA_1 + Sites.LOC, self.beta_loc * jnp.ones((num_countries, num_basis))
        #         )
        #         beta_1_scale = npy.param(
        #             Sites.BETA_1 + Sites.SCALE,
        #             self.init_scale
        #             * self.beta_scale
        #             * jnp.stack(num_countries * [jnp.eye(num_basis)]),
        #             constraint=dist.constraints.lower_cholesky,
        #         )

        #         # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        #         beta_1 = npy.sample(
        #             Sites.BETA_1, dist.MultivariateNormal(beta_1_loc, scale_tril=beta_1_scale)
        #         )  # cov

        mu_b_loc = npy.param(Sites.MU_B + Sites.LOC, jnp.repeat(self.mu_b_loc, num_lin))

        if self.fit_mu_b_mvn:
            mu_b_scale = npy.param(
                Sites.MU_B + Sites.SCALE,
                jnp.diag(self.init_scale * self.mu_b_scale * jnp.ones(num_lin)),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample(
                Sites.MU_B, dist.MultivariateNormal(mu_b_loc, scale_tril=mu_b_scale)
            )
        else:
            mu_b_scale = npy.param(
                Sites.MU_B + Sites.SCALE,
                self.init_scale * self.mu_b_scale * self.time_scale * jnp.ones(num_lin),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.MU_B, dist.Normal(mu_b_loc, mu_b_scale))

        bc_loc = npy.param(
            "bc_loc",
            jnp.repeat(
                jnp.concatenate(
                    [self.b_loc * jnp.ones(num_lin), self.c_loc * jnp.ones(num_lin)]
                ).reshape(1, -1),
                num_ltla_lin,
                0,
            ),
        )
        bc_scale = npy.param(
            "bc_scale",
            jnp.repeat(
                jnp.diag(
                    jnp.concatenate(
                        [
                            self.init_scale
                            * self.b_scale
                            * self.time_scale
                            * jnp.ones(num_lin),
                            self.init_scale * self.c_scale * jnp.ones(num_lin),
                        ]
                    )
                ).reshape(1, 2 * num_lin, 2 * num_lin),
                num_ltla_lin,
                0,
            ),
            constraint=dist.constraints.lower_cholesky,
        )

        npy.sample("bc", dist.MultivariateNormal(bc_loc, scale_tril=bc_scale))
