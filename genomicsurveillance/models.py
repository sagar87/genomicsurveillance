import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax.ops import index, index_update
from jax.scipy.special import logsumexp

from genomicsurveillance.distributions import MultinomialProbs, NegativeBinomial
from genomicsurveillance.handler import SVIModel
from genomicsurveillance.utils import create_spline_basis, is_nan_row


def nan_idx(cases, lineages):
    exclude = list(set(is_nan_row(lineages)) | set(is_nan_row(cases)))
    return np.array([i for i in range(cases.shape[0]) if i not in exclude])


def missing_lineages(lineages):
    return lineages[..., :-1].sum(1) != 0


def expand(array: jnp.ndarray, index, shape: tuple) -> jnp.ndarray:
    """Creates an a zero array with shape `shape` and fills it with `array` at index."""
    expanded_array = jnp.zeros(shape)
    expanded_array = index_update(expanded_array, index, array)
    return expanded_array


def pad(array: jnp.ndarray, func=jnp.zeros):
    """Adds an additional column to an three dimensional array."""
    return jnp.concatenate([array, func((array.shape[0], 1, 1))], -1)


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
    MU_BC = "mu_bc"

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


class MultiLineage(SVIModel):
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
        mu_b_scale: float = 0.2,
        mu_c_loc: float = -10.0,
        mu_c_scale: float = 5.0,
        c_loc: float = 0.0,
        c_scale: float = 10.0,
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

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale

        self.mu_b_loc = 0.0
        self.mu_b_scale = mu_b_scale
        self.mu_c_loc = mu_c_loc
        self.mu_c_scale = mu_c_scale

        self.c_loc = c_loc
        self.c_scale = c_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale
        self.exclude = exclude

        self.multinomial_scale = multinomial_scale
        self.time_scale = time_scale

        self._nan_idx = nan_idx(cases, lineages)
        self._missing_lineages = missing_lineages(lineages)[self._nan_idx].astype(int)

        if basis is None:
            _, self.B = create_spline_basis(
                np.arange(cases.shape[1]),
                num_knots=int(np.ceil(cases.shape[1] / 10)),
                add_intercept=False,
            )
        else:
            self.B = basis

    def model(self):
        """The model."""
        num_ltla = self.cases.shape[0]
        num_time = self.cases.shape[1]
        num_lin = self.lineages.shape[-1] - 1
        num_basis = self.B.shape[-1]
        num_ltla_lin = self._nan_idx.shape[0]

        plate_ltla = npy.plate("ltla", num_ltla, dim=-2)
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

        beta_0 = expand(beta_0, index[self._nan_idx, :], (num_ltla, num_basis))

        # MVN prior for b and c parameter
        mu_bc_loc = jnp.concatenate(
            [jnp.repeat(self.mu_b_loc, num_lin), jnp.repeat(self.mu_c_loc, num_lin)]
        )
        sd_bc_scale = jnp.diag(
            jnp.concatenate(
                [
                    jnp.repeat(self.time_scale * self.mu_b_scale, num_lin),
                    jnp.repeat(self.mu_c_scale, num_lin),
                ]
            )
        )
        mu_bc = npy.sample(Sites.MU_BC, dist.MultivariateNormal(mu_bc_loc, sd_bc_scale))

        # split the array
        b = mu_bc[:num_lin] / self.time_scale

        # offset terms
        mu_c = mu_bc[num_lin:]
        c_offset = npy.sample(
            Sites.C,
            dist.Normal(
                jnp.tile(self.c_loc, (num_ltla_lin, num_lin)),
                jnp.tile(self.c_scale, (num_ltla_lin, num_lin)),
            ),
        )

        c = c_offset + mu_c

        b_1 = pad(
            expand(
                (self._missing_lineages * b).reshape(num_ltla_lin, 1, -1),
                index[self._nan_idx, :, :],
                (num_ltla, 1, num_lin),
            )
        )
        c_1 = pad(
            expand(
                (self._missing_lineages * c).reshape(num_ltla_lin, 1, -1),
                index[self._nan_idx, :, :],
                (num_ltla, 1, num_lin),
            )
        )

        # Lineage specific regression coefficients (num_ltla x num_basis x num_lin)
        logits = b_1 * jnp.arange(num_time).reshape(1, -1, 1) + c_1
        p = jnp.exp(logits) / jnp.exp(logsumexp(logits, -1, keepdims=True))

        mu = beta_0 @ self.B[0].T
        lamb = self.population.reshape(-1, 1) * jnp.exp(mu)

        npy.sample(
            Sites.SPECIMEN,
            NegativeBinomial(lamb[self._nan_idx], jnp.exp(rho)),
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
            lamb_lin = (
                self.population.reshape(-1, 1, 1) * jnp.exp(mu[..., jnp.newaxis]) * p
            )
            for i, region in enumerate(self.regions):
                reg = np.array(
                    [region[self._nan_idx] == i for i in np.unique(region)]
                ).astype(
                    "float"
                )  # Indicator of regions

                lamb_reg = jnp.einsum(
                    "ij,jkl->ikl", reg, lamb_lin[self._nan_idx][:, self.lineage_dates]
                )
                npy.deterministic(
                    Sites.P + f"_{i}", lamb_reg / lamb_reg.sum(-1, keepdims=True)
                )

        npy.deterministic("sa", jnp.exp(b * self.tau))

    def guide(self):
        num_ltla = self.cases.shape[0]
        num_lin = self.lineages.shape[-1] - 1
        num_basis = self.B.shape[-1]
        num_ltla_lin = self._nan_idx.shape[0]

        if self.fit_rho:
            rho_loc = npy.param(
                Sites.RHO + Sites.LOC,
                jnp.tile(self.rho_loc, (num_ltla, 1)),
            )
            rho_scale = npy.param(
                Sites.RHO + Sites.SCALE,
                jnp.tile(self.init_scale * self.rho_scale, (num_ltla, 1)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.RHO, dist.Normal(rho_loc, rho_scale))

        # mean / sd for parameter s
        beta_0_loc = npy.param(
            Sites.BETA_0 + Sites.LOC,
            jnp.tile(self.beta_loc, (num_ltla_lin, num_basis)),
        )
        beta_0_scale = npy.param(
            Sites.BETA_0 + Sites.SCALE,
            self.init_scale
            * self.beta_scale
            * jnp.stack(num_ltla_lin * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        npy.sample(
            Sites.BETA_0, dist.MultivariateNormal(beta_0_loc, scale_tril=beta_0_scale)
        )

        mu_bc_loc = npy.param(
            Sites.MU_BC + Sites.LOC,
            jnp.concatenate(
                [jnp.repeat(self.mu_b_loc, num_lin), jnp.repeat(self.mu_c_loc, num_lin)]
            ),
        )
        mu_bc_scale = npy.param(
            Sites.MU_BC + Sites.SCALE,
            jnp.diag(
                jnp.concatenate(
                    [
                        jnp.repeat(
                            self.init_scale * self.mu_b_scale * self.time_scale, num_lin
                        ),
                        jnp.repeat(self.init_scale * self.mu_c_scale, num_lin),
                    ]
                )
            ),
            constraint=dist.constraints.lower_cholesky,
        )
        npy.sample(
            Sites.MU_BC, dist.MultivariateNormal(mu_bc_loc, scale_tril=mu_bc_scale)
        )

        c_loc = npy.param(
            Sites.C + Sites.LOC, jnp.tile(self.c_loc, (num_ltla_lin, num_lin))
        )

        c_scale = npy.param(
            Sites.C + Sites.SCALE,
            jnp.tile(self.init_scale * self.c_scale, (num_ltla_lin, num_lin)),
        )
        npy.sample(Sites.C, dist.Normal(c_loc, c_scale))
