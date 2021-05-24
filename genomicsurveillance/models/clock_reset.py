import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax.ops import index
from jax.scipy.special import logsumexp

from genomicsurveillance.distributions import NegativeBinomial
from genomicsurveillance.handler import Model, make_array

from .base import Lineage
from .sites import Sites


class MultiLineageClockReset(Model, Lineage):
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
    :param kwargs: SVI Handler arguments.
    """

    _latent_variables = [Sites.BETA1, Sites.BC0, Sites.B1, Sites.C1, Sites.T]

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
        alpha0=0.01,
        alpha1=100.0,
        b0_loc: float = 0.0,
        b0_scale: float = 0.2,
        c0_loc: float = 0.0,
        c0_scale: float = 10.0,
        c_scale: float = 10.0,
        b_scale=False,
        fit_rho: bool = False,
        rho_loc: float = np.log(10.0),
        rho_scale: float = 1.0,
        auto_correlation: float = 0.5,
        offset: int = 21,
        independent_clock: bool = False,
        posterior=None,
        model_kwargs: dict = dict(
            rng_key=4587, handler="SVI", num_epochs=30000, lr=0.001, num_samples=1000
        ),
    ):
        """
        Constructor.
        """
        # TODO: More sanity checks
        assert (
            cases.shape[0] == lineages.shape[0]
        ), "cases and lineages must have the number of location"
        Model.__init__(self, **model_kwargs)
        Lineage.__init__(
            self,
            tau=tau,
            cases=cases,
            lineages=lineages,
            lineage_dates=lineage_dates,
            population=population,
            basis=basis,
            auto_correlation=auto_correlation,
            posterior=posterior,
        )

        self.init_scale = init_scale
        self.fit_rho = fit_rho

        self.beta_loc = beta_loc
        self.beta_scale = beta_scale
        self.alpha0 = alpha0
        self.alpha1 = self._check_alpha1(alpha1)

        self.b0_loc = b0_loc
        self.b0_scale = b0_scale
        self.c0_loc = c0_loc
        self.c0_scale = c0_scale

        self.c_loc = 0.0
        self.c_scale = c_scale
        self.b_loc = 0.0
        self.b_scale = b_scale

        self.rho_loc = rho_loc
        self.rho_scale = rho_scale

        self.offset = offset
        self.independent_clock = independent_clock
        self.time, self.intercept = self.clock()
        self.u, self.v0, self.w = np.ogrid[tuple(map(slice, self.time.shape))]

    def _check_alpha1(self, alpha1):
        if type(alpha1) == int or type(alpha1) == float:
            return alpha1
        elif type(alpha1) == np.ndarray:
            assert (
                alpha1.shape[0] == self.num_ltla
            ), "If alpha1 is an array, alpha1.shape[0] must match cases.shape[0]."
            return alpha1[self.nan_idx]

    def get_logits(self, ltla=None, time=None, lineage=None):

        # this is a a bit complicatd
        b = self.posterior.dist(Sites.B1, ltla, None, lineage)
        c = self.posterior.dist(Sites.C1, ltla, None, lineage)

        # print(b.shape, c.shape)
        # regenerate the offset array
        idx = self._indices(self.time.shape, ltla, None, lineage)

        # print(idx)
        t_expanded = np.stack([self.time[idx]] * self.num_samples, 0)
        t_expanded = t_expanded.reshape(-1, t_expanded.shape[-2], t_expanded.shape[-1])
        g_expanded = np.stack([self.intercept[idx]] * self.num_samples, 0)
        g_expanded = g_expanded.reshape(-1, g_expanded.shape[-2], g_expanded.shape[-1])

        # index array
        if ltla is not None:
            if self.independent_clock:
                copies = 1
            else:
                copies = len(make_array(ltla))
        else:
            if self.independent_clock:
                copies = 1
            else:
                copies = self.num_ltla

        if self.independent_clock:
            t_idx = np.repeat(self.posterior.dist(Sites.T, ltla, lineage), copies, 1)
        else:
            t_idx = np.repeat(self.posterior.dist(Sites.T, None, lineage), copies, 1)

        t_idx = t_idx.reshape(-1, t_idx.shape[-1])
        t_idx = (t_idx - self.offset) % t_expanded.shape[1]

        # create index array
        u, v, w = np.ogrid[
            : t_expanded.shape[0], : t_expanded.shape[1], : t_expanded.shape[2]
        ]
        v = v - t_idx[:, np.newaxis]
        t = t_expanded[u, v, w][:, self.offset : -self.offset].reshape(
            self.num_samples,
            -1,
            t_expanded.shape[-2] - 2 * self.offset,
            t_expanded.shape[-1],
        )
        g = g_expanded[u, v, w][:, self.offset : -self.offset].reshape(
            self.num_samples,
            -1,
            g_expanded.shape[-2] - 2 * self.offset,
            g_expanded.shape[-1],
        )

        idx = self.posterior.indices(t.shape, None, time)

        return b * t[idx] + (c + g[idx])

    def clock(self, to_jax=True):
        t0 = (self.lineages > 0).argmax(1)  # - self.clock_shift
        t0[t0 < 0] = 0
        lineage_present = self.lineages.sum(1) > 0

        t = []
        g = []

        for ltla in range(self.cases.shape[0]):
            larr = []
            garr = []

            for i, l in enumerate(t0[ltla]):
                if lineage_present[ltla, i]:
                    start_offset = self.lineage_dates[l] + self.offset
                    end_offset = -self.lineage_dates[l] + self.offset
                    larr.append(
                        np.concatenate(
                            [
                                np.repeat(0, start_offset),
                                np.arange(0, self.num_time + end_offset),
                            ]
                        )
                    )
                    garr.append(
                        np.concatenate(
                            [
                                np.repeat(self.EPS, start_offset),
                                np.zeros(self.num_time + end_offset),
                            ]
                        )
                    )
                else:
                    larr.append(np.zeros(self.num_time + 2 * self.offset))
                    garr.append(np.repeat(self.EPS, self.num_time + 2 * self.offset))

            t.append(np.stack(larr).T)
            g.append(np.stack(garr).T)

        t = np.stack(t)
        g = np.stack(g)

        if to_jax:
            return jnp.asarray(t), jnp.asarray(g)

        return t, g

    def shift_clock(self, t_offset):
        t_index = (t_offset - self.offset) % self.time.shape[1]
        v = self.v0 - t_index[:, jnp.newaxis]
        t = self.time[self.u, v, self.w][:, self.offset : -self.offset, :]
        g = self.intercept[self.u, v, self.w][:, self.offset : -self.offset, :]
        return t, g

    def get_lambda(self, ltla=None, time=None):
        beta = self._expand_dims(self.posterior.dist(Sites.BETA1, ltla), self.LIN_DIM)
        basis = self._expand_dims(
            self.B[self._indices(self.B.shape, 0, time)].T.squeeze(), self.TIME_DIM
        )

        lamb = self.population[ltla].reshape(1, -1, 1) * np.exp(
            np.einsum("ijk,kl->ijl", beta, basis) + self.beta_loc
        )
        lamb = self._expand_dims(lamb, dim=self.LIN_DIM)
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
                    self.SCALE * jnp.repeat(self.b0_scale, self.num_lin),
                    jnp.repeat(self.c0_scale, self.num_lin),
                ]
            )
        )
        bc0 = npy.sample(Sites.BC0, dist.MultivariateNormal(bc0_loc, bc0_scale))

        # split the array
        if self.b_scale:
            b_offset = npy.sample(
                Sites.B,
                dist.Normal(
                    jnp.tile(self.b_loc, (self.num_ltla_lin, self.num_lin)),
                    jnp.tile(
                        self.SCALE * self.b_scale,
                        (self.num_ltla_lin, self.num_lin),
                    ),
                ),
            )
            b = npy.deterministic("b", (b_offset + bc0[: self.num_lin]) / self.SCALE)
        else:
            b = bc0[: self.num_lin] / self.SCALE

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

        if self.independent_clock:
            t_offset = npy.sample(
                Sites.T,
                dist.Categorical(jnp.ones(self.offset) / self.offset).expand(
                    [self.num_ltla, self.num_lin + 1]
                ),
            )
        else:
            t_offset = npy.sample(
                Sites.T,
                dist.Categorical(jnp.ones(self.offset) / self.offset).expand(
                    [1, self.num_lin + 1]
                ),
            )

        t, g = self.shift_clock(t_offset)
        logits = b1 * t[:, self.lineage_dates] + (c1 + g[:, self.lineage_dates])

        p = jnp.exp(logits - logsumexp(logits, -1, keepdims=True))

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
        if self.alpha0 is not None:
            # conc = (
            #     p[self.nan_idx]
            #     * (self.lineages[self.nan_idx].sum(-1, keepdims=True) + 1.0)
            #     / self.alpha
            # )
            conc = self.alpha0 + self.alpha1 * p[self.nan_idx]
            npy.sample(
                Sites.LINEAGE,
                dist.DirichletMultinomial(
                    conc,
                    total_count=self.lineages[self.nan_idx].sum(-1),
                ),
                obs=self.lineages[self.nan_idx],
            )
        else:
            npy.sample(
                Sites.LINEAGE,
                dist.MultinomialProbs(
                    p[self.nan_idx],
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
                        * self.SCALE
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

        if self.b_scale:
            b_loc = npy.param(
                Sites.B + Sites.LOC,
                self.b_loc * jnp.ones((self.num_ltla_lin, self.num_lin)),
            )
            b_scale = npy.param(
                Sites.B + Sites.SCALE,
                self.init_scale
                * self.b_scale
                * self.SCALE
                * jnp.ones((self.num_ltla_lin, self.num_lin)),
                constraint=dist.constraints.positive,
            )
            npy.sample(Sites.B, dist.Normal(b_loc, b_scale))

    def deterministic(self):
        """
        Performs post processing steps
        """
        if Sites.BC0 in self.posterior.keys():
            self.posterior[Sites.B0] = (
                self.posterior[Sites.BC0][:, : self.num_lin] / self.SCALE
            )
