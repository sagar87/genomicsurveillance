from typing import Optional

import numpy as np
import scipy.stats as stats


def reparametrise_gamma(mean, cv):
    alpha = 1 / cv ** 2
    beta = mean * cv ** 2
    return alpha, beta


def epiestim_R(
    cases: np.ndarray,
    mu: float,
    sigma: Optional[float],
    cv: Optional[float],
    a: float = 1,
    b: float = 5,
    tau: int = 7,
    q_lower: float = 0.025,
    q_upper: float = 0.975,
):
    """
    Calculates the R value as described in Anne Core et. al. 2013.

    :param cases: Array with daily number of cases.
    :param mu: Mean of the serial interval distribution.
    :param sigma: Standard deviation of the serial interval distribution,
        defaults to None.
    :param cv: Coefficient of variation of the serial interval distribution,
        defaults to None. Either sigma OR cv must be specified.
    :param a: Prior shape of the gamma distribution of R.
    :param b: Prior scale of the gamma distribution of R.
    :param tau: Time interval in days over which the convolution is performed.
    :param q_lower: Lower bound of the returned R value.
    :param q_upper: Upper bound of the returned R value.
    :returns: An array with shape (3, cases.shape[0]) containing
        the mean, lower and upper bounds of the calculated R value.
    """
    # TODO: Refactor this logic ?
    assert ((sigma is not None) and (cv is None)) or (
        (sigma is None) and (cv is not None)
    ), "Either sigma OR cv must be defined."

    if cv is not None:
        sigma = cv * mu

    T = cases.shape[0]
    p = np.array([epiestim_discretise_serial_interval(i, mu, sigma) for i in range(T)])
    # E = (np.arange(T) * p).sum()
    # process I
    incidence = np.zeros((cases.shape[0], 2))
    incidence[1:, 0] = cases[1:]
    incidence[0, 1] = cases[0]
    # compute posterior
    a_posterior = a + np.convolve(incidence[:, 0], np.ones(tau + 1), "valid")
    b_posterior = 1 / (
        1 / b
        + np.convolve(
            np.convolve(incidence.sum(1), p, "full")[:T], np.ones(tau + 1), "valid"
        )
    )
    d = stats.gamma(a=a_posterior, scale=b_posterior)
    lower = d.ppf(q_lower)
    upper = d.ppf(q_upper)
    return (
        np.append([np.nan] * tau, a_posterior * b_posterior),
        np.append([np.nan] * tau, lower),
        np.append([np.nan] * tau, upper),
    )


def epiestim_discretise_serial_interval(
    k: int, mu: float = 6.3, sigma: Optional[float] = None, cv: Optional[float] = 0.62
):
    """
    Discretises a gamma distribution according to Cori et al. 2013.


    :param k: Day of the serial interval (k >= 0).
    :param mu: Mean of the serial interval distribution.
    :param sigma: Standard deviation of the serial interval distribution,
        defaults to None.
    :param cv: Coefficient of variation of the serial interval distribution,
        defaults to None. Either sigma OR cv must be specified.
    :returns: Discretised serial interval distribution.
    """
    # TODO: Refactor this logic ?
    assert ((sigma is not None) and (cv is None)) or (
        (sigma is None) and (cv is not None)
    ), "Either sigma OR cv must be defined."

    if cv is not None:
        sigma = cv * mu

    a = ((mu - 1) / sigma) ** 2
    b = sigma ** 2 / (mu - 1)

    cdf_gamma = stats.gamma(a=a, scale=b).cdf
    cdf_gamma2 = stats.gamma(a=a + 1, scale=b).cdf

    res = k * cdf_gamma(k) + (k - 2) * cdf_gamma(k - 2) - 2 * (k - 1) * cdf_gamma(k - 1)
    res = res + a * b * (2 * cdf_gamma2(k - 1) - cdf_gamma2(k - 2) - cdf_gamma2(k))

    return max(res, 0)


def infection_to_test(k: int, mu=1.92, sigma=0.65):
    """
    Infection to postive test result. Derived from the incubation time distribution in
    Bi et. al. (2020).
    """
    test_dist = stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))
    return np.diff([test_dist.cdf(i) for i in range(k + 2)])[k]
