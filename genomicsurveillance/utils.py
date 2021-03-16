import re
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import BSpline


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
    k: int, mu: float, sigma: Optional[float] = None, cv: Optional[float] = None
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


def create_date_list(
    periods: int, start_date: str = "2020-09-01", freq: str = "d"
) -> list:
    """
    Creates a consecutive date list starting at start_day for period days.

    :param periods: number of days
    :param start_date: a date in format in the format YYYY-MM-DD
    :param freq: frequency, defaults to "d" (days)
    :return: a list of dates
    :rtype: list
    """
    return [str(d)[:10] for d in pd.date_range(start_date, periods=periods, freq=freq)]


def create_spline_basis(
    x, knot_list=None, num_knots=None, degree: int = 3, add_intercept=True
):
    """
    Creates a spline basis functions.

    :param x: array of size num time steps
    :param knot_list: indices of knots
    :param num_knots: number of knots
    :param degree: degree of the spline function
    :param add_intercept: appends an additional column of ones, defaults
        to False
    :return: list of knots, basis functions
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    assert ((knot_list is None) and (num_knots is not None)) or (
        (knot_list is not None) and (num_knots is None)
    ), "Define knot_list OR num_knot"
    if knot_list is None:
        knot_list = np.quantile(x, q=np.linspace(0, 1, num=num_knots))
    else:
        num_knots = len(knot_list)

    knots = np.pad(knot_list, (degree, degree), mode="edge")
    B0 = BSpline(knots, np.identity(num_knots + 2), k=degree)
    # B0 = BSpline(knot_list, np.identity(num_knots), k=degree)
    B = B0(x)
    Bdiff = B0.derivative()(x)

    if add_intercept:
        B = np.hstack([np.ones(B.shape[0]).reshape(-1, 1), B])
        Bdiff = np.hstack([np.zeros(B.shape[0]).reshape(-1, 1), Bdiff])

    return knot_list, np.stack([B, Bdiff])


def is_nan_row(array: np.ndarray) -> np.ndarray:
    """
    Helper function to extract the indices of rows (1st dimension) in an array that
    contains nan values

    :param array: a numpy array
    :returns: an array of indices
    """
    return np.where(np.isnan(array.sum(axis=tuple(range(1, array.ndim)))))[0]


def sort_lineages(lineage_list, pattern=re.compile(r"[A-Z](\.\d+)*$")):
    # extract lineages that follow the specific pattern
    identifier = [lineage for lineage in lineage_list if pattern.match(lineage)]
    max_levels = max([len(lineage.split(".")) for lineage in identifier])

    # all other identifier
    other_identifier = [
        lineage for lineage in lineage_list if not pattern.match(lineage)
    ]

    identifier_levels = []
    for lineage in identifier:
        levels = lineage.split(".")
        while len(levels) < max_levels:
            levels = levels + ["0"]

        identifier_levels.append(levels)

    for i in reversed(range(max_levels)):
        identifier_levels.sort(key=lambda x: int(x[i]) if x[i].isdigit() else x[i])

    sorted_identifier = []
    for lineage in identifier_levels:
        sorted_identifier.append(".".join([i for i in lineage if i != "0"]))

    return sorted_identifier, other_identifier


def alias_lineages(lineage_list, alias, anti_alias=False):
    if anti_alias:
        alias = {v: k for k, v in alias.items()}
    return [alias[lineage] if lineage in alias else lineage for lineage in lineage_list]


def merge_lineages(
    lineage_identifier,
    lineage_counts,
    cutoff=100,
    skip=[],
    pattern=re.compile(r"[A-Z](\.\d+)*$"),
):
    def depth(x):
        return len(x.split("."))

    def parent(x):
        return ".".join(x.split(".")[:-1])

    cluster_dict = {
        k: [v, [k]]
        for k, v in sorted(
            zip(lineage_identifier, lineage_counts),
            key=lambda x: depth(x[0]),
            reverse=True,
        )
    }

    iteration = 0
    while True:
        remove_lineages = []
        for lineage, (lineage_count, cluster) in cluster_dict.items():
            if not pattern.match(lineage):
                continue

            parent_lineage = parent(lineage)
            lineage_cutoff = lineage_count < cutoff
            parent_exists = len(parent_lineage) > 0 and parent_lineage in cluster_dict

            if lineage_cutoff and parent_exists and lineage not in skip:
                cluster_dict[parent_lineage][0] += cluster_dict[lineage][0]
                cluster_dict[parent_lineage][1] += cluster_dict[lineage][1]
                remove_lineages.append(lineage)

        if len(remove_lineages) == 0:
            break
        else:
            for lineage in remove_lineages:
                # print(f'Pruning lineage {lineage}')
                del cluster_dict[lineage]
        iteration += 1

    merging_indices = {}
    merged_lineages = []

    sorted_identifier, other_identifier = sort_lineages(cluster_dict.keys())

    for lineage in sorted_identifier + other_identifier:
        (lineage_count, cluster) = cluster_dict[lineage]
        # drop lineages with zero counts
        if lineage_count == 0:
            continue
        merged_lineages.append(lineage)
        merging_indices[lineage_identifier.index(lineage)] = [
            lineage_identifier.index(i) for i in cluster
        ]

    return merged_lineages, merging_indices


def aggregate_tensor(lineage_tensor, cluster):
    lineage_red = np.zeros(
        (lineage_tensor.shape[0], lineage_tensor.shape[1], len(cluster))
    )
    for i, (k, v) in enumerate(cluster.items()):
        lineage_red[..., i] = lineage_tensor[..., v].sum(-1)

    return lineage_red


def preprocess_lineage_tensor(
    lineage_list: list,
    lineage_tensor: np.ndarray,
    aliases: Optional[dict] = None,
    vocs: list = [],
):
    """
    Preprocesses the lineage tensor.

    :param lineage_list: A list of all lineages.
    :param lineage_tensor: The lineage tensor (shape: (num_location, num_time, num_lineages).
    :param aliases: A dictionary with 1:1 mappings of lineages that shall be renamed.
    :param vocs: Variants of concerns, not to be merged.
    :return: The list of merged lineages and the reduced lineage tensor.
    """
    if aliases:
        alias_list = alias_lineages(lineage_list, aliases)
    else:
        alias_list = lineage_list

    lineage_counts = np.nansum(lineage_tensor, axis=(0, 1))

    # lineages of current interest
    refractory = pd.DataFrame(
        np.nansum(lineage_tensor, axis=(0)), columns=alias_list
    ).iloc[-1]
    refractory = refractory.index[refractory > 0].tolist()

    merged_lineages, cluster = merge_lineages(
        alias_list, lineage_counts, skip=refractory + vocs
    )
    lineage_tensor_red = aggregate_tensor(lineage_tensor, cluster)
    return merged_lineages, lineage_tensor_red


def discretise_serial_interval(k, mu: float = 6.3, sigma: float = 4.2):
    return stats.gamma.pdf(k, mu ** 2 / sigma ** 2, sigma ** 2 / mu)


class KnotList(object):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        starting_day: str = "Wednesday",
        periods: int = 7,
        padding: int = 20,
        mu: float = 6.3,
        sigma: float = 4.2,
        degree: int = 3,
        offset: int = 0,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.starting_day = starting_day
        self.periods = periods
        self.offset = offset

        # self.padding_multiplier = padding_multiplier
        self.padding = padding
        self.degree = degree

        # set the start and end of the data
        self.t_start = pd.to_datetime(self.start_date)
        self.t_end = pd.to_datetime(self.end_date)
        # pad the start and end dates
        self.t_start_pad = self.t_start - pd.to_timedelta(self.padding, unit="d")
        self.t_end_pad = self.t_end + pd.to_timedelta(self.padding, unit="d")

        # date ranges
        self.date_range = pd.date_range(
            self.t_start.strftime("%Y-%m-%d"),
            end=self.t_end.strftime("%Y-%m-%d"),
            freq="d",
        )
        self.date_range_pad = pd.date_range(
            self.t_start_pad.strftime("%Y-%m-%d"),
            end=self.t_end_pad.strftime("%Y-%m-%d"),
            freq="d",
        )

        self.dates = self.get_date_df()
        self.day = self.dates[
            (self.dates.idx > 0) & (self.dates.day == self.starting_day)
        ].iloc[0, self.dates.columns.get_loc("idx")]

        self.knot_list = np.arange(
            self.day - self.padding - 1,
            self.dates.shape[0] - self.padding,
            self.periods,
        )
        self.t = np.arange(-self.padding, self.date_range.shape[0] + self.padding)

        # serial interval distribution
        self.p = np.array(
            [discretise_serial_interval(i, mu, sigma) for i in range(self.padding)]
        )
        self.p /= self.p.sum()

        self.get_spline_basis()
        self.convolve_spline_basis()

    def get_date_df(self):
        df = pd.DataFrame(
            {
                "date": [date.strftime("%Y-%m-%d") for date in self.date_range_pad],
                "month": [date.month_name() for date in self.date_range_pad],
                "day": [date.day_name() for date in self.date_range_pad],
            },
        ).assign(idx=lambda df: np.arange(-self.padding, df.shape[0] - self.padding))
        return df

    def get_spline_basis(self):
        self.knots = np.pad(self.knot_list, (self.degree, self.degree), mode="edge")
        B0 = BSpline(self.knots, np.identity(len(self.knots)), k=self.degree)
        self.B_pad = B0(self.t)
        self.B = self.B_pad[self.padding : -self.padding]

        if self.degree > 0:
            self.B_diff = B0.derivative()(self.t)[self.padding : -self.padding]
        else:
            self.B_diff = np.zeros(self.B.shape)[self.padding : -self.padding]

    def convolve_spline_basis(self):
        B_conv = np.vstack(
            [
                np.convolve(self.B_pad[:, i], self.p, "full")[
                    self.padding + self.offset :
                ][: self.B.shape[0]]
                for i in range(self.B.shape[1])
            ]
        ).T

        b = np.where(B_conv.sum(0) == 0)[0]
        self.B_conv = np.delete(B_conv, b, axis=1)
        self.B_diff = np.delete(self.B_diff, b, axis=1)
