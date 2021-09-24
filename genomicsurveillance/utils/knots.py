from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .epiestim import infection_to_test


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


class Knots(object):
    def __init__(
        self,
        days: int,
        shift: int = 1,
        periods: int = 7,
        padding: int = 21,
        dist: Callable = infection_to_test,
        dist_kwargs: dict = {},
        degree: int = 3,
        offset: int = 0,
    ):
        self.days = days
        self.shift = shift
        self.padding = padding
        self.periods = periods
        self.offset = offset
        self.degree = degree

        self.t = np.arange(-self.padding, self.days + self.padding)

        self.day = shift

        self.knot_list = np.arange(
            self.day - self.padding - 1,
            self.t.shape[0] - self.padding,
            self.periods,
        )

        self.p = np.array([dist(i, **dist_kwargs) for i in range(self.padding)])
        self.p /= self.p.sum()

        self.get_spline_basis()
        self.convolve_spline_basis()

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
        self.basis = np.stack([self.B_conv, self.B_diff])


class KnotList(Knots):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        starting_day: str = "Wednesday",
        periods: int = 7,
        padding: int = 21,
        dist: Callable = infection_to_test,
        dist_kwargs: dict = {},
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
        self.p = np.array([dist(i, **dist_kwargs) for i in range(self.padding)])
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


class TruncatedKnots(object):
    def __init__(
        self,
        days: int,
        shift: int = 1,
        periods: int = 7,
        padding: int = 21,
        dist: Callable = infection_to_test,
        dist_kwargs: dict = {},
        degree: int = 3,
        offset: int = 0,
        truncate: int = 21,
    ):
        self.days = days
        self.shift = shift
        self.padding = padding
        self.periods = periods
        self.offset = offset
        self.degree = degree

        self.t = np.arange(-self.padding, self.days + self.padding)

        self.day = shift

        self.knot_list = np.arange(
            self.day - self.padding - 1,
            self.t.shape[0] - self.padding,
            self.periods,
        )
        self.truncate = [i for i in range(self.days - truncate, self.days)]
        self.knot_list = np.array(
            [
                i
                for i in range(
                    self.day - self.padding - 1,
                    self.t.shape[0] - self.padding,
                    self.periods,
                )
                if i not in self.truncate
            ]
        )

        self.p = np.array([dist(i, **dist_kwargs) for i in range(self.padding)])
        self.p /= self.p.sum()

        self.get_spline_basis()
        self.convolve_spline_basis()

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
        self.basis = np.stack([self.B_conv, self.B_diff])


class NowCastKnots(object):
    def __init__(self, days, short_interval=14, long_interval=31, cutoff=6):
        self.cutoff = cutoff
        self.short = Knots(days, periods=short_interval)
        self.long = TruncatedKnots(days, periods=long_interval, truncate=long_interval)

        self.basis = np.concatenate(
            [self.long.basis, self.short.basis[..., :-cutoff]], -1
        )

        self.num_long_basis = self.long.basis.shape[-1]
        self.num_short_basis = self.short.basis.shape[-1] - cutoff
