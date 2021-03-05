import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


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
