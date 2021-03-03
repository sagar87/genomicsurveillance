import numpy as np
from scipy.interpolate import BSpline


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
    :rtype: np.ndarray
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
