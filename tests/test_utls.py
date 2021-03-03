import numpy as np
from genomicsurveillance.utils import create_spline_basis


def test_create_spline_basis():
    x = np.arange(100)
    _, B = create_spline_basis(x, num_knots=5, add_intercept=False)

    assert B.ndim == 3
    assert B.shape[0] == 2
    assert B.shape[1] == 100
    assert B.shape[2] == 5 + 2
