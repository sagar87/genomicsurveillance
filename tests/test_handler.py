import numpy as np
import pytest

from genomicsurveillance.handler import Posterior


def test_posterior_getter(raw_posterior):
    p = Posterior(raw_posterior)
    a = p["a"]

    assert isinstance(a, np.ndarray)
    with pytest.raises(KeyError):
        p["c"]


def test_posterior_dist(raw_posterior):
    """Posterior.dist returns the slices irrespective of whether np.arrays, lists or integers are provided."""
    p = Posterior(raw_posterior)

    assert p.dist("a").shape == (5, 5, 5, 5)
    assert p.dist("a", None).shape == (5, 5, 5, 5)
    assert np.all(p.dist("a", 1, 1, 1).squeeze() == np.array([31, 156, 281, 406, 531]))
    assert p.dist("a", 1).shape == (5, 1, 5, 5)
    assert p.dist("b", 1, None, 2).shape == (5, 1, 5, 1)
    assert p.dist("b", 1, 1, None).shape == (5, 1, 1, 5)
    assert p.dist("b", [1, 2], None, 2).shape == (5, 2, 5, 1)
    assert p.dist("b", [1, 2], [1, 2], [1, 2, 3]).shape == (5, 2, 2, 3)

    assert p.dist(
        "b", np.array([1, 2]), np.array([1, 2]), np.array([1, 2, 3])
    ).shape == (5, 2, 2, 3)
    assert p.dist("a", 2, np.array([1, 2]), np.array([1, 2, 3])).shape == (5, 1, 2, 3)
    assert p.dist("b", np.array([1, 2]), 2, np.array([1, 2, 3])).shape == (5, 2, 1, 3)
    assert p.dist("b", np.array([1, 2]), 2, [1, 2, 3]).shape == (5, 2, 1, 3)


def test_posterior_mean(raw_posterior):
    """Posterior.mean returns the slices irrespective of whether np.arrays, lists or integers are provided."""
    p = Posterior(raw_posterior)

    assert p.mean("a").shape == (5, 5, 5)
    assert p.mean("a", 1).shape == (1, 5, 5)
    assert p.mean("a", None, 1).shape == (5, 1, 5)
    assert p.mean("a", [1, 2]).shape == (2, 5, 5)
