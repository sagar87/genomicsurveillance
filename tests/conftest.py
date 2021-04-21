# src/tests/conftest.py

import numpy as np
import pytest


@pytest.fixture(scope="module")
def raw_posterior():
    raw_posterior = {
        "a": np.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5),
        "b": np.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5),
    }
    return raw_posterior
