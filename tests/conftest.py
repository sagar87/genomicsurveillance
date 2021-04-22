# src/tests/conftest.py
import os
import pickle

import numpy as np
import pytest

from genomicsurveillance.handler import Posterior
from genomicsurveillance.models.models import Lineage, MultiLineageClockReset


@pytest.fixture(scope="module")
def raw_posterior():
    raw_posterior = {
        "a": np.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5),
        "b": np.arange(5 * 5 * 5 * 5).reshape(5, 5, 5, 5),
    }
    return raw_posterior


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_posterior(rootdir):
    test_file = os.path.join(rootdir, "test_files/test.pkl")
    data = pickle.load(open(test_file, "rb"))
    return data


@pytest.fixture
def test_lineage_model(test_posterior):
    m = Lineage(
        5.0,
        test_posterior["cases"],
        test_posterior["lineages"],
        test_posterior["dates"],
        test_posterior["population"],
        test_posterior["basis"],
        Posterior(test_posterior),
    )
    return m


@pytest.fixture
def test_clock_reset_model(test_posterior):
    m = MultiLineageClockReset(
        cases=test_posterior["cases"],
        lineages=test_posterior["lineages"],
        lineage_dates=test_posterior["dates"],
        population=test_posterior["population"],
        basis=test_posterior["basis"],
        posterior=Posterior(test_posterior),
    )
    return m
