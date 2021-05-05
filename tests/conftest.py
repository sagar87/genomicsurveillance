# src/tests/conftest.py
import os
import pickle

import numpy as np
import pandas as pd
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
def lineage_model(test_posterior):
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
def independent_clock_reset_model(test_posterior):
    m = MultiLineageClockReset(
        cases=test_posterior["cases"],
        lineages=test_posterior["lineages"],
        lineage_dates=test_posterior["dates"],
        population=test_posterior["population"],
        basis=test_posterior["basis"],
        posterior=Posterior(test_posterior),
        num_samples=100,
        independent_clock=True,
    )
    return m


@pytest.fixture
def clock_reset_model(test_posterior):
    test_posterior["t"] = test_posterior["t"][:, 1, :].reshape(100, 1, -1)
    print(test_posterior["t"].shape)

    m = MultiLineageClockReset(
        cases=test_posterior["cases"],
        lineages=test_posterior["lineages"],
        lineage_dates=test_posterior["dates"],
        population=test_posterior["population"],
        basis=test_posterior["basis"],
        posterior=Posterior(test_posterior),
        num_samples=100,
        independent_clock=False,
    )
    return m


@pytest.fixture
def ltla_dfs(rootdir):

    df1 = pd.read_csv(os.path.join(rootdir, "test_files/E06000024.csv"), index_col=0)
    df2 = pd.read_csv(os.path.join(rootdir, "test_files/E06000022.csv"), index_col=0)
    df3 = pd.read_csv(os.path.join(rootdir, "test_files/E06000021.csv"), index_col=0)
    return [df1, df2, df3]
