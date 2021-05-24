import numpy as np
import pytest

from genomicsurveillance.utils import create_spline_basis, sort_lineages
from genomicsurveillance.utils.epiestim import epiestim_discretise_serial_interval
from genomicsurveillance.utils.lineages import alias_lineages


def test_create_spline_basis():
    x = np.arange(100)
    _, B = create_spline_basis(x, num_knots=5, add_intercept=False)

    assert B.ndim == 3
    assert B.shape[0] == 2
    assert B.shape[1] == 100
    assert B.shape[2] == 5 + 2


def test_alias_lineages():
    test_lineages = ["B.1", "B.1.2", "B.1.34", "B.1.1.234.1"]
    aliases = {"B.1.1.234.1": "B.1.1.234"}
    aliased_lineages = alias_lineages(test_lineages, aliases)

    assert len(aliased_lineages) == 4
    assert "B.1.1.234.1" not in aliased_lineages
    assert "B.1.1.234" in aliased_lineages

    test_lineages = ["B.1", "B.1.2", "B.1.34", "B.1.1.234.1"]
    aliases = {"B.1.1.234.1": "B.1.1"}
    aliased_lineages = alias_lineages(test_lineages, aliases)

    assert len(aliased_lineages) == 4
    assert "B.1.1.234.1" not in aliased_lineages
    assert "B.1.1" in aliased_lineages

    test_lineages = ["B.1", "B.1.2", "B.1.34", "B.1.1.234.1"]
    aliases = {"B.1.1.234.134": "B.1.1"}
    aliased_lineages = alias_lineages(test_lineages, aliases)

    assert len(aliased_lineages) == 4
    assert "B.1.1.234.1" in aliased_lineages
    assert "B.1.1" not in aliased_lineages


def test_alias_lineages_assertion():
    test_lineages = ["B.1", "B.1.2", "B.1.34", "B.1.1.234.1"]
    aliases = {"B.1.1.234.1": "B.1"}

    with pytest.raises(AssertionError):
        alias_lineages(test_lineages, aliases)

    test_lineages = ["B.1", "B.1.2", "B.1.34", "B.1.1.234.1"]
    aliases = {"B.1.1.234.1": "B.1.1.234", "B.1.34": "B.1.1.234"}

    with pytest.raises(AssertionError):
        alias_lineages(test_lineages, aliases)


def test_sort_lineages(lineage_list):
    ordered_lineages, other_lineages = sort_lineages(lineage_list)

    assert len(ordered_lineages) == len(lineage_list) - 1
    assert len(other_lineages) == 1
    assert ordered_lineages == [
        "A.18",
        "A.25",
        "B",
        "B.1.1",
        "B.1.1.25",
        "B.1.1.39",
        "B.1.1.50",
        "B.1.1.51",
        "B.1.1.54",
        "B.1.1.141",
        "B.1.1.189",
        "B.1.1.227",
        "B.1.1.235",
        "B.1.1.255",
        "B.1.1.277",
        "B.1.1.286",
        "B.1.1.288",
        "B.1.1.305",
        "B.1.2",
        "B.1.36",
        "B.1.36.16",
        "B.1.36.17",
        "B.1.149",
        "B.1.177",
        "B.1.235",
        "B.1.258",
        "B.1.258.3",
        "B.1.416.1",
        "B.1.505",
    ]

    ordered_lineages, other_lineages = sort_lineages(lineage_list + ["LineageX"])

    assert len(ordered_lineages) == len(lineage_list) - 1
    assert len(other_lineages) == 2
    assert ordered_lineages == [
        "A.18",
        "A.25",
        "B",
        "B.1.1",
        "B.1.1.25",
        "B.1.1.39",
        "B.1.1.50",
        "B.1.1.51",
        "B.1.1.54",
        "B.1.1.141",
        "B.1.1.189",
        "B.1.1.227",
        "B.1.1.235",
        "B.1.1.255",
        "B.1.1.277",
        "B.1.1.286",
        "B.1.1.288",
        "B.1.1.305",
        "B.1.2",
        "B.1.36",
        "B.1.36.16",
        "B.1.36.17",
        "B.1.149",
        "B.1.177",
        "B.1.235",
        "B.1.258",
        "B.1.258.3",
        "B.1.416.1",
        "B.1.505",
    ]


def test_sort_lineages_assertions():
    test_lineages = ["B.1", "B.1.2", "B.1", "B.1.34", "B.1.1.234.1"]

    with pytest.raises(AssertionError):
        sort_lineages(test_lineages)


def test_discretise():
    d2 = [epiestim_discretise_serial_interval(i, mu=6.2, cv=0.62) for i in range(25)]
    assert len(d2) == 25
