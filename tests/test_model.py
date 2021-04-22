import numpy as np


def test_posterior_fixture(test_posterior):
    assert isinstance(test_posterior, dict)
    assert isinstance(test_posterior["basis"], np.ndarray)
    assert isinstance(test_posterior["dates"], np.ndarray)
    assert isinstance(test_posterior["lineages"], np.ndarray)
    assert isinstance(test_posterior["cases"], np.ndarray)


# def test_clock_reset_fixture(test_clock_reset_model):
#     pass


def test_get_lambda_interface(test_lineage_model):

    for model in [test_lineage_model]:
        assert model.get_lambda().ndim == 4
        assert model.get_lambda().shape == (100, 4, 205, 1)

        # latla
        assert model.get_lambda(1).ndim == 4
        assert model.get_lambda(1).shape == (100, 1, 205, 1)
        assert model.get_lambda([1, 2]).shape == (100, 2, 205, 1)
        assert model.get_lambda(np.array([1, 2])).shape == (100, 2, 205, 1)

        # time
        assert model.get_lambda(None, np.arange(10)).ndim == 4
        assert model.get_lambda(None, np.arange(10)).shape == (100, 4, 10, 1)
        assert model.get_lambda(None, 1).ndim == 4
        assert model.get_lambda(None, 1).shape == (100, 4, 1, 1)
        assert model.get_lambda(None, [1, 2]).ndim == 4
        assert model.get_lambda(None, [1, 2]).shape == (100, 4, 2, 1)

        # both at the same time
        assert model.get_lambda(np.arange(2), np.arange(10)).ndim == 4
        assert model.get_lambda(np.arange(2), np.arange(10)).shape == (100, 2, 10, 1)
        assert model.get_lambda(1, np.arange(10)).ndim == 4
        assert model.get_lambda(1, np.arange(10)).shape == (100, 1, 10, 1)


def test_get_logits_interface(test_lineage_model):
    assert test_lineage_model.get_logits().ndim == 4
    assert test_lineage_model.get_logits().shape == (100, 4, 205, 59)

    # latla
    assert test_lineage_model.get_logits(1).ndim == 4
    assert test_lineage_model.get_logits(1).shape == (100, 1, 205, 59)
    assert test_lineage_model.get_logits([1, 2]).shape == (100, 2, 205, 59)
    assert test_lineage_model.get_logits(np.array([1, 2])).shape == (100, 2, 205, 59)


def test_get_lambda_lineage_interface(test_lineage_model):
    pass
    # assert test_lineage_model.get_lambda_lineage().ndim == 4
