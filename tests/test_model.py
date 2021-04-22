import numpy as np


def test_posterior_fixture(test_posterior):
    assert isinstance(test_posterior, dict)
    assert isinstance(test_posterior["basis"], np.ndarray)
    assert isinstance(test_posterior["dates"], np.ndarray)
    assert isinstance(test_posterior["lineages"], np.ndarray)
    assert isinstance(test_posterior["cases"], np.ndarray)


# def test_clock_reset_fixture(test_clock_reset_model):
#     pass


def test_get_lambda_interface(
    lineage_model, independent_clock_reset_model, clock_reset_model
):

    for model in [lineage_model, independent_clock_reset_model, clock_reset_model]:
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


def test_get_logits_interface(
    lineage_model, independent_clock_reset_model, clock_reset_model
):

    for model in [lineage_model, independent_clock_reset_model, clock_reset_model]:
        assert model.get_logits().ndim == 4
        assert model.get_logits().shape == (100, 4, 205, 59)

        assert model.get_logits(1).ndim == 4
        assert model.get_logits(1).shape == (100, 1, 205, 59)
        assert model.get_logits([1, 2]).shape == (100, 2, 205, 59)
        assert model.get_logits(np.array([1, 2])).shape == (100, 2, 205, 59)

        assert model.get_logits(None, np.arange(10)).ndim == 4
        assert model.get_logits(None, np.arange(10)).shape == (100, 4, 10, 59)
        assert model.get_logits(None, 1).ndim == 4
        assert model.get_logits(None, 1).shape == (100, 4, 1, 59)
        assert model.get_logits(None, [1, 2]).ndim == 4
        assert model.get_logits(None, [1, 2]).shape == (100, 4, 2, 59)

        assert model.get_logits(None, None, np.arange(10)).ndim == 4
        assert model.get_logits(None, None, np.arange(10)).shape == (100, 4, 205, 10)
        assert model.get_logits(None, None, 1).ndim == 4
        assert model.get_logits(None, None, 1).shape == (100, 4, 205, 1)
        assert model.get_logits(None, None, [1, 2]).ndim == 4
        assert model.get_logits(None, None, [1, 2]).shape == (100, 4, 205, 2)
        assert model.get_logits(None, None, np.array([1, 2])).shape == (100, 4, 205, 2)

        assert model.get_logits(np.arange(2), np.arange(10)).ndim == 4
        assert model.get_logits(np.arange(2), np.arange(10)).shape == (100, 2, 10, 59)
        assert model.get_logits(1, np.arange(10)).ndim == 4
        assert model.get_logits(1, np.arange(10)).shape == (100, 1, 10, 59)
        assert model.get_logits(np.arange(2), np.arange(10), np.arange(10)).ndim == 4
        assert model.get_logits(np.arange(2), np.arange(10), np.arange(10)).shape == (
            100,
            2,
            10,
            10,
        )


def test_get_probabilities_interface(
    lineage_model, independent_clock_reset_model, clock_reset_model
):

    for model in [lineage_model, independent_clock_reset_model, clock_reset_model]:
        assert model.get_probabilities().ndim == 4
        assert model.get_probabilities().shape == (100, 4, 205, 59)

        assert model.get_probabilities(1).ndim == 4
        assert model.get_probabilities(1).shape == (100, 1, 205, 59)
        assert model.get_probabilities([1, 2]).shape == (100, 2, 205, 59)
        assert model.get_probabilities(np.array([1, 2])).shape == (100, 2, 205, 59)

        assert model.get_probabilities(None, np.arange(10)).ndim == 4
        assert model.get_probabilities(None, np.arange(10)).shape == (100, 4, 10, 59)
        assert model.get_probabilities(None, 1).ndim == 4
        assert model.get_probabilities(None, 1).shape == (100, 4, 1, 59)
        assert model.get_probabilities(None, [1, 2]).ndim == 4
        assert model.get_probabilities(None, [1, 2]).shape == (100, 4, 2, 59)

        assert model.get_probabilities(None, None, np.arange(10)).ndim == 4
        assert model.get_probabilities(None, None, np.arange(10)).shape == (
            100,
            4,
            205,
            10,
        )
        assert model.get_probabilities(None, None, 1).ndim == 4
        assert model.get_probabilities(None, None, 1).shape == (100, 4, 205, 1)
        assert model.get_probabilities(None, None, [1, 2]).ndim == 4
        assert model.get_probabilities(None, None, [1, 2]).shape == (100, 4, 205, 2)
        assert model.get_probabilities(None, None, np.array([1, 2])).shape == (
            100,
            4,
            205,
            2,
        )

        assert model.get_probabilities(np.arange(2), np.arange(10)).ndim == 4
        assert model.get_probabilities(np.arange(2), np.arange(10)).shape == (
            100,
            2,
            10,
            59,
        )
        assert model.get_probabilities(1, np.arange(10)).ndim == 4
        assert model.get_probabilities(1, np.arange(10)).shape == (100, 1, 10, 59)
        assert (
            model.get_probabilities(np.arange(2), np.arange(10), np.arange(10)).ndim
            == 4
        )
        assert model.get_probabilities(
            np.arange(2), np.arange(10), np.arange(10)
        ).shape == (
            100,
            2,
            10,
            10,
        )


def test_get_lambda_lineage_interface(
    lineage_model, independent_clock_reset_model, clock_reset_model
):

    for model in [lineage_model, independent_clock_reset_model, clock_reset_model]:
        assert model.get_lambda_lineage().ndim == 4
        assert model.get_lambda_lineage().shape == (100, 4, 205, 59)

        assert model.get_lambda_lineage(1).ndim == 4
        assert model.get_lambda_lineage(1).shape == (100, 1, 205, 59)
        assert model.get_lambda_lineage([1, 2]).shape == (100, 2, 205, 59)
        assert model.get_lambda_lineage(np.array([1, 2])).shape == (100, 2, 205, 59)

        assert model.get_lambda_lineage(None, np.arange(10)).ndim == 4
        assert model.get_lambda_lineage(None, np.arange(10)).shape == (100, 4, 10, 59)
        assert model.get_lambda_lineage(None, 1).ndim == 4
        assert model.get_lambda_lineage(None, 1).shape == (100, 4, 1, 59)
        assert model.get_lambda_lineage(None, [1, 2]).ndim == 4
        assert model.get_lambda_lineage(None, [1, 2]).shape == (100, 4, 2, 59)

        assert model.get_lambda_lineage(None, None, np.arange(10)).ndim == 4
        assert model.get_lambda_lineage(None, None, np.arange(10)).shape == (
            100,
            4,
            205,
            10,
        )
        assert model.get_lambda_lineage(None, None, 1).ndim == 4
        assert model.get_lambda_lineage(None, None, 1).shape == (100, 4, 205, 1)
        assert model.get_lambda_lineage(None, None, [1, 2]).ndim == 4
        assert model.get_lambda_lineage(None, None, [1, 2]).shape == (100, 4, 205, 2)
        assert model.get_lambda_lineage(None, None, np.array([1, 2])).shape == (
            100,
            4,
            205,
            2,
        )

        assert model.get_lambda_lineage(np.arange(2), np.arange(10)).ndim == 4
        assert model.get_lambda_lineage(np.arange(2), np.arange(10)).shape == (
            100,
            2,
            10,
            59,
        )
        assert model.get_lambda_lineage(1, np.arange(10)).ndim == 4
        assert model.get_lambda_lineage(1, np.arange(10)).shape == (100, 1, 10, 59)
        assert (
            model.get_lambda_lineage(np.arange(2), np.arange(10), np.arange(10)).ndim
            == 4
        )
        assert model.get_lambda_lineage(
            np.arange(2), np.arange(10), np.arange(10)
        ).shape == (
            100,
            2,
            10,
            10,
        )


def test_get_transmissibility(
    lineage_model, independent_clock_reset_model, clock_reset_model
):

    for model in [lineage_model, independent_clock_reset_model, clock_reset_model]:
        assert model.get_transmissibility().ndim == 2
        assert model.get_transmissibility().shape == (100, 59)
        assert model.get_transmissibility(rebase=30).ndim == 2
        assert model.get_transmissibility(rebase=30).shape == (100, 59)
