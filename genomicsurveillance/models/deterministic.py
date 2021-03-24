import numpy as np
from scipy.special import logsumexp

from .sites import Sites


def compute_mu(model, func="mean", BETA=Sites.BETA):
    return model.posterior.__getattribute__(func)(BETA) @ model.B[0].T


def compute_lambda(model, func="mean", BETA=Sites.BETA):
    return model.population.reshape(-1, 1) * np.exp(
        compute_mu(model, func=func, BETA=BETA)
    )


def compute_lambda_lineage(model, func="mean", BETA=Sites.BETA, B=Sites.B0, C=Sites.C):
    lamb = compute_lambda(model, func=func, BETA=BETA)
    prob = compute_probabilities(model, func=func, B=B, C=C)
    return lamb.reshape(*lamb.shape, 1) * prob


def compute_probabilities(model, func="mean", B=Sites.B0, C=Sites.C):
    b = model.posterior.__getattribute__(func)(B)
    c = model.posterior.__getattribute__(func)(C)

    logits = (
        b.reshape(1, 1, model.num_lin + 1)
        * np.arange(model.num_time).reshape(
            1,
            -1,
            1,
        )
        + c.reshape(model.num_ltla, 1, model.num_lin + 1)
    )

    p = np.exp(logits) / np.exp(logsumexp(logits, -1, keepdims=True))
    return p


def compute_R(model, func="mean", BETA=Sites.BETA, B=Sites.B0):
    beta = model.posterior.__getattribute__(func)(BETA)
    b = model.posterior.__getattribute__(func)(B)

    R = np.exp(
        ((beta @ model.B[1].T)[..., np.newaxis] + b.reshape(-1, 1, model.num_lin + 1))
        * model.tau
    )
    return R


def compute_region_lambda_lineage(
    model, region, func="mean", BETA=Sites.BETA, B=Sites.B0, C=Sites.C
):
    region_indicator = np.array(
        [region[model.nan_idx] == i for i in np.unique(region)]
    ).astype("float")
    lambda_lineage = compute_lambda_lineage(model, func=func, BETA=BETA, B=B, C=C)
    lambda_lineage_regions = np.einsum(
        "ij,jkl->ikl", region_indicator, lambda_lineage[model.nan_idx]
    )
    return lambda_lineage_regions


def compute_region_probabilities(
    model, region, func="mean", BETA=Sites.BETA, B=Sites.B0, C=Sites.C
):
    lambda_lineage_regions = compute_region_lambda_lineage(
        model, region, func=func, BETA=BETA, B=B, C=C
    )
    return lambda_lineage_regions / lambda_lineage_regions.sum(-1, keepdims=True)


def compute_transmissibility(model, func="mean", B=Sites.B0):
    b = model.posterior.__getattribute__(func)(B)
    return np.exp(b * model.tau)


def compute_growth_rate(model, func="mean", BETA=Sites.BETA):
    beta = model.posterior.__getattribute__(func)(BETA)
    return beta @ model.B[1].T
