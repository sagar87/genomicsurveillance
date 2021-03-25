import numpy as np
from scipy.special import logsumexp

from .sites import Sites


def compute_logits(model, idx):
    return model.posterior.dist(Sites.B1, idx) * np.arange(0, model.num_time).reshape(
        1, -1, 1
    ) + model.posterior.dist(Sites.C1, idx)


def compute_probabilities(model, idx):
    logits = compute_logits(model, idx)
    p = np.exp(logits) / np.exp(logsumexp(logits, -1, keepdims=True))
    if p.ndim <= 3:
        p = np.expand_dims(p, 1)
    return p


def compute_lambda(model, idx):
    beta = model.posterior.dist(Sites.BETA1, idx)
    if beta.ndim < 3:
        beta = np.expand_dims(beta, 1)

    return model.population[idx].reshape(1, -1, 1) * np.exp(
        np.einsum("ijk,kl->ijl", beta, model.B[0].T)
    )


def compute_lambda_lineage(model, idx):
    prob = compute_probabilities(model, idx)
    lamb = compute_lambda(model, idx)
    return lamb.reshape(*lamb.shape, 1) * prob


def compute_R(model, idx):
    return np.exp(
        (
            (model.posterior.dist(Sites.BETA1, idx) @ model.B[1].T)[..., np.newaxis]
            + model.posterior.dist(Sites.B1, idx)
        )
        * model.tau
    )


def compute_transmissibility(model):
    return np.exp(model.posterior.dist(Sites.B1) * model.tau)


def compute_growth_rate(model, idx):
    return model.posterior.dist(Sites.BETA1, idx) @ model.B[1].T
