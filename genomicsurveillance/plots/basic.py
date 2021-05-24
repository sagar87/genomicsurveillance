import matplotlib.pyplot as plt
import numpy as np


def dot_plot(
    dist,
    variable="sa",
    baseline="B.1",
    title=None,
    ylabel="Relative Growth Rate",
    xlabel="Lineage",
    xticklabels=None,
    ax=None,
):
    y = np.median(dist, 0)
    x = np.arange(y.shape[0])
    ci = np.abs((np.quantile(dist, [0.025, 0.975], 0) - y))

    if ax is None:
        ax = plt.gca()

    ax.errorbar(x, y, yerr=ci, ls="", marker=".")
    ax.axhline(1, color="k", linestyle=":", linewidth=0.5, label=baseline, alpha=0.8)
    ax.grid()

    ax.set_xticks(x)
    ax.margins(x=0.01)

    if xticklabels:
        _ = ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_median_and_ci(
    dist, x=None, lineages=None, colors=None, ax=None, label=None, alpha=0.2
):
    if ax is None:
        ax = plt.gca()

    if lineages is None:
        lineages = np.arange(dist.shape[-1])

    y = np.median(dist.squeeze(), 0)

    if x is None:
        x = np.arange(y.shape[0])

    ci = np.quantile(dist.squeeze(), [0.025, 0.975], 0)
    if dist.squeeze().ndim >= 3:
        for lin in lineages:

            ax.plot(x, y[:, lin], c=f"C{lin%10}", label=label)
            ax.fill_between(
                x,
                ci[0, ..., lin],
                ci[1, ..., lin],
                color=f"C{lin%10}" if colors is None else colors[lin],
                alpha=alpha,
            )
    else:
        ax.plot(x, y, c="C0" if colors is None else colors, label=label)
        ax.fill_between(
            x,
            ci[0],
            ci[1],
            color="C0" if colors is None else colors,
            alpha=alpha,
        )

    return ax


def plot_genome_proportion(
    lineage_dates, lineage_tensor, idx, lineages=None, colors=None, ax=None
):
    if ax is None:
        ax = plt.gca()
    for lin in lineages:
        ax.scatter(
            lineage_dates,
            lineage_tensor[idx, :, lin] / (lineage_tensor[idx] + 1e-16).sum(1),
            c=f"C{lin%10}" if colors is None else colors[lin],
        )

    return ax
