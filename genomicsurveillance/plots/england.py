"""
Convinience Plotting codes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genomicsurveillance.data import get_england

from .basic import plot_median_and_ci


def plot_lad(
    model,
    lad,
    cases,
    lineage_tensor,
    lin_date_idx,
    start_date="2020-09-01",
    show_start_date="2020-09-01",
    show_end_date="2020-09-01",
    lin=[0, 1],
    colors=["C1", "C2", "C3", "C4", "C5"],
    labels=["B.1.1.7", "Other"],
):
    eng = get_england()
    date_range = pd.date_range(start_date, periods=cases.shape[1])

    fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    plot_median_and_ci(model.get_lambda(lad), x=date_range, ax=ax[0])
    lamb_lin = model.get_lambda_lineage(lad)

    ax[0].set_title(eng.lad19nm.values[lad])
    for i, l in enumerate(lin):
        j = lin_date_idx[(lineage_tensor[lad, ..., l] > 0).argmax()]
        plot_median_and_ci(
            lamb_lin[..., l][..., j:],
            ax=ax[0],
            x=date_range[j:],
            colors=colors[i],
            label=labels[i],
        )
    ax[0].legend()
    ax[0].plot(date_range, cases[lad], "C0.")
    ax[0].margins(x=0.001)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("New cases")
    secax = ax[0].secondary_yaxis(
        "right",
        functions=(
            lambda x: x / eng.pop18.values[lad] * 1e5,
            lambda x: x / 1e5 * eng.pop18.values[lad],
        ),
    )
    secax.set_ylabel("Cases per 100k")

    prop = model.get_probabilities(lad)
    for i, l in enumerate(lin):
        j = lin_date_idx[(lineage_tensor[lad, ..., l] > 0).argmax()]
        if j > 21:
            jj = j - 21
        else:
            jj = j
        plot_median_and_ci(
            prop[..., l].squeeze()[:, jj:],
            colors=colors[i],
            x=date_range[jj:],
            ax=ax[1],
        )

        j = (lineage_tensor[lad, ..., l] > 0).argmax()
        ax[1].scatter(
            date_range[lin_date_idx][j:],
            (lineage_tensor[lad, :, l] / (lineage_tensor[lad] + 1e-16).sum(1))[j:],
            c=f"C{lin%10}" if colors is None else colors[i],
            label="Data" if i == 0 else None,
        )
    ax[1].legend()
    ax2 = ax[1].twinx()
    ax2.bar(
        date_range[lin_date_idx],
        lineage_tensor[lad].sum(-1),
        color="C7",
        width=7,
        alpha=0.2,
    )
    ax2.set_ylabel("Genomes")
    ax[1].set_ylabel("Probability")
    ax[1].margins(x=0.001)
    ax[1].set_xticklabels([])
    ax[1].set_ylim(bottom=0, top=1)

    R = np.exp(model.get_log_R_lineage(lad))
    R0 = np.exp(model.get_log_R(lad)).squeeze()

    plot_median_and_ci(
        R0[:, :-7], colors="C0", x=date_range[:-7], ax=ax[2], label="Avg. R"
    )
    for i, l in enumerate(lin):
        j = lin_date_idx[(lineage_tensor[lad, ..., l] > 0).argmax()]
        plot_median_and_ci(
            R[..., l][..., j:-7],
            colors=colors[i],
            x=date_range[j:-7],
            ax=ax[2],
            alpha=0.05,
        )

    ax[2].set_ylim([0.5, 3])
    ax[2].axhline(1, color="k")
    ax[2].margins(x=0.001)
    ax[2].axvspan(
        pd.to_datetime("2020-11-05"),
        pd.to_datetime("2020-12-01"),
        color="C7",
        alpha=0.1,
    )
    ax[2].axvspan(
        pd.to_datetime("2021-01-06"),
        pd.to_datetime("2021-03-07"),
        color="C7",
        alpha=0.1,
    )
    ax[2].grid(True)
    ax[2].set_ylabel("R")
    ax[2].set_xlabel("Date")
    ax[2].legend()

    ax[0].set_xlim(left=pd.to_datetime(show_start_date))
    ax[1].set_xlim(left=pd.to_datetime(show_start_date))
    ax[2].set_xlim(left=pd.to_datetime(show_start_date))
    fig.autofmt_xdate(ha="center")
