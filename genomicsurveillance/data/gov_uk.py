from typing import Optional

import pandas as pd
from uk_covid19 import Cov19API

from genomicsurveillance.config import GovUKAPI

from .meta_data import get_meta_data


def get_specimen(ltla_list: Optional[list] = None):
    """
    Downloads the newest specimen data (newCasesBySpecimenDate) from the
    GOV UK Covid-19 api.

    :param ltla_list: A list of UK LTLA (lad19cd) codes, defaults to None
        in which case all LTLA codes are downloaded.
    :return: A (LTLA x dates) specimen table.
    """
    if ltla_list is None:
        ltla_list = get_meta_data().lad19cd.tolist()

    specimen_dfs = get_ltla_data(ltla_list)
    specimen = extract_covariate(specimen_dfs, "newCasesBySpecimenDate")
    return specimen


def extract_covariate(dataframes, covariate: str = "newCasesByPublishDate"):
    filtered = pd.concat(
        [
            current_df.drop_duplicates()
            # .assign(date=lambda df: pd.DatetimeIndex(df.date))
            .sort_values(by="date")
            .reset_index(drop=True)
            .loc[:, ["date", "areaCode", covariate]]
            for current_df in dataframes
        ]
    ).reset_index(drop=True)

    pivot = (
        filtered.pivot_table(
            index="date", columns="areaCode", values=covariate, dropna=False
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .set_index("date")
    )

    return pivot


def get_ltla_data(
    ltla_list: list, structure: dict = GovUKAPI.specimen, max_retry: int = 3
) -> list:
    """
    Downloads the the data specified in `structure` for LTLA in `ltla_list`
    from the GOV UK API. If the API does not response it will try to download
    requested data for `max_retry` times.

    :param ltla_list: A list of UK LTLA (lad19cd) codes.
    :param structure: A dictionary that specifies which data to download from the
        GOVUK api (see the documentation of uk_covid19 for more information).
    :param max_retry: Maximum number of tries to download requested data, defaults
        to 3.
    :return: A list of dataframes one for each LTLA.
    """
    dataframes = []
    for i, ltla in enumerate(ltla_list):
        # print(f"{ltla['lad19nm']}")
        retry = 1
        failure = False
        while not failure:
            try:
                api = Cov19API(filters=[f"areaCode={ltla}"], structure=structure)
                df = api.get_dataframe()
                break
            except Exception:
                print(
                    f"Failed downloading areaCode={ltla} ... retrying {retry}/{max_retry}."
                )
                retry += 1

            finally:
                if retry == max_retry:
                    failure = True

        if df.shape == (0, 0) or failure:
            continue
        dataframes.append(df)

    return dataframes
