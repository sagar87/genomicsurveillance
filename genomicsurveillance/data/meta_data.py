import io

import pandas as pd

from genomicsurveillance.config import Files


def get_aliases(aliases: bytes = Files.ALIASES):
    aliases = pd.read_csv(io.BytesIO(aliases))
    return dict(zip(aliases.alias.tolist(), aliases.lineage.tolist()))


def get_merged_delta_aliases(aliases: bytes = Files.ALIASES):
    aliases = pd.read_csv(io.BytesIO(aliases))
    aliases = dict(zip(aliases.alias.tolist(), aliases.lineage.tolist()))

    for k, v in aliases.items():
        if k.startswith("AY."):
            aliases[k] = "B.1.617.2"

    return aliases


def get_meta_data(meta_data: bytes = Files.META_DATA):
    """
    Load the UK meta data.

    :param meta_data: uk meta data, defaults csv in the data dir
        of the package
    :returns: pandas dataframe
    """
    meta = pd.read_csv(io.BytesIO(meta_data), index_col=0)
    return meta


def get_england():
    """
    Returns meta data for england only.
    """
    uk = get_meta_data()
    eng = uk[uk.ctry19nm == "England"]
    return eng
