import geopandas as gpd
import pandas as pd
import io
from genomicsurveillance.config import Files


def get_geodata(geo_data: bytes = Files.GEO_JSON):
    """
    Loads the UK GeoJson file.

    :param geo_data: uk geojson data, defaults uk.geojson
        in the data dir of the package
    :returns: geopandas dataframe
    """
    uk = gpd.read_file(io.BytesIO(geo_data))
    return uk


def get_meta_data(meta_data: bytes = Files.META_DATA):
    """
    Load the UK meta data.

    :param meta_data: uk meta data, defaults csv in the data dir
        of the package
    :returns: pandas dataframe
    """
    meta = pd.read_csv(io.BytesIO(meta_data), index_col=0)
    return meta
