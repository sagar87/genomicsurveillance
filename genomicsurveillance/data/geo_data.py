import io

import geopandas as gpd

from genomicsurveillance.config import Files


def get_geo_data(geo_data: bytes = Files.GEO_JSON):
    """
    Loads the UK GeoJson file. Requires geopandas.

    :param geo_data: uk geojson data, defaults uk.geojson
        in the data dir of the package
    :returns: geopandas dataframe
    """
    uk = gpd.read_file(io.BytesIO(geo_data))
    return uk
