import os
import pkgutil


class Files:
    META_DATA = pkgutil.get_data(__name__, "data/uk_meta.csv")
    GEO_JSON = pkgutil.get_data(__name__, "data/uk.geojson")
