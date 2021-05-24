from uk_covid19 import Cov19API

from genomicsurveillance.config import GovUKAPI


def get_ltla(ltla, structure: GovUKAPI.MINIMAL):
    api = Cov19API(filters=[f"areaCode={ltla}"], structure=structure)
    df = api.get_dataframe()
    return df
