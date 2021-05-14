import pkgutil


class Files:
    META_DATA = pkgutil.get_data(__name__, "data/uk_meta.csv")
    GEO_JSON = pkgutil.get_data(__name__, "data/uk.geojson")
    ALIASES = pkgutil.get_data(__name__, "data/aliases.txt")


class GovUKAPI:
    ENDPOINT = "https://api.coronavirus.data.gov.uk/v1/data"
    AREA_TYPE = "ltla"

    MINIMAL = {
        "date": "date",
        "areaName": "areaName",
        "areaCode": "areaCode",
        "newCasesBySpecimenDate": "newCasesBySpecimenDate",
    }

    FULL = {
        "date": "date",
        "areaName": "areaName",
        "areaCode": "areaCode",
        "newPillarOneTestsByPublishDate": "newPillarOneTestsByPublishDate",
        "newCasesBySpecimenDate": "newCasesBySpecimenDate",
        "newPillarTwoTestsByPublishDate": "newPillarTwoTestsByPublishDate",
        "newPillarThreeTestsByPublishDate": "newPillarThreeTestsByPublishDate",
        "newPillarFourTestsByPublishDate": "newPillarFourTestsByPublishDate",
        "hospitalCases": "hospitalCases",
        "newDeathsByDeathDate": "newDeathsByDeathDate",
        "newCasesByPublishDate": "newCasesByPublishDate",
        "cumCasesByPublishDate": "cumCasesByPublishDate",
        "newDeathsByDeathDate": "newDeathsByDeathDate",
        "cumDeathsByDeathDate": "cumDeathsByDeathDate",
    }
