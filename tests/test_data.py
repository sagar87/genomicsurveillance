from genomicsurveillance.data import get_meta_data
from genomicsurveillance.data.api import extract_covariate


def test_get_meta_data():
    df = get_meta_data()
    assert df.shape == (382, 13)


def test_extract_covariates(ltla_dfs):
    """
    Tests extract covariates.
    """
    specimen = extract_covariate(ltla_dfs)

    assert specimen.shape == (421, 3)
    assert "E06000022" in specimen.columns


# def test_get_specimen():
#     ltla = ['E06000021', 'E06000022', 'E06000023', 'E06000025']
#     df =  get_specimen()
#     import pdb; pdb.set_trace()
