from genomicsurveillance import get_quote
from genomicsurveillance.quotes import quotes


def test_get_quote():
    """
    GIVEN
    WHEN get_quote is called
    THEN random quote from quotes is returned
    """

    quote = get_quote()

    assert quote in quotes
