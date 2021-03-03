import random

from genomicsurveillance.quotes import quotes


def get_quote() -> dict:
    """
    Get random quote

    Get randomly selected quote from database our programming quotes

    :return: selected quote
    :rtype: dict
    """

    return quotes[random.randint(0, len(quotes) - 1)]
