def time_to_str(date):
    """
    Helper function to convert pd.DateTime objects into str.

    :param date: A pd.DateTime object
    :returns: A string in the formal 'YYYY-MM-DD'
    """
    return str(date)[:10]
