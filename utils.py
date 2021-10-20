import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#####                              UNDER DEVELOPMENT
# Hi there, it's docstring time!
#
# When working with a team of developers, keeping track of functions and variable names can get a little overwhelming.
# Therefore, keeping a detailed docstring for classes, functions, etc is a core component of teamwork for project development.
# Another key component is to create clean and reusable blocks of code, that can be easily fixed and can be applied safely applied
# even if data slightly changes.
#
# Lez go!
#####


# From Part 1:

def clean_price_and_quantity(dataf):
    """ Only keeps items with Price and Quantity greater than 0. 

    Args:
        dataf (pd.DataFrame): input dataframe to be cleaned

    Returns:
        df (pd.DataFrame): cleaned dataframe
    """
    df = dataf.copy()
    df.query("(Price > 0) & (Quantity > 0)", inplace=True)
    return df


def clean_stockcodes (dataf):
    """ Removes invalid items (for all lengths of `StockCode`). 

    Args:
        dataf (pd.DataFrame): input dataframe to be cleaned

    Returns:
        df (pd.DataFrame): cleaned dataframe
    """
    df = dataf.copy()
    removals = []   # list of items to be removed
    aux = df['StockCode'].str.len()   # series containing the length of each value of 'StockCode'
    pass