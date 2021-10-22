import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statistics import mode

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


# From Part 01:

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_invoicedate(dataf):
    """
    Normalize 'InvoiceDate'.
    
    Casts datetime dtype into the column and then removes hours, minutes and seconds. 

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = dataf.copy()
    df = df.astype({'InvoiceDate': 'datetime64'})
    df['InvoiceDate'] = df['InvoiceDate'].dt.normalize()   # extra code! used just to remove timestamps
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_price_and_quantity(dataf):
    """
    Normalize 'Price' and 'Quantity' columns.
    
    Only keeps items with Price and Quantity greater than 0. 

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe with columns to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = dataf.query("(Price > 0) & (Quantity > 0)", inplace=True)
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_invoice(dataf):
    """
    Normalize 'Invoice' column.
    
    Removes the only non-full-number invoice ('C496350') and casts dtype int32 into the Invoice column. 

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = dataf.query("Invoice != 'C496350'").astype({"Invoice": "int32"})
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_stockcode(dataf):
    """
    Normalize 'StockCode' column.
    
    Puts all characters in upper case, fixes all '47503J ' invoice entries to '47503J' and removes invalid items (for all lengths of `StockCode`). 

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = dataf.copy()
    df['StockCode'] = df['StockCode'].str.upper()
    df.loc[df['StockCode'] == '47503J ', 'StockCode'] = '47503J'
    removals = ['D',
    'm',
    'S',
    'M',
    'C2',
    'DOT',
    'PADS',
    'POST',
    'TEST001',
    'TEST002',
    'ADJUST2',
    'AMAZONFEE',
    'BANK CHARGES',
    'ADJUST']   # list of items to be removed
    df = df.query("StockCode not in @removals")
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def create_invoice_dataframe(df, group=1):
    """
    Create dataframes of invoices for the specified group.
    
    The full details of each group can be seen in notebook of section 01. Quick reminder:
    Group 1: all entries after the data cleaning from section 01;
    Group 2: all extreme outliers removed;
    Group 3: all mild outliers removed (and, consequently, all extreme outliers as well).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of items to be grouped by invoices.
    group : {0, 1, 2, 3}, default 1
        Group number of the desired output dataframe. '0' returns the dataframes for all groups, in the format: group1, group2, group3

    Returns
    -------
    pandas.DataFrame
        Dataframe grouped by invoices.
    
    Raises
    ------
    ValueError
        If `group` is not valid.
    """
    if group not in [0, 1, 2, 3]:
        raise ValueError("'group' parameter must be an int between 0 a 3")

    else:
        invoices = df.groupby(['Invoice']).agg({'Quantity': 'sum',
                                            'Price': 'sum',
                                            'Customer ID': mode,
                                            'InvoiceDate': mode})
        # Interquartile range, inner and outer fences:
        IQ = invoices.quantile(.75) - invoices.quantile(.25)
        IQ_quantity = IQ['Quantity']
        IQ_price = IQ['Price']
        UIF_quantity = invoices.Quantity.quantile(.75) + 1.5 * IQ_quantity
        UOF_quantity = invoices.Quantity.quantile(.75) + 3 * IQ_quantity
        UIF_price = invoices.Price.quantile(.75) + 1.5 * IQ_price
        UOF_price = invoices.Price.quantile(.75) + 3 * IQ_price
        # Group 2:
        df2 = invoices.query('(Quantity <= @UOF_quantity) | (Price <= @UOF_price)')
        # Group 3:
        df3 = invoices.query('(Quantity <= @UIF_quantity) | (Price <= @UIF_price)')

        if group == 1:
            return invoices
        if group == 2:
            return df2
        if group == 3:
            return df3
        if group == 0:
            return invoices, df2, df3

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def plot_group_box(df):
    """
    Generate a box plot of 'Price' and 'Quantity'. 
    
    Both box plots are shown side by side, in the same figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    graph_objects.Figure
        Figure containing both box plots.
    """
    fig = make_subplots(rows=1, cols=2)
    fig.add_box(y=df.Price, name='Price', row=1, col=1)
    fig.add_box(y=df.Quantity, name='Quantity', row=1, col=2)
    return fig


# From Part 02:

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_monthly_resample(df):
    """
    Generate a bar plot of 'Price' and 'Quantity' grouped by month. 
    
    Both bar plots are shown side by side, in the same figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    graph_objects.Figure
        Figure containing both boxplots.
    """
    resampled_df = df.resample(rule='M', on='InvoiceDate').sum()   # grouping by month
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(y=resampled_df.Price, x=resampled_df.index, name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(y=resampled_df.Quantity, x=resampled_df.index, name='Quantity'), row=1, col=2)
    # by default, the last day of the month is shown, but the data represents the whole month
    return fig

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def adjust_time_window(df):
    """
    Filter invoices to appopriate time window (Jan/2010 to Nov/2010).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    pandas.DataFrame
        Dataframe within desired timespan.
    """
    adjusted_df = df.query("(InvoiceDate > '2009-12-31') & (InvoiceDate < '2010-12-01')").sort_values('InvoiceDate')
    return adjusted_df

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_active_customers_and_orders(df):
    """
    Generate a single bar plot of monthly active customers and monthly orders.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    graph_objects.Figure
        Figure containing a single bar plot, with both data.
    """
    df_monthly_customers = (
            df[['Customer ID']]
            .groupby(df['InvoiceDate'].dt.month_name().str[:3], sort=False)
            .nunique()
    )

    df_monthly_orders = (
            df[['Price']]   # selecting any column (we just need the number of entries for each month, since each entry is a unique invoice number)
            .groupby(df['InvoiceDate'].dt.month_name().str[:3], sort=False)
            .count()
    )   # the column for this dataframe is named 'Price', but we know that it is not the case, right?

    fig = go.Figure(data=[
            go.Bar(name='Monthly Active Customers', y=df_monthly_customers['Customer ID'], x=df_monthly_customers.index),
            go.Bar(name='Monthly Orders', y=df_monthly_orders['Price'], x=df_monthly_orders.index)
    ])
    return fig

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_avg_revenue(df):
    """
    Generate a bar plot of the monthly average revenue per order.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    graph_objects.Figure
        Figure containing a bar plot.
    """
    df_avg_revenue = (
        df[['Price']]
        .groupby(df['InvoiceDate'].dt.month_name().str[:3], sort=False)
        .mean()
    )
    fig = (
    px.bar(data_frame=df_avg_revenue, y='Price')
    )
    return fig

# Now things start to get messy!
# Most of this is just copypasta from part 02, but now we are changing our database.
# Data from NaN customers were taken into account in monthly analyses, such as revenue and number of orders.
# However, for detailed customer analysis, we dropped all NaN ID's, remember?
# (You can check the notebook from part 02 for further details)
# Therefore, we must write a helper function that does all that preprocessing before any other action.
# And yes, we might need to call it inside any other function that does not accept NaN ID's.
# Here it goes!
def clean_customer_id(dataf):
    """
    Remove NaN's from 'Customer ID' column.

    Drops all invoices with NaN as Customer ID and prettifies the column.  

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = dataf.dropna(subset=['Customer ID'])
    df['Customer ID'] = pd.to_numeric(df['Customer ID']).astype(int).astype(str)
    return df

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_customer_rate(dataf):
    """
    Remove NaN's from 'Customer ID' column.

    Drops all invoices with NaN as Customer ID and prettifies the column.  

    Parameters
    ----------
    dataf : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = clean_customer_id(dataf)   # preprocessing
    first_month_series = (
        invoices1
        .groupby('Customer ID')
        ['InvoiceDate']
        .min()
        .dt.month
        .rename('FirstMonth')
        )
    df = df.merge(first_month_series, on='Customer ID').set_axis(df.index)
    pass