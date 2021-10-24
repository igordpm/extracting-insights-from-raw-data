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
# Another key component is to create clean and reusable blocks of code, that can be easily fixed and can be safely applied
# even if data slightly changes.
#
# Lez go!
#####


# From Part 01:

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_invoicedate(d_f):
    """
    Normalize 'InvoiceDate'.
    
    Casts datetime dtype into the column and then removes hours, minutes and seconds. 

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = d_f.copy()
    df = df.astype({'InvoiceDate': 'datetime64'})
    df['InvoiceDate'] = df['InvoiceDate'].dt.normalize()   # extra code! used just to remove timestamps
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_price_and_quantity(d_f):
    """
    Normalize 'Price' and 'Quantity' columns.
    
    Only keeps items with Price and Quantity greater than 0. 

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe with columns to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = d_f.copy()
    df = d_f.query("(Price > 0) & (Quantity > 0)")
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_invoice(d_f):
    """
    Normalize 'Invoice' column.
    
    Removes the only non-full-number invoice ('C496350') and casts dtype int32 into the Invoice column. 

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = d_f.query("Invoice != 'C496350'").astype({"Invoice": "int32"})
    return df

# This is just a copypasta of code from part 01, check the notebook for detailed info:
def normalize_stockcode(d_f):
    """
    Normalize 'StockCode' column.
    
    Puts all characters in upper case, fixes all '47503J ' invoice entries to '47503J' and removes invalid items (for all lengths of `StockCode`). 

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe with column to be normalized.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = d_f.copy()
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
        raise ValueError("'group' parameter must be an int between 0 and 3")

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
        removed_invoices = invoices.query('(Quantity > @UOF_quantity) | (Price > @UOF_price)').index.tolist()
        invoices2 = invoices.query("index not in @removed_invoices")
        # Group 3:
        removed_invoices = invoices.query('(Quantity > @UIF_quantity) | (Price > @UIF_price)').index.tolist()
        invoices3 = invoices.query("index not in @removed_invoices")
        if group == 1:
            return invoices
        if group == 2:
            return invoices2
        if group == 3:
            return invoices3
        if group == 0:
            return invoices, invoices2, invoices3

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
def plot_active_customers_and_orders(df, title=None):
    """
    Generate a single bar plot of monthly active customers and monthly orders.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.
    title : str, optional
        Title of plot.

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
    fig.update_layout(title=title,
                    title_x=0.5,
                    xaxis_title='Month',
                    yaxis_title='Amount')
    return fig

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_avg_revenue(df, title=None):
    """
    Generate a bar plot of the monthly average revenue per order.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.
    title : str, optional
        Title of plot.

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
    fig.update_layout(title=title,
                    title_x=0.5,
                    xaxis_title='Month',
                    yaxis_title='Revenue')
    return fig

# Now things start to get messy!
# Most of this is just copypasta from part 02, but now we are changing our database.
# Data from NaN customers were taken into account in monthly analyses, such as revenue and number of orders.
# However, for detailed customer analysis, we dropped all NaN ID's, remember?
# (You can check the notebook from part 02 for further details)
# Therefore, we must write a helper function that does all that preprocessing before any other action.
# And yes, we might need to call it inside any other function that does not accept NaN ID's.
# Here it goes!
def clean_customer_id(d_f):
    """
    Remove NaN's from 'Customer ID' column.

    Drops all invoices with NaN as Customer ID and prettifies the column.  

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """
    df = d_f.dropna(subset=['Customer ID'])
    df['Customer ID'] = pd.to_numeric(df['Customer ID']).astype(int).astype(str)
    return df

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def get_month_name(df):
    """
    Convert datetime into 3-letter month name.

    This helper function converts the `InvoiceDate` column to a 3-letter word of the respective month.  

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of invoices.

    Returns
    -------
    pandas.DataFrame
        Dataframe of invoices with modified dates.
    """
    df2 = df.copy().sort_values(by='InvoiceDate')
    df2['InvoiceDate'] = df2['InvoiceDate'].dt.month_name()   # converts to month name
    df2['InvoiceDate'] = df2['InvoiceDate'].str[:3]   # gets the first 3 characters
    return df2

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_revenue_by_type(d_f, title=None):
    """
    Generate a line plot of revenue by customer type.

    Creates needed columns to plot the revenue generated by each type of customer and the total revenue, by month.  

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe of invoices.
    title : str, optional
        Title of plot.

    Returns
    -------
    graph_objects.Figure
        Figure containing three line plots.
    """
    df = clean_customer_id(d_f)   # preprocessing
    first_month_series = (
        df
        .groupby('Customer ID')
        ['InvoiceDate']
        .min()
        .dt.month
        .rename('FirstMonth')
    )
    df = df.merge(first_month_series, on='Customer ID').set_axis(df.index)
    # Let's start by assuming all customers are new ones
    df['CustomerType'] = 'New'
    # Getting all entries where the condition is met, and then getting the index of this resulting dataframe
    aux_index = df.query("InvoiceDate.dt.month > FirstMonth").index
    # Changing `UserType` of the previously selected invoices
    df.loc[aux_index, 'CustomerType'] = 'Existing'

    invoices_by_type = (
        df.pipe(get_month_name)
        .groupby(['CustomerType', 'InvoiceDate'], as_index=False, sort=False)
        ['Price']
        .sum()
    )
    fig = go.Figure(data=[
        go.Scatter(
            name='New',
            y=invoices_by_type.query("CustomerType == 'New'")['Price'],
            x=invoices_by_type.query("CustomerType == 'New'")['InvoiceDate']),
        go.Scatter(
            name='Existing',
            y=invoices_by_type.query("CustomerType == 'Existing'")['Price'],
            x=invoices_by_type.query("CustomerType == 'Existing'")['InvoiceDate']),
        go.Scatter(
            name='Total',
            y=invoices_by_type.groupby("InvoiceDate", sort=False)['Price'].sum(),
            x=invoices_by_type.groupby("InvoiceDate", sort=False)['Price'].sum().index)
    ])
    fig.update_layout(title=title,
                    title_x=0.5,
                    xaxis_title='Month',
                    yaxis_title='Revenue')
    return fig

# This is just a copypasta of code from part 02, check the notebook for detailed info:
def plot_retention_rate(d_f, title=None):
    """
    Generate a line plot of retention rate by month.  

    Parameters
    ----------
    d_f : pandas.DataFrame
        Dataframe of invoices.
    title : str, optional
        Title of plot.

    Returns
    -------
    graph_objects.Figure
        Figure containing a line plot.
    """
    active_months = (
        d_f
        .pipe(get_month_name)   # using that old helper function
        .groupby(['Customer ID','InvoiceDate'], as_index=False, sort=False)['Price']
        .sum()
    )
    ordered_months = active_months['InvoiceDate'].unique().tolist()   # list of months
    retention = (
        pd.crosstab(
            index=active_months['Customer ID'], 
            columns=active_months['InvoiceDate']
            )
        .reindex(columns=ordered_months)   # crosstab messes our month orders, ugh!
    )
    retention_array = []
    for i in range(len(ordered_months)-1):
        retention_data = {}
        selected_month = ordered_months[i+1]
        previous_month = ordered_months[i]
        retention_data['Month'] = selected_month
        retention_data['TotalCustomers'] = retention[selected_month].sum()
        retention_data['RetainedCustomers'] = retention.query(f"({selected_month} > 0) & ({previous_month} > 0)")[selected_month].sum()
        retention_array.append(retention_data)
    retention = pd.DataFrame(retention_array)
    retention['RetentionRate'] = retention['RetainedCustomers']/retention['TotalCustomers']
    fig = (
        px.line(
            data_frame=retention,
            y='RetentionRate',
            x='Month')
    )
    fig.update_layout(title=title,
                    title_x=0.5
    )
    return fig