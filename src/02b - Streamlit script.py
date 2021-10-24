import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import *

#####     UNDER DEVELOPMENT

# Making things wide
st.set_page_config(layout="wide")

# We should start by importing our data, but we need a little something first!
# Streamlit has a nice decorator that allows us to cache stuff and increase performance.
# Yes, we could simply go through each preprocessing function from utils.py everytime this script is runned,
# but that would make debugging horribly painful.
# Therefore, let's create a function that does all the importing-cleaning-whatnot for us, and then cache it.

@st.cache
def import_and_preprocess():
    # Import data
    data = pd.read_csv("../data/online_retail_II.csv")

    # All preprocessing needed to get the df of invoices for all groups
    # Note: .pipe() allows us to apply a function to self
    base = (
        data
        .pipe(normalize_invoicedate)
        .pipe(normalize_price_and_quantity)
        .pipe(normalize_invoice)
        .pipe(normalize_stockcode)
    )

    # Get the df of invoices of all groups
    df1, df2, df3 = create_invoice_dataframe(base, group=0)
    return df1, df2, df3

# Now that all those steps are cached, let's get our dataframes of invoices
invoices1, invoices2, invoices3 = import_and_preprocess()

# We'll be using this a lot for the next selectors, so let's just declare it here
options = {
    'Group 1': invoices1,
    'Group 2': invoices2,
    'Group 3': invoices3,
    'Show all': {
            'Group 1': invoices1,
            'Group 2': invoices2,
            'Group 3': invoices3
        }
}

# So far so good. Let's start getting our dashboard rollin'
# We need a title, right?
st.title("Almighty dashboard!")   # sorry, i couldn't come up with anything else

# ...and a sidebar!
st.sidebar.title("Select visualization")
section = st.sidebar.radio(
    "Go to:",
    ("Box plot of invoices",
     "Active customers and number of orders",
    #  "Ranking de Máquinas por Riscos",
    #  "Média Móvel por Risco",
    #  "Média Móvel por Equipe",
    #  "Média Móvel por Tipo de Atividade",
    #  "Análise Comparativa da Percepção",
    #  "Sistemas de Recomendação",
    #  "Simulação APR e Recomendação",
    #  "Configurações"
    )
)

###### Boxplot of invoices ######
if section == "Box plot of invoices":
    st.header("**Box plot of invoices**")
    # Good. Now we need a selector so that people can choose which group they want to visualize
    option = st.selectbox('Select group:', options)
    st.write('')   # skip the line fandango!
    st.write("In descriptive statistics, a box plot is a type of chart often used in explanatory data analysis.")
    st.write("Box plots visually show the distribution of numerical data and [skewness](https://en.wikipedia.org/wiki/Skewness) through displaying the data quartiles (or percentiles) and averages.")
    st.write("Box plots show the five-number summary of a set of data: including the minimum score, first (lower) quartile, median, third (upper) quartile, and maximum score.")
    st.write("*Source:* [PsychologyNow](https://www.simplypsychology.org/boxplots.html)")
    st.write('')   # skip the line fandango!
    # And then the plot is shown according to the selected option: 
    if option != 'Show all':
        st.subheader(f"Box plot of invoices of {option}")
        st.plotly_chart(plot_group_box(options[option]), use_container_width=True)
    else:
        st.subheader(f"Box plots of invoices for all groups")
        fig_all = make_subplots(
            rows=3, cols=2,
            vertical_spacing=0.08,
            subplot_titles=['Group 1', 'Group 1', 'Group 2', 'Group 2', 'Group 3', 'Group 3']
        )
        for idx, df_invoices in enumerate(options['Show all'].values()):
            fig_all.add_box(y=df_invoices.Price, name='Price', row=idx+1, col=1)
            fig_all.add_box(y=df_invoices.Quantity, name='Quantity', row=idx+1, col=2)
        fig_all.update_layout(height=900, showlegend=False)
        st.plotly_chart(fig_all, use_container_width=True)

###### Active customers and number of orders ######
elif section == "Active customers and number of orders":
    st.header("**Active customers and number of orders**")
    # Our good old selector
    option = st.selectbox('Select group:', options)
    st.write('')
    # ...and the plots
    if option != 'Show all':
        st.subheader(f"Active customers and number of orders - {option}")
        st.plotly_chart(plot_active_customers_and_orders(options[option]), use_container_width=True)
    else:
        st.subheader(f"Active customers and number of orders - All groups")
        for idx, df_invoices in enumerate(options['Show all'].values()):
            st.plotly_chart(plot_active_customers_and_orders(df_invoices, f"Grupo {idx+1}"), use_container_width=True)