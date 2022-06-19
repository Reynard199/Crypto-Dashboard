# streamlit run "/Users/danreeynard/Documents/Crypto_Dash_Board/Metrics Page.py"

import streamlit as st
import pandas as pd
from pandas_datareader import data as web
from datetime import timedelta, date
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from fear_greed_index.CNNFearAndGreedIndex import CNNFearAndGreedIndex
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pytz

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    
    msft = yf.Ticker("MSFT")

    msft.actions

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

