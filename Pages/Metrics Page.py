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

msft = yf.Ticker("MSFT")

msft.actions