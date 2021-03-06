# streamlit run "/Users/danreeynard/Documents/Crypto_Dash_Board/CS_CryptoDashBoard.py"

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

st.set_page_config(layout = 'wide', initial_sidebar_state = 'expanded', page_title = "Dan's Crypto Dashboard", page_icon = "https://raw.githubusercontent.com/Reynard199/Crypto-Dashboard/main/Photos/Pizza%20Angel%20Icon.jpg")

# setting a timezone for the app
tz = pytz.timezone('Africa/Johannesburg')

col_1, col_2= st.columns(2)

with col_1 :
    st.image(
        "https://raw.githubusercontent.com/Reynard199/Crypto-Dashboard/main/Photos/Pizza%20Angel%20Icon.jpg",
        use_column_width = True,
        )

with col_2 :
    st.write("""
        # Pizza Angel's Crypto and Commodity DashBoard
        **Made for the Idiots, by an Idiot. Geniet.**
        """)
    with st.expander('Brief Overview : How to use dash', expanded = False) :
        st.write("1) The Control Panel is where all the viables are inserted into the dashboard.")
        st.write('2) Set the App background to dark mode in the setting on the top right handside of the Dash.')
        st.write("3) The Legend can be used to Hide / Display certain stock when the line is clicked on.")
        st.write("4) Click on the 'Expand' buttons to receive explanations.")
        st.write("5) This App is best viewed on larger screen.")
        
    
st.sidebar.header("*Control Panel*")

def get_input():
    crypto_symbol = st.sidebar.selectbox('Crypto Coin or Commodity', options = ['BTC-USD', 'DOGE-USD', 'ETH-USD', 'GC=F', 'CL=F'])
    with st.sidebar.expander('Note about Date Selection', expanded = False) :
        st.write('Some of the tickers have different dates when they where created, so please use an ending value later than 1 Jan 2018. Otherwise it breaks the dash. Thanks!')
    start_date = st.sidebar.date_input("Start Date", value = datetime.date(2021,1,1), max_value = (datetime.date.today() - datetime.timedelta(days = 30)), min_value = datetime.date(2015,1,1))
    end_date = st.sidebar.date_input("End Date", value = datetime.date.today(), max_value = datetime.date.today(), min_value = start_date + datetime.timedelta(days = 30))
    selected_stock = st.sidebar.text_input('Select a Ticker as per the Yahoo Finance Ticker Format (ABG.JO is ABSA)', 'ABG.JO').upper()
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'][1:-1]
    ticker_list = ticker_list.append(pd.Series(['^GSPC', 'ETH-USD', 'DOGE-USD', 'BTC-USD', '^J203.JO', 'CL=F', 'GC=F', selected_stock]))
    ticker_list = ticker_list.unique()
    ticker = st.sidebar.multiselect('Selection of Ticker Prices', options = ticker_list, default = ['BTC-USD', 'ETH-USD', 'DOGE-USD', '^GSPC', 'CL=F', 'GC=F', selected_stock])
    if ((end_date - start_date).days) > 120 :
               x = 120
    else : x = ((end_date - start_date).days)
    moving_averages = st.sidebar.slider(label = 'Moving Averages Time Period',
                    min_value = (3),
                    max_value=(x),
                    value = (7),
                    step=(1))
    
    return start_date, end_date, crypto_symbol, moving_averages, ticker, selected_stock

def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC-USD" :
        return 'Bitcoin'
    if symbol == 'ETH-USD' :
        return 'Ethereum'
    if symbol == "DOGE-USD" :
        return 'Dogecoin'
    if symbol == 'GC=F' :
        return 'Gold'
    if symbol == 'CL=F' :
        return 'Crude Oil'
    else :
        return "None"

def get_data(symbol, start_date, end_date) :
    symbol = symbol.upper()
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if symbol == "BTC-USD" :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
    elif symbol == "ETH-USD" :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
    elif symbol == 'DOGE-USD' :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
    elif symbol == 'CL=F' :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
    elif symbol == 'GC=F' :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
    else :
        df = pd.DataFrame(columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    
    df['Date'] = df.index
    
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Total Returns (%)'] = (df['Close'] / df["Close"].iloc[0] - 1) * 100
    df['Daily Returns (%)'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    
    cols = df.columns.tolist()
    reorder = cols[0:4] + cols[5:6] + cols[4:5] + cols[6:]
    df = df[reorder]
    
    return df.loc[start_date : end_date]


start, end, symbol, moving_averages, ticker, selected_stock = get_input()
df = get_data(symbol, start, end)

crypto_name = get_crypto_name(symbol)

fig = go.Figure(
    data = [go.Candlestick(
        x = df.index,
        open = df['Open'],
        high = df['High'],
        close = df['Close'],
        low = df['Low'],
        increasing_line_color = 'green',
        decreasing_line_color = 'red'
        )])

st.markdown("***")

st.header('Return Comparisons between ' + crypto_name + ' and Other S&P500 Stock')
st.subheader("First is the Historical Data of the " + crypto_name + ' in Days')
def comparison_pricing(ticker) :
    comparison_pricing_df = pd.DataFrame()
    for i in ticker :
        comparison_pricing_df[i] = web.DataReader(i, data_source = 'yahoo', start = start, end = end)['Close']
    return comparison_pricing_df

comparison_pricing_df = comparison_pricing(ticker)
comparison_pricing_df[symbol] = df['Adj Close']
comparison_returns = pd.DataFrame()
ticker_title = ""
comparison_pricing_plot = make_subplots(specs=[[{"secondary_y": True}]])
for i in ticker :
    comparison_returns[i] = (comparison_pricing_df[i] / comparison_pricing_df[i].loc[comparison_pricing_df[i].first_valid_index()] - 1) * 100
    if comparison_returns[i].max() > 1000 :
        comparison_pricing_plot.add_trace(
            go.Scatter(x = comparison_returns[i].index, y = comparison_returns[i], mode = 'lines', connectgaps = True, name = i),
            secondary_y=True)
        ticker_title += str(i)
        ticker_title += ', '
    else : comparison_pricing_plot.add_trace(go.Scatter(x = comparison_returns.index, y = comparison_returns[i], mode = 'lines', name = i, connectgaps = True), secondary_y=False)

    
comparison_pricing_plot.update_layout(
        title = ("The Daily Returns Data for " + crypto_name + " and Comparison Stock"),
        plot_bgcolor = 'rgba(1,1,1,1)',
        title_x = 0.5,
        xaxis_title = ("Dates between " + str(start) + ' and ' + str(end)),
        yaxis_title = "Percentage Returns (%)",
        legend_title = 'Ticker Legend')

comparison_pricing_plot.update_yaxes(rangemode='tozero', constraintoward='bottom', title_text = ('Percentage Returns (%) of ' + ticker_title))
st.plotly_chart(comparison_pricing_plot, use_container_width = True)

with st.expander('Commentary on the Returns of each crypto as well as some equity :', expanded = False) :
    st.write("I would recommend hiding DOGE-USD (when using long periods of comparison data) for the time being due to its weird (but correct and accurate) return statistic")
    
st.subheader('Secondly is the Historical Data of ' + crypto_name + ' in Minutes.')
st.write('Please be kind to this section of the app - the work around the dates ranges is in HEAVY development Beta.')
with st.expander('Reveal the Guidelines on Dates', expanded = False) :
    st.write('Data returned in 1 minute intervals can only be called for a time frame of 1 week.')

col_1, col_2= st.columns(2)
with col_1 :
    start_date = st.date_input("Start Time", value = datetime.datetime.now(tz = tz) - datetime.timedelta(days = 4), min_value = datetime.date.today() - datetime.timedelta(days = 7), max_value = datetime.date.today() - datetime.timedelta(days = 1))
    if (start_date + datetime.timedelta(days = 7)) > datetime.date.today() :
        max_value_determined = datetime.datetime.now()
    else :
        max_value_determined = start_date + datetime.timedelta(days = 7)
with col_2 :
    end_date = st.date_input("End Time", value = datetime.datetime.now(), max_value = max_value_determined, min_value = start_date + datetime.timedelta(days = 1))

ticker_str = ""
for k in ticker :
    ticker_str += str(k)
    ticker_str += ' '
short_data = yf.download(ticker_str, start = start_date, end = end_date, interval = "1m")['Close']
short_data_returns = pd.DataFrame()
short_data_returns_plot = go.Figure()
for i in ticker :
    short_data_returns[i] = (short_data[i] / short_data[i].loc[short_data[i].first_valid_index()] - 1) * 100
    # beginning_seconds, ending_seconds = st.slider('Select a time length to be displayed (in minutes)', value = short_data_returns.index[-60], min_value = short_data_returns.index[0], max_value = short_data_returns.index[-1], step = datetime.timedelta(minutes = 60))
    short_data_returns_plot.add_trace(
    go.Scatter(x = short_data_returns.index, y = short_data_returns[i], mode = 'lines', connectgaps = True, name = i),
    )
short_data_returns_plot.update_layout(
    title = ("The 1 Minute Returns Data for " + crypto_name + " and Comparison Instruments"),
    plot_bgcolor = 'rgba(1,1,1,1)',
    title_x = 0.5,
    xaxis_title = ("Dates between " + str(start) + ' and ' + str(end)),
    yaxis_title = "Percentage Returns (%)",
    legend_title = 'Ticker Legend')
st.plotly_chart(short_data_returns_plot, use_container_width = True)

st.markdown('---')

def color_df(val):
    color = 'red' if val <= -3 else 'orange' if -3 < val < 0 else 'yellow' if 0 < val < 3 else 'green'
    return 'color: %s' % color
    
st.header(crypto_name + ' Data for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
st.dataframe(df.sort_values(by = 'Date', ascending=False).drop(columns = ['Date', 'Year', 'Month', 'High', 'Low', "Open"]).style.applymap(color_df, subset = ['Daily Returns (%)']))

st.markdown('---')

st.header(crypto_name + ' Volume for ' + str(start) + ' to ' + str(end))
with st.expander('Click to Reveal ' + crypto_name + ' Volume', expanded = False) :
    st.bar_chart(data = df['Volume'], use_container_width = True)

st.markdown("---")

st.header('Return Correlation Matrix and Heatmap of ' + crypto_name)

col_1, col_2 = st.columns(2)
with col_1 :
    st.subheader('Daily Data')
    comparison_returns_corr = comparison_returns.corr()
    comparison_returns_corr_plot, ax = plt.subplots()
    sns.heatmap(comparison_returns_corr, ax=ax, cmap="Reds", annot = True)
    st.write(comparison_returns_corr_plot)
with col_2 :
    st.subheader('Minute Data')
    short_data_returns_corr = short_data_returns.corr()
    short_data_returns_corr_plot, ax = plt.subplots()
    sns.heatmap(short_data_returns_corr, ax=ax, cmap="Blues", annot = True)
    st.write(short_data_returns_corr_plot)
    
st.markdown("***")

st.header(crypto_name + ' Closing Price Statistics for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
with st.expander('Reveals the ' + crypto_name + ' Price Statisitics', expanded = False) :
    st.table(df.groupby(df.Date.dt.year)['Adj Close'].describe().transpose())
    st.subheader('Closing Price of ' + crypto_name)
    st.plotly_chart(px.line(df['Adj Close']), use_container_width = True)

st.markdown("***")

st.header(crypto_name + ' Returns Statistics for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
return_stats = pd.DataFrame(df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').sum())
return_stats["Annual Standard Deviation"] = df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').std()
return_stats["Average Return (%)"] = df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').mean()
return_stats["Median Annual Return (%)"] = (df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').median())
return_stats["Number of Days"] = (df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').count())
def dammit(y): 
    x = []
    for i in df['Year'].unique():
       (x.append(str(i)))
    return x
unique_year = dammit(df['Year'].unique())
selected_year = st.multiselect(label = 'Select Year to inspect Returns Distribution', options =  unique_year, default = (unique_year[-1]))
conditions = [
    (df['Daily Returns (%)'] <= -5),
    (df['Daily Returns (%)'] > -5) & (df['Daily Returns (%)'] <= -3),
    (df['Daily Returns (%)'] > -3) & (df['Daily Returns (%)'] <= 0),
    (df['Daily Returns (%)'] > 0) & (df['Daily Returns (%)'] <= 3),
    (df['Daily Returns (%)'] > 3)
    ]
values = ['Dammit', 'OH NO!', 'It Is What It Is', "We'll Take It", 'I <3 Cryptos']
df['Commentary'] = np.select(conditions, values)
for i in selected_year : 
    return_stats_plot = px.histogram(df[df['Year'] == int(i)],
                    x = 'Daily Returns (%)', 
                    color = 'Commentary', 
                    nbins = 100,
                    category_orders = dict({"Commentary" : ['Dammit', 'OH NO!', 'It Is What It Is', "We'll Take It", 'I <3 Cryptos']}),
                    color_discrete_sequence=["red", "orange", "yellow", "blue", "green"])
    return_stats_plot.update_layout(
            plot_bgcolor = 'rgba(1,1,1,1)',
            title_x = 0.5,
            xaxis_title = ("Daily Returns (%) of " + i),
            yaxis_title = "Frequency of Daily Returns",
            legend_title = "Legend",
        )
    return_stats_plot.update_xaxes(range = [-25,25])
    st.plotly_chart(return_stats_plot, use_container_width = True)

# st.write('The number of days that earned a loss of 5% or greater was ' + str((df['Daily Returns (%)'] <= -5).count() + ' Days')

st.subheader('Count of the number of each type of return day')
with st.expander('Reveal the number of each type of return day', expanded = False) :
    return_types = st.table(df['Commentary'].value_counts()[:-1])

st.subheader('Basic Return Stats for ' + crypto_name)
st.table(return_stats.transpose())

st.markdown('---')

# pie_plot = px.pie(df, values = return_types[:-1], names = ['Fuck', 'Oh Shit', 'It Is What It Is', "We'll Take It", 'I <3 Cryptos'])
# st.plotly_chart(pie_plot, use_container_width = True)

# st.markdown('---')

# st.header(crypto_name + " CandleStick Chart for " + str(datetime.date.strftime(start, '%d %B %Y') + " to " + str(datetime.date.strftime(end, '%d %B %Y'))))
# candle_stick = st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header(crypto_name + ' : Basic "Technical" Trading')
with st.expander("Click to Expand Explanation of Moving Averages Trading Strategy", expanded = False) :
    explanation = "<p style='font-family:Times New Roman; font-size: 16px;'>A SMA-EWM Trading Strategy has simple underlying principles. When the EWM (Exponential Weighted Moving Average) crosses the SMA (Simple Moving Average), " \
        + str(crypto_name) + " is bought at the crossover price, indicated in green. The opposite action is taken if the EWM crosses the SMA, indicated in red. The number of periods that the weighted averages are calculated over is \
            controlled by the 'Moving Averages Time Period' slider, in this case being " + str(moving_averages) + " days.</p>"
    st.write(explanation, unsafe_allow_html=True)
    st.markdown("    ")
    st.markdown("<p style='font-family:Times New Roman; font-size: 16px;'>General Note - This trading strategy is rarely effective (Thanks Weak Efficient Market Hypothesis), but it was certainly interesting to code.<p>", unsafe_allow_html = True)

with st.container() :
    
    def trading(moving_averages) : 
        trading_df = pd.DataFrame()
        
        trading_df['Close'] = df['Close']
        trading_df["SMA"] = df["Close"].rolling(window = moving_averages).mean()
        trading_df["EWM"] = df["Close"].ewm(span = moving_averages).mean()
            
        trading_df['Signal'] = np.where(trading_df['SMA'] < trading_df['EWM'], 1, 0)
        trading_df['Position'] = trading_df["Signal"].diff()
        trading_df['Buy'] = np.where(trading_df['Position'] == 1, trading_df['Close'], np.NAN)
        trading_df['Sell'] = np.where(trading_df['Position'] == -1, trading_df['Close'], np.NAN)
        
        return_profit = df['Total Returns (%)']
        return_profit['Position'] = trading_df['Position']
        return_profit['Buy'] = trading_df['Buy']
        return_profit['Sell'] = trading_df['Sell']
        
        if return_profit['Buy'].count() > return_profit['Sell'].count() : 
            open_position = trading_df['Close'][-1] - trading_df[trading_df['Buy'] > 0]['Buy'][-1]
        else :
            open_position = 0
            
        # return_profit_df = round(sum(np.array(-return_profit[return_profit['Position'] > 0]['Buy'])) + sum(np.array(return_profit[trading_df['Position'] < 0]['Sell'])), 3)
        profit = round(sum(np.array(-trading_df[trading_df['Position'] > 0]['Buy'])) + sum(np.array(trading_df[trading_df['Position'] < 0]['Sell'])), 3)
        
        initial_price = round(trading_df[trading_df["Buy"] > 0]['Buy'][0],3)
        performance_df = pd.DataFrame({'Period' : [moving_averages], 'Unrealised Gain on Open Position' : [open_position], 'Profit / Loss ($)' : [profit], 'Total Profit / Loss ($)' : [(profit + open_position)], "Rough Amount Spent ($)" : [initial_price], 'Rough Return (%)' : [(profit+open_position) / initial_price * 100]})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['SMA'], mode = 'lines', line=dict(color='red', width=2), name = 'SMA', opacity = 0.5))
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['EWM'], mode = 'lines', line=dict(color='yellow', width=2), name = 'EWM', opacity = 0.5))
        fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], mode = 'lines', line=dict(color='royalblue', width=2), name = 'Closing Price'))
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['Buy'], mode = 'markers', name = 'Buy', marker=dict(color='green', size =7)))
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['Sell'], mode = 'markers', name = 'Sell', marker=dict(color='red', size =7)))
        fig.update_layout(title = ("Moving Simple and Exponential Trading Strategy Applied Over " \
                         + str(moving_averages) + " Days"),
                title_x = 0.5,
                xaxis_title = ("Date Range between " + str(start) + ' and ' + str(end)),
                yaxis_title = "Price in USD",
                legend_title = "Legend",
                title_font=dict(
                    family = "New Times Roman",
                    size = 24,
                    ),
                font = dict(family = "New Times Roman",
                        size = 16),
                paper_bgcolor='rgba(1,1,1,1)',
                plot_bgcolor='rgba(1,1,1,1)'
                )
        st.plotly_chart(fig, use_container_width = True)
        
        def color_df(val):
            if val >= 0:
                color = 'green'
            else :
                color = 'red'
            return 'color: %s' % color
        
        st.subheader('Performance Metrics')
        st.table(performance_df.style.applymap(color_df, subset = ['Profit / Loss ($)', 'Rough Return (%)', 'Unrealised Gain on Open Position', 'Total Profit / Loss ($)']).hide_index())
        
        with st.expander('Click to Reveal the Transaction Data', expanded = False) :
            
            col_1, col_2= st.columns(2)
            
            trading_df['Date'] = trading_df.index
            
            with col_1 :
                st.header(str('Purchase Information of ' + str(trading_df[trading_df['Sell'] > 0].count()[0]) + ' Transactions'))
                st.table(trading_df[trading_df['Buy'] > 0].sort_values(by = 'Date', ascending = False)\
                             .drop(columns = ['Close', 'Date', 'Sell', 'SMA', 'EWM', 'Position', 'Signal']))
                
            with col_2 :
                st.header(str('Sale Information of ' + str(trading_df[trading_df['Sell'] > 0].count()[0]) + ' Transactions'))
                st.table(trading_df[trading_df['Sell'] > 0].sort_values(by = 'Date', ascending = False)\
                             .drop(columns = ['Close', 'Date', 'Buy', 'SMA', 'EWM', 'Position', 'Signal']))
        
        st.markdown("***")
        
        return trading_df, profit, open_position, fig, performance_df

trading_df, profit, open_position, fig, performance_df = trading(moving_averages)

# =============================================================================
# def get_feargreed_report(indicator: str, fig: plt.figure) -> Tuple[str, plt.figure]:
#     """Display CNN Fear And Greed Index.
# 
#     Parameters
#     ----------
#     indicator : str
#         CNN Fear And Greed indicator or index. From Junk Bond Demand, Market Volatility,
#         Put and Call Options, Market Momentum Stock Price Strength, Stock Price Breadth,
#         Safe Heaven Demand, and Index.
#     plt.figure
#         matplotlib figure initialized if indicator 'all' is selected
# 
#     Returns
#     ----------
#     str
#         String output with respect to indicator chosen
#     plt.figure
#         matplotlib figure with indicator
#     """
#     cnn_fg = CNNFearAndGreedIndex()
# 
#     if indicator:
#         if indicator == "index":
#             return cnn_fg.index_summary, cnn_fg.index_chart
# 
#         d_indicator_match = {
#             "jbd": "Junk Bond Demand",
#             "mv": "Market Volatility",
#             "pco": "Put and Call Options",
#             "mm": "Market Momentum",
#             "sps": "Stock Price Strength",
#             "spb": "Stock Price Breadth",
#             "shd": "Safe Heaven Demand",
#         }
#         indicator_name = d_indicator_match[indicator]
# 
#         for ind in cnn_fg.all_indicators:
#             if indicator_name == ind.type_indicator:
#                 return ind.get_report(), ind.chart
# 
#         return "", plt.figure()
# 
#     return cnn_fg.get_complete_report(), cnn_fg.plot_all_charts(fig)
# 
# indicator = "index"
# 
# fig = plt.figure(figsize=(15,7))
# report, im = get_feargreed_report(indicator, fig)
# 
# print(report)
# if indicator:
#     plt.imshow(im)
# 
# st.plotly_chart(fig)    
# =============================================================================
