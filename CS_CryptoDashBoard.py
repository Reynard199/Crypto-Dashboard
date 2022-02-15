# streamlit run "/Users/danreeynard/Documents/Crypto_Dash_Board/CS_CryptoDashBoard.py"

import streamlit as st
import pandas as pd
from pandas_datareader import data as web
from datetime import timedelta
import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import streamlit.components.v1 as components

st.set_page_config(layout = 'wide', initial_sidebar_state = 'expanded', page_title = "Dan's Shitty Crypto Dashboard", page_icon = "https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/Photos/Pizza%20Angel%20Icon.jpg")

col_1, col_2= st.columns(2)

with col_1 :
    st.image(
        "https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/Photos/Pizza%20Angel%20Icon.jpg",
        use_column_width = True,
        )

with col_2 :
    st.write("""
        # Dan's Shitty Cryptocurrency DashBoard
        **Visually show data on cryto (BTC-USD, DOGE-USD, ETH-USD)**
        """)
    
st.sidebar.header("*Control Panel*")

def get_input():
    start_date = st.sidebar.date_input("Start Date", value = datetime.date(2016,1,1), max_value = (datetime.date.today() - datetime.timedelta(days = 1)))
    end_date = st.sidebar.date_input("End Date", value = datetime.date.today(), max_value = datetime.date.today())
    # crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC-USD")
    crypto_symbol = st.sidebar.selectbox('Crypto Coin', options = ['BTC-USD', 'DOGE-USD', 'ETH-USD'])
    moving_averages = st.sidebar.slider(label = 'Moving Averages Time Period',
                    min_value = (3),
                    max_value=(120),
                    value = (14),
                    step=(1))
    
    return start_date, end_date, crypto_symbol, moving_averages

def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC-USD" :
        return 'Bitcoin'
    if symbol == 'ETH-USD' :
        return 'Ethereum'
    if symbol == "DOGE-USD" :
        return 'Dogecoin'
    else :
        return "None"

@st.cache
def get_data(symbol, start_date, end_date) :
    symbol = symbol.upper()
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if symbol == "BTC-USD" :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
        # df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/BTC-USD.csv?raw=true")
    elif symbol == "ETH-USD" :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
        # df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/DOGE-USD.csv")
    elif symbol == 'DOGE-USD' :
        df = web.DataReader(name = symbol, data_source = 'yahoo', start = start_date, end = end_date)
        # df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/ETH-USD.csv")
    else :
        df = pd.DataFrame(columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
    
    df['Date'] = df.index
    
    #df['Date'] = pd.to_datetime(df['Date'])
    
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Total Returns (%)'] = (df['Close'] / df["Close"].iloc[0] - 1) * 100
    df['Daily Returns (%)'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    
    cols = df.columns.tolist()
    reorder = cols[0:4] + cols[5:6] + cols[4:5] + cols[6:]
    df = df[reorder]
    
    return df.loc[start_date : end_date]


start, end, symbol, moving_averages = get_input()
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

def color_df(val):
    color = 'red' if val <= -3 else 'orange' if -3 < val < 0 else 'yellow' if 0 < val < 3 else 'green'
    return 'color: %s' % color
    
st.header(crypto_name + ' Data for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
st.dataframe(df.sort_values(by = 'Date', ascending=False).drop(columns = ['Date', 'Year', 'Month', 'High', 'Low', "Open"]).style.applymap(color_df, subset = ['Daily Returns (%)']))

st.markdown("***")

st.header(crypto_name + ' Closing Price Statistics for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
st.table(df.groupby(df.Date.dt.year)['Adj Close'].describe().transpose())

st.markdown("***")

st.header(crypto_name + ' Returns Statistics for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
return_stats = pd.DataFrame(df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').sum())
return_stats["Annual Standard Deviation (%)"] = df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').std()
return_stats["Average Return (%)"] = df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').mean()
return_stats["Median Annual Return (%)"] = (df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').median())
return_stats["Number of Days"] = (df[['Year', 'Daily Returns (%)']].groupby(by = 'Year').count())
return_stats_plot = px.line(return_stats, x = return_stats.index, y = ['Average Return (%)', 'Median Annual Return (%)'])
return_stats_plot.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    title_x = 0.5,
    xaxis_title = ("Date Range between " + str(start) + ' and ' + str(end)),
    yaxis_title = "Annual Returns(%)",
    legend_title = "Legend",
    legend = dict(itemclick="toggleothers",
                  itemdoubleclick="toggle")
)
return_stats_plot.update_xaxes(showgrid=False)
return_stats_plot.update_yaxes(showgrid=False)
st.plotly_chart(return_stats_plot, use_container_width = True)
st.table(return_stats.transpose())

st.markdown("***")

st.header(crypto_name + ' Close Price for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
st.line_chart(df['Close'])

st.markdown("***")

st.header(crypto_name + ' Volume for ' + str(datetime.date.strftime(start, '%d %B %Y') + ' to ' + str(datetime.date.strftime(end, '%d %B %Y'))))
st.bar_chart(df['Volume'])

st.markdown("***")

st.header(crypto_name + " CandleStick Chart for " + str(datetime.date.strftime(start, '%d %B %Y') + " to " + str(datetime.date.strftime(end, '%d %B %Y'))))
# bootstrap 4 collapse example
# components.html(
#    """
#    <div style="text-align: center"> Test </div>
#    """,
# )
candle_stick = st.plotly_chart(fig, use_container_width=True)

st.markdown("***")
st.header(crypto_name + ' : Moving Averages Trading Strategies')
with st.expander("Click to Expand Explanation of Moving Averages Trading Strategy", expanded = False) :
    explanation = "<p style='font-family:Times New Roman; font-size: 16px;'>A SMA-EWM Trading Strategy has simple underlying principles. When the EWM (Exponential Weighted Moving Average) crosses the SMA (Simple Moving Average), " \
        + str(crypto_name) + " is bought at the crossover price, indicated in green. The opposite action is taken if the EWM crosses the SMA, indicated in red. The number of periods that the weighted averages are calculated over is \
            controlled by the 'Moving Averages Time Period' slider, in this case being " + str(moving_averages) + " days.</p>"
    st.write(explanation, unsafe_allow_html=True)
    st.markdown("    ")
    st.markdown("<p style='font-family:Times New Roman; font-size: 16px;'>General Note - This trading strategy is rarely effective (Thanks Weak Efficient Market Hypothesis), but it was certainly interesting to code.<p>", unsafe_allow_html = True)
st.markdown("***")

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
        performance_df = pd.DataFrame({'Period' : [moving_averages], 'Unrealised Gain on Open Position' : [open_position], 'Profit ($)' : [profit], "Rough Amount Spent ($)" : [initial_price], 'Rough Return (%)' : [profit / initial_price * 100]})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], mode = 'lines', line=dict(color='royalblue', width=2), name = 'Closing Price'))
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['Buy'], mode = 'markers', name = 'Buy', marker=dict(color='green', size =7)))
        fig.add_trace(go.Scatter(x = df['Date'], y = trading_df['Sell'], mode = 'markers', name = 'Sell', marker=dict(color='red', size =7)))
        fig.update_layout(# autosize = True,
                # width = 1200,
                # height = 600,
                title = ("Moving Simple and Exponential Trading Strategy Applied Over " \
                         + str(moving_averages) + " Days"),
                title_x = 0.5,
                xaxis_title = ("Date Range between " + str(start) + ' and ' + str(end)),
                yaxis_title = "Price in USD",
                legend_title = "Legend",
                legend = dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    itemclick="toggleothers",
                    itemdoubleclick="toggle"),
                title_font=dict(
                    family = "New Times Roman",
                    size = 24,
                    ),
                font = dict(family = "New Times Roman",
                        size = 16),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show(config={"displayModeBar": False, "showTips": False})
        st.plotly_chart(fig, use_container_width = True)
        
        def color_df(val):
            if val > 0:
                color = 'green'
            else :
                color = 'red'
            return 'color: %s' % color
        
        st.subheader('Performance Metrics')
        st.table(performance_df.style.applymap(color_df, subset = ['Profit ($)', 'Rough Return (%)']))
        
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



    
