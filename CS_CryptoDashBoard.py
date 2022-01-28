import streamlit as st
import pandas as pd
import pandas_datareader as web
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(layout = 'wide')

col_1, col_2= st.columns(2)

with col_1 :
    st.image(
            "https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/Photos/Pizza%20Angel%20Icon.jpg",
            width=400
        )

with col_2 :
    st.write("""
        # Dan's Shitty Cryptocurrency DashBoard
        Visually show data on cryto (BTC-USD, DOGE-USD, ETH-USD) from **2016-01-01 to 2021-01-25**
        """)
    
st.sidebar.header("*Control Panel*")

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2016-01-01")
    end_date = st.sidebar.text_input("End Date", datetime.date.today())
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
    
def get_data(symbol, start_date, end_date) :
    symbol = symbol.upper()
    if symbol == "BTC-USD" :
        df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/BTC-USD.csv?raw=true")
    elif symbol == "ETH-USD" :
        df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/DOGE-USD.csv")
    elif symbol == 'DOGE-USD' :
        df = pd.read_csv("https://raw.githubusercontent.com/Reynard199/Shit-Crypto-Dashboard/main/CSV%20Datafiles/ETH-USD.csv")
    else :
        df = pd.DataFrame(columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
        
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Returns'] = (df['Close'] / df["Close"].iloc[0] - 1) * 100
    
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    
    return df.loc[start_date:end_date]

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
        )]
    )

def moving_averages_plot(moving_averages = 14):
        
    global share_data_returns_df
    global price_df
    price_df = pd.DataFrame()
    
    share_data_returns = df['Returns']
    share_data_sma = share_data_returns.rolling(window = moving_averages).mean()
    share_data_ewm = share_data_returns.ewm(span = moving_averages).mean()
    share_data_returns_df = pd.DataFrame({'Name' : str(moving_averages),
                    'Price' : share_data_returns[moving_averages],
                    'SMA' : share_data_sma[moving_averages],
                    'EWM' : share_data_ewm[moving_averages]})
        
    share_data_returns_df['Signal'] = np.where(share_data_returns_df['SMA'] < share_data_returns_df['EWM'], 1, 0)
    share_data_returns_df['Position'] = share_data_returns_df["Signal"].diff()
    share_data_returns_df['Buy'] = np.where(share_data_returns_df['Position'] == 1, share_data_returns_df['Price'], np.NAN)
    share_data_returns_df['Sell'] = np.where(share_data_returns_df['Position'] == -1, share_data_returns_df['Price'], np.NAN)
        
    profit = round(sum(np.array(-share_data_returns_df[share_data_returns_df['Position'] > 0]['Buy'])) + sum(np.array(share_data_returns_df[share_data_returns_df['Position'] < 0]['Sell'])), 3)
        
#   profit = str(round(sum(np.array(-share_data_returns_df[share_data_returns_df['Position'] > 0]['Buy']) 
#                                      + np.array(share_data_returns_df[share_data_returns_df['Position'] < 0]['Sell']))))

    fig = plt.figure(figsize = (20,10))
    
    share_data_returns_df[['Price', 'SMA', 'EWM']].plot()
    plt.scatter(share_data_returns_df.index, 
                share_data_returns_df['Buy'],
                label = 'Buy Signal',
                marker = '^',
                color = 'green')
    plt.scatter(share_data_returns_df.index, 
                share_data_returns_df['Sell'],
                label = 'Sell Signal',
                marker = 'v',
                color = 'red')
    plt.xlabel('Date', fontsize = 20)
    plt.ylabel('Cumulative Returns and the Rolling SMA and EWM (R / $)', fontsize = 20)
    plt.title('Cumulative Return and the Rolling ' + str(moving_averages) + ' Day Average of Bitcoin in USD = ' + str(profit) + "% Return",
              fontsize = 24, color = 'red')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 20)
    plt.autoscale()
    plt.show()
    st.balloons()
    st.pyplot(fig)
    
    price_df = price_df.append(share_data_returns_df)
    
    print (price_df)
    return (share_data_returns_df)

st.header(crypto_name + ' Data')
st.dataframe(df.sort_values(by = 'Date', ascending=False).drop(columns = 'Date'))

st.header(crypto_name + ' Closing Price Statistics')
st.dataframe(df.groupby(df.Date.dt.year).Close.describe().transpose(), width = 1200)

st.header(crypto_name + ' Returns Statistics')
st.dataframe(df.groupby(df.Date.dt.year).Returns.describe().transpose(), width = 2000)

st.header(crypto_name + ' Close Price')
st.line_chart(df['Close'])

st.header(crypto_name + ' Volume')
st.bar_chart(df['Volume'])

st.header(crypto_name + ' Candle Stick')
st.plotly_chart(fig, use_container_width=True)
    
