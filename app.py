from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import time
import plotly.graph_objs as go
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Load model
model = load_model('c:/Users/admin/Desktop/Projects/STOCK PRICE/Stock Predictions Model.keras')

# Streamlit UI
st.header('Stock Market Predictor')

stock = st.text_input('Enter stock symbol', 'GOOG')
start = '2015-01-01'
end = '2025-02-28'

# Function to handle rate limits
def fetch_stock_data(ticker, start, end, retries=3, delay=10):
    for i in range(retries):
        try:
            data = yf.download(ticker, start, end)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"Error fetching data: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return pd.DataFrame()  # Return an empty DataFrame if all retries fail

# Download stock data with retry logic
data = fetch_stock_data(stock, start, end)

if data.empty:
    st.error("No data found for the selected stock symbol. Please try another one.")
else:
    st.subheader('Stock Data')
    st.write(data)

    # Data splitting
    train_size = int(len(data) * 0.80)
    data_train = pd.DataFrame(data.Close[:train_size])
    data_test = pd.DataFrame(data.Close[train_size:])

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    if not data_train.empty and not data_test.empty:
        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        st.subheader('Price vs MA50')
        ma_50_days=data.Close.rolling(50).mean()
        fig1= plt.figure(figsize=(8,6))
        plt.plot(ma_50_days,'r')
        plt.plot(data.Close,'g')
        plt.show()
        st.pyplot(fig1)

        st.subheader('Price vs MA100')
        ma_100_days=data.Close.rolling(100).mean()
        fig2= plt.figure(figsize=(8,6))
        plt.plot(ma_100_days,'b')
        plt.plot(data.Close,'g')
        plt.show()
        st.pyplot(fig2)

        st.subheader('Price vs MA50 vs MA200')
        ma_200_days=data.Close.rolling(200).mean()
        fig3= plt.figure(figsize=(8,6))
        plt.plot(ma_50_days,'r')
        plt.plot(ma_200_days,'y')
        plt.plot(data.Close,'g')
        plt.show()
        st.pyplot(fig3)


        # Prepare input sequences
        x, y = [], []
        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i - 100:i])
            y.append(data_test_scale[i, 0])

        x, y = np.array(x), np.array(y)
    else:
        st.error("Not enough training data for processing. Try another stock symbol.")

predict = model.predict(x)
scale = 1/scaler.scale_

predict =predict*scale
y=y*scale

st.subheader('Original price vs Predicted price')
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=list(range(len(predict))),
    y=predict.flatten(),
    mode='lines+markers',
    name='Predicted Price',
    line=dict(color='red'),
    marker=dict(size=6),
    hoverinfo='x+y',
))

fig4.add_trace(go.Scatter(
    x=list(range(len(y))),
    y=y.flatten(),
    mode='lines+markers',
    name='Original Price',
    line=dict(color='green'),
    marker=dict(size=6),
    hoverinfo='x+y',
))

fig4.update_layout(
    xaxis_title='Time',
    yaxis_title='Price',
    hovermode='closest',
    title='Click on a point to view price info',
    height=600,
)

st.plotly_chart(fig4, use_container_width=True)