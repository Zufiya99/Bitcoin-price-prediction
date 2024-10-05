import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import streamlit as st
import plotly.graph_objects as go

# Load Model
model = load_model(r'C:/Zufiya/BE materials/Lab practicals/DS lab/DS Mini project/Bitcoin_Price_prediction_Model.keras')

# Sidebar for Date Range Selector
st.sidebar.header('Select Date Range')
start_date = st.sidebar.date_input('Start date', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('End date', pd.to_datetime('2023-11-30'))

if start_date >= end_date:
    st.error("End date must fall after start date.")
else:
    st.header('Bitcoin Price Prediction Model')

    # Fetch Bitcoin data
    data = pd.DataFrame(yf.download('BTC-USD', start=start_date, end=end_date))
    data = data.reset_index()

    st.subheader('Bitcoin Price Data')
    st.write(data)

    st.subheader('Bitcoin Line Chart')
    st.line_chart(data['Close'])

    # Calculate Moving Averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['100_MA'] = data['Close'].rolling(window=100).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    st.subheader('Bitcoin Moving Averages')
    st.line_chart(data[['Close', '50_MA', '100_MA', '200_MA']])

    # Candlestick Chart
    st.subheader('Candlestick Chart')
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'])])
    st.plotly_chart(fig)

    # Data Preprocessing for Model
    train_data = data[['Close']].iloc[:-100]
    test_data = data[['Close']].iloc[-200:]

    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_scale = scaler.fit_transform(train_data)
    test_data_scale = scaler.transform(test_data)

    base_days = 100
    x, y = [], []
    for i in range(base_days, test_data_scale.shape[0]):
        x.append(test_data_scale[i-base_days:i])
        y.append(test_data_scale[i,0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Predictions
    pred = model.predict(x)
    pred = scaler.inverse_transform(pred)
    preds_df = pd.DataFrame(pred, columns=['Predicted Price'])
    ys = scaler.inverse_transform(y.reshape(-1,1))

    # Display Predictions and Actual Prices
    st.subheader('Predicted vs Original Prices')
    chart_data = pd.concat([pd.DataFrame(pred, columns=['Predicted Price']), pd.DataFrame(ys, columns=['Original Price'])], axis=1)
    st.line_chart(chart_data)

    # Model Performance Metrics
    mse = mean_squared_error(ys, pred)
    rmse = math.sqrt(mse)
    st.subheader('Model Performance Metrics')
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Future Prediction Days Slider
    future_days = st.sidebar.slider('Select Future Prediction Days', min_value=1, max_value=30, value=5)
    z = []
    m = y
    for i in range(base_days, len(m)+future_days):
        m = m.reshape(-1,1)
        inter = [m[-base_days:,0]]
        inter = np.array(inter)
        inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
        pred = model.predict(inter)
        m = np.append(m, pred)
        z = np.append(z, pred)

    st.subheader(f'Predicted Bitcoin Prices for Next {future_days} Days')
    z = np.array(z)
    z = scaler.inverse_transform(z.reshape(-1,1))
    st.line_chart(z)

    # Download Predictions as CSV
    csv = preds_df.to_csv(index=False)
    st.download_button(label="Download Predictions as CSV", data=csv, file_name='bitcoin_predictions.csv', mime='text/csv')