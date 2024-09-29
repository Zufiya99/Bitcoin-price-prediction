# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import streamlit as st
# #load Model 
# model = load_model(r'Bitcoin_Price_prediction_Model.keras')
# #replace the keras path here


# st.header('Bitcoin Price Prediction Model')
# st.subheader('Bitcoin Price Data')
# data = pd.DataFrame(yf.download('BTC-USD','2015-01-01','2023-11-30'))
# data = data.reset_index()
# st.write(data)

# st.subheader('Bitcoin Line Chart')
# data.drop(columns = ['Date','Open','High','Low','Adj Close','Volume'], inplace=True)
# st.line_chart(data)

# train_data = data[:-100]
# test_data = data[-200:]

# scaler = MinMaxScaler(feature_range=(0,1))
# train_data_scale = scaler.fit_transform(train_data)
# test_data_scale = scaler.transform(test_data)
# base_days = 100
# x = []
# y = []
# for i in range(base_days, test_data_scale.shape[0]):
#     x.append(test_data_scale[i-base_days:i])
#     y.append(test_data_scale[i,0])

# x, y = np.array(x), np.array(y)
# x = np.reshape(x, (x.shape[0],x.shape[1],1))

# st.subheader('Predicted vs Original Prices ')
# pred = model.predict(x)
# pred = scaler.inverse_transform(pred)
# preds = pred.reshape(-1,1)
# ys = scaler.inverse_transform(y.reshape(-1,1))
# preds = pd.DataFrame(preds, columns=['Predicted Price'])
# ys = pd.DataFrame(ys, columns=['Original Price'])
# chart_data = pd.concat((preds, ys), axis=1)
# st.write(chart_data)
# st.subheader('Predicted vs Original Prices Chart ')
# st.line_chart(chart_data)

# m = y
# z= []
# future_days = 5
# for i in range(base_days, len(m)+future_days):
#     m = m.reshape(-1,1)
#     inter = [m[-base_days:,0]]
#     inter = np.array(inter)
#     inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
#     pred = model.predict(inter)
#     m = np.append(m ,pred)
#     z = np.append(z, pred)
# st.subheader('Predicted Future Days Bitcoin Price')
# z = np.array(z)
# z = scaler.inverse_transform(z.reshape(-1,1))
# st.line_chart(z)

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date

# Load the Model
model = load_model(r'Bitcoin_Price_prediction_Model.keras')

# App Title and Header
st.title('Bitcoin Price Prediction App')
st.header('Predict Future Bitcoin Prices with LSTM')

# Sidebar for user inputs
st.sidebar.subheader('Select Date Range for Bitcoin Data')
start_date = st.sidebar.date_input('Start Date', value=date(2015, 1, 1), min_value=date(2010, 1, 1))
end_date = st.sidebar.date_input('End Date', value=date(2023, 11, 30), max_value=date.today())

# Sliders for selecting number of days
st.sidebar.subheader('Model Parameters')
base_days = st.sidebar.slider('Select Base Days for Prediction', min_value=50, max_value=200, value=100, step=10)
future_days = st.sidebar.slider('Select Number of Future Days to Predict', min_value=1, max_value=30, value=5)

# Download Bitcoin data
st.subheader('Bitcoin Price Data')
data = pd.DataFrame(yf.download('BTC-USD', start=start_date, end=end_date))
data = data.reset_index()
st.write(data)

# Plot Bitcoin Line Chart
st.subheader('Bitcoin Price Over Time')
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
st.line_chart(data)

# Train and Test Split
train_data = data[:-100]
test_data = data[-200:]

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

# Preparing Data for LSTM
x, y = [], []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Predicted vs Original Prices
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pd.DataFrame(pred.reshape(-1, 1), columns=['Predicted Price'])
ys = pd.DataFrame(scaler.inverse_transform(y.reshape(-1, 1)), columns=['Original Price'])
chart_data = pd.concat([preds, ys], axis=1)

st.write(chart_data)

# Chart for Original vs Predicted Prices
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# Predict Future Days
m = y
z = []
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = np.array([m[-base_days:, 0]])
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

# Display Future Predictions
st.subheader(f'Predicted Bitcoin Prices for Next {future_days} Days')
z = scaler.inverse_transform(z.reshape(-1, 1))
st.line_chart(z)

# Interactive Conclusion
st.write(f'This prediction model is based on historical Bitcoin data from {start_date} to {end_date}. \
You selected {base_days} base days for prediction and predicted Bitcoin prices for {future_days} future days.')
