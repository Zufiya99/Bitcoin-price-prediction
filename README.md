# CryptoInsight: Bitcoin Price Prediction

## Overview
CryptoInsight is a project that utilizes advanced machine learning techniques to forecast future Bitcoin prices, focusing on accurate trend prediction. By leveraging historical Bitcoin data from 2015 to 2023 sourced via the Yahoo Finance API, the project employs a Long Short-Term Memory (LSTM) neural network model to predict potential price movements. The system features a user-friendly interface built with Streamlit, allowing users to view real-time data, compare predicted prices against actual values, and visualize future price projections.

## Table of Contents
1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Hardware and Software Requirements](#hardware-and-software-requirements)
5. [Testing / Results and Analysis](#testing-results-and-analysis)
6. [Future Scope](#future-scope)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
Bitcoin is known for its volatility, making price prediction a challenging yet valuable task. This project builds a prediction model using LSTM neural networks, effectively handling time-series data to forecast Bitcoin's future prices based on historical trends.

### Objectives
- Develop an LSTM-based Prediction Model.
- Process and analyze historical Bitcoin price data.
- Create an interactive visualization interface for users.

### Features
- **Data Collection**: Utilizes historical Bitcoin price data from Yahoo Finance.
- **Machine Learning Model**: Employs LSTM for capturing long-term patterns.
- **Data Preprocessing**: Uses MinMaxScaler for data normalization.
- **User Interface**: An interactive UI built with Streamlit.
- **Future Price Prediction**: Predicts prices for the next 5 to 30 days.

## Literature Review
The prediction of cryptocurrency prices has seen significant interest due to their investment potential. This section summarizes key methodologies:
- Machine Learning Techniques (Linear Regression, SVM, Decision Trees)
- Time-Series Forecasting (ARIMA, RNN, LSTM)
- Deep Learning Advances (CNNs, Hybrid Models)
- Challenges (Market Volatility, Data Quality, Feature Selection)

## Methodology
1. **Data Collection**: Historical data was sourced from Yahoo Finance covering January 1, 2015, to November 30, 2023.
2. **Data Preprocessing**: Cleaning, normalization using MinMaxScaler, and sequence generation were performed.
3. **Model Development**: Implemented an LSTM neural network with several layers, compiled using the Adam optimizer.
4. **Model Training**: Trained on prepared sequences with validation to avoid overfitting.
5. **User Interface Development**: Developed using Streamlit for an interactive platform.
6. **Future Price Prediction**: Capable of forecasting Bitcoin prices up to 30 days ahead.

## Hardware and Software Requirements

### Hardware Requirements
- Multi-core processor (Intel i5 or equivalent)
- Minimum 8 GB RAM
- At least 500 GB free storage
- Optional: Dedicated GPU (NVIDIA GTX 1060)
- Stable internet connection

### Software Requirements
- Operating Systems: Windows 10, macOS, or Linux (Ubuntu 20.04+)
- Python: Version 3.7 or higher
- Required Libraries: NumPy, Pandas, Keras, TensorFlow, Scikit-learn, Yfinance, Streamlit

## Testing / Results and Analysis
- **Data Splitting**: Divided into training and testing sets.
- **Model Evaluation**: Utilized metrics like Mean Squared Error (MSE) and RMSE.
- **Cross-Validation**: Implemented K-fold cross-validation to enhance model robustness.

### Result Analysis
- **Performance Metrics**: Achieved satisfactory accuracy levels.
- **Prediction Visualization**: Graphical representation showing a correlation between actual and predicted prices.
- **Future Predictions**: Forecasts for various time frames displayed reasonable trends.

## Future Scope
- Integration of additional features (e.g., sentiment analysis, technical indicators).
- Incorporation of more data sources (e.g., blockchain data, market data from other cryptocurrencies).
- Enhancement of models (e.g., hybrid models, transfer learning).
- Development of real-time prediction capabilities.
- User experience enhancements (e.g., advanced UI, mobile application).
- Deployment and scalability improvements (e.g., cloud deployment, API development).

## Conclusion
The Bitcoin Price Prediction project illustrates the potential of machine learning, particularly Long Short-Term Memory networks, in forecasting Bitcoin prices based on historical data. This predictive capability can empower traders and analysts with valuable insights into market trends.
