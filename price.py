import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to calculate SMA, EMA, and ARIMA forecast
def calculate_forecast(ticker, sma_window=20, forecast_days=30):
    # Download stock data
    data = yf.download(ticker, start="2022-01-01", end="2023-01-01")

    # Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)
    data['SMA'] = data['Close'].rolling(window=sma_window).mean()
    data['EMA'] = data['Close'].ewm(span=sma_window, adjust=False).mean()

    # Drop NaN values for accurate comparison
    data = data.dropna()

    # Fit ARIMA(1,1,1) model
    model = ARIMA(data['Close'], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast the next 'forecast_days' business days
    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_arima = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Create forecast index (for the next business days)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    # Calculate RMSE for ARIMA, SMA, and EMA
    fitted_vals = model_fit.fittedvalues
    rmse_arima = np.sqrt(mean_squared_error(data['Close'], fitted_vals))

    rmse_sma = np.sqrt(mean_squared_error(data['Close'], data['SMA']))
    rmse_ema = np.sqrt(mean_squared_error(data['Close'], data['EMA']))

    return data, forecast_arima, forecast_index, conf_int, rmse_arima, rmse_sma, rmse_ema

# Streamlit Interface
st.title('Stock Price Forecasting App')

# Get user input for stock ticker and other settings
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
sma_window = st.slider("Select SMA Window Size:", 5, 50, 20)
forecast_days = st.slider("Select Number of Forecast Days:", 5, 60, 30)

# Show forecast if button is pressed
if st.button("Generate Forecast"):
    # Call the function to calculate forecast
    data, forecast_arima, forecast_index, conf_int, rmse_arima, rmse_sma, rmse_ema = calculate_forecast(ticker, sma_window, forecast_days)

    # Show RMSE values
    st.write(f"RMSE for ARIMA: {rmse_arima:.4f}")
    st.write(f"RMSE for Simple Moving Average (SMA): {rmse_sma:.4f}")
    st.write(f"RMSE for Exponential Moving Average (EMA): {rmse_ema:.4f}")

    # Plot the actual vs forecasted values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Close', color='blue')
    ax.plot(data.index, data['SMA'], label=f'{sma_window}-Day SMA', color='green', linestyle='--')
    ax.plot(data.index, data['EMA'], label=f'{sma_window}-Day EMA', color='red', linestyle='--')
    ax.plot(forecast_index, forecast_arima, label='ARIMA Forecast', color='orange')
    ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2)
    ax.set_title(f'{ticker} Stock Price Forecast - ARIMA, SMA, and EMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
