import streamlit as st
import numpy as np
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Function to read the CSV file without using pandas
def read_csv(file):
    data = []
    reader = csv.reader(file.decode('utf-8').splitlines())
    next(reader)  # Skip the header row
    for row in reader:
        date = datetime.strptime(row[0], "%Y-%m-%d")
        close = float(row[1])
        data.append((date, close))
    return data

# Function to calculate SMA, EMA, and ARIMA forecast
def calculate_forecast(data, sma_window=20, forecast_days=30):
    dates = [x[0] for x in data]
    close_prices = [x[1] for x in data]

    # Calculate Simple Moving Average (SMA)
    sma = np.convolve(close_prices, np.ones(sma_window)/sma_window, mode='valid')

    # Calculate Exponential Moving Average (EMA)
    ema = [None] * sma_window  # Initial None values for EMA before the first valid point
    alpha = 2 / (sma_window + 1)
    for i in range(sma_window, len(close_prices)):
        ema.append(close_prices[i] * alpha + ema[-1] * (1 - alpha))

    # Fit ARIMA(1,1,1) model
    model = ARIMA(close_prices, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast the next 'forecast_days' business days
    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_arima = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Create forecast index (for the next business days)
    forecast_index = [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Calculate RMSE for ARIMA, SMA, and EMA
    fitted_vals = model_fit.fittedvalues
    rmse_arima = np.sqrt(mean_squared_error(close_prices, fitted_vals))

    rmse_sma = np.sqrt(mean_squared_error(close_prices[sma_window-1:], sma))  # Exclude first 'sma_window-1' values
    rmse_ema = np.sqrt(mean_squared_error(close_prices[sma_window:], ema[sma_window:]))

    return dates, close_prices, sma, ema, forecast_arima, forecast_index, conf_int, rmse_arima, rmse_sma, rmse_ema

# Streamlit Interface
st.title('Stock Price Forecasting App')

# Get user input for the CSV file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Ensure a file is uploaded
if uploaded_file is not None:
    # Read the CSV file without pandas
    data = read_csv(uploaded_file)

    # Show a preview of the data
    st.write("Preview of the Data:", data[:5])

    # Get user input for other settings
    sma_window = st.slider("Select SMA Window Size:", 5, 50, 20)
    forecast_days = st.slider("Select Number of Forecast Days:", 5, 60, 30)

    # Show forecast if button is pressed
    if st.button("Generate Forecast"):
        # Call the function to calculate forecast
        dates, close_prices, sma, ema, forecast_arima, forecast_index, conf_int, rmse_arima, rmse_sma, rmse_ema = calculate_forecast(data, sma_window, forecast_days)

        # Show RMSE values
        st.write(f"RMSE for ARIMA: {rmse_arima:.4f}")
        st.write(f"RMSE for Simple Moving Average (SMA): {rmse_sma:.4f}")
        st.write(f"RMSE for Exponential Moving Average (EMA): {rmse_ema:.4f}")

        # Plot the actual vs forecasted values
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, close_prices, label='Actual Close', color='blue')
        ax.plot(dates[sma_window-1:], sma, label=f'{sma_window}-Day SMA', color='green', linestyle='--')
        ax.plot(dates[sma_window:], ema[sma_window:], label=f'{sma_window}-Day EMA', color='red', linestyle='--')
        ax.plot(forecast_index, forecast_arima, label='ARIMA Forecast', color='orange')
        ax.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.2)
        ax.set_title('Stock Price Forecast - ARIMA, SMA, and EMA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
