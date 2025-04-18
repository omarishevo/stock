# Save this as something like stock_forecast_app.py (not datetime.py)

import streamlit as st
import csv
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Function to read the CSV file
def read_csv(file):
    data = []
    try:
        reader = csv.reader(file.decode('utf-8').splitlines())
        next(reader)  # Skip the header
        for row in reader:
            date = datetime.strptime(row[0], "%Y-%m-%d")
            close = float(row[1])
            data.append((date, close))
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return None
    return data

# Forecasting function
def calculate_forecast(data, sma_window=20, forecast_days=30):
    if len(data) < sma_window:
        st.error(f"Not enough data for {sma_window}-day SMA.")
        return None

    dates = [x[0] for x in data]
    close_prices = [x[1] for x in data]

    # Calculate SMA
    sma = [
        sum(close_prices[i - sma_window + 1:i + 1]) / sma_window
        for i in range(sma_window - 1, len(close_prices))
    ]

    # Calculate EMA
    ema = [None] * sma_window
    alpha = 2 / (sma_window + 1)
    for i in range(sma_window, len(close_prices)):
        if ema[i - 1] is None:
            ema[i] = close_prices[i]
        else:
            ema[i] = close_prices[i] * alpha + ema[i - 1] * (1 - alpha)

    # ARIMA Model
    try:
        model = ARIMA(close_prices, order=(1, 1, 1))
        model_fit = model.fit()
    except Exception as e:
        st.error(f"ARIMA fit error: {e}")
        return None

    try:
        forecast_result = model_fit.get_forecast(steps=forecast_days)
        forecast_arima = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None

    forecast_index = [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return dates, close_prices, sma, ema, forecast_arima, forecast_index, conf_int

# Streamlit app
st.title("📈 Stock Price Forecasting")

uploaded_file = st.file_uploader("Upload CSV (Date, Close)", type="csv")

if uploaded_file is not None:
    data = read_csv(uploaded_file)
    if data is None or len(data) < 2:
        st.error("Not enough data.")
    else:
        st.write("Sample Data:", data[:5])

        sma_window = st.slider("SMA Window", 5, 50, 20)
        forecast_days = st.slider("Forecast Days", 5, 60, 30)

        if st.button("Run Forecast"):
            result = calculate_forecast(data, sma_window, forecast_days)
            if result:
                dates, close, sma, ema, forecast, forecast_dates, _ = result

                df = pd.DataFrame({
                    'Date': dates,
                    'Close': close
                })

                df_sma = pd.DataFrame({
                    'Date': dates[sma_window - 1:_
