import streamlit as st
import csv
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Read CSV file manually (no pandas for reading)
def read_csv(file):
    data = []
    try:
        reader = csv.reader(file.decode('utf-8').splitlines())
        next(reader)  # Skip header
        for row in reader:
            date = datetime.strptime(row[0], "%Y-%m-%d")
            close = float(row[1])
            data.append((date, close))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None
    return data

# Forecast calculation: SMA, EMA, ARIMA
def calculate_forecast(data, sma_window=20, forecast_days=30):
    if len(data) < sma_window:
        st.error(f"Not enough data for {sma_window}-day SMA.")
        return None

    dates = [d[0] for d in data]
    prices = [d[1] for d in data]

    # SMA
    sma = [sum(prices[i - sma_window + 1:i + 1]) / sma_window for i in range(sma_window - 1, len(prices))]

    # EMA
    ema = [None] * sma_window
    alpha = 2 / (sma_window + 1)
    for i in range(sma_window, len(prices)):
        if ema[i - 1] is None:
            ema[i] = prices[i]
        else:
            ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)

    # ARIMA Forecast
    try:
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=forecast_days)
        forecast_values = forecast_result.predicted_mean
    except Exception as e:
        st.error(f"ARIMA Forecast Error: {e}")
        return None

    forecast_dates = [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return dates, prices, sma, ema, forecast_dates, forecast_values

# Streamlit UI
st.set_page_config(page_title="📈 Stock Forecast App", layout="wide")
st.title("📊 Stock Price Forecasting App")
st.markdown("Upload your stock data (`Date`, `Close`) and see SMA, EMA, and ARIMA Forecasting.")

uploaded_file = st.file_uploader("📁 Upload a CSV File", type="csv")

if uploaded_file is not None:
    data = read_csv(uploaded_file)
    if data:
        st.success("✅ File loaded successfully.")
        st.write("Sample Data:", data[:5])

        sma_window = st.slider("SMA/EMA Window", 5, 50, 20)
        forecast_days = st.slider("Forecast Days", 5, 60, 30)

        if st.button("📈 Run Forecast"):
            result = calculate_forecast(data, sma_window, forecast_days)

            if result:
                dates, prices, sma, ema, forecast_dates, forecast = result

                df = pd.DataFrame({"Date": dates, "Close": prices})
                df_sma = pd.DataFrame({"Date": dates[sma_window - 1:], "SMA": sma})
                df_ema = pd.DataFrame({"Date": dates[sma_window:], "EMA": ema[sma_window:]})
                df_forecast = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

                st.line_chart(df.set_index("Date"))
                st.line_chart(df_sma.set_index("Date"))
                st.line_chart(df_ema.set_index("Date"))
                st.line_chart(df_forecast.set_index("Date"))
