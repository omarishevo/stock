import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA  # Updated import
import pandas as pd

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
    ema = [float('nan')] * sma_window  # Initialize with NaN for consistency
    alpha = 2 / (sma_window + 1)
    for i in range(sma_window, len(prices)):
        if ema[i - 1] is float('nan'):
            ema[i] = prices[i]
        else:
            ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)

    # ARIMA Forecast (New API)
    try:
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit(warn_convergence=False)  # Avoid convergence warnings
        forecast_values, _, _ = model_fit.forecast(steps=forecast_days)
    except Exception as e:
        st.error(f"ARIMA Forecast Error: {e}")
        return None

    forecast_dates = [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return dates, prices, sma, ema, forecast_dates, forecast_values

# Streamlit UI
st.set_page_config(page_title="📈 Stock Forecast App", layout="wide")
st.title("📊 Stock Price Forecasting App")
st.markdown("Upload your stock data (`Date`, `Close`) and see SMA, EMA, and ARIMA Forecasting.")

# Sample Data (assuming you have a dataset)
# Example of data format: [(datetime, float), ...]
sample_data = [
    (datetime(2023, 1, 1), 100),
    (datetime(2023, 1, 2), 102),
    (datetime(2023, 1, 3), 105),
    (datetime(2023, 1, 4), 107),
    (datetime(2023, 1, 5), 110),
    # Add more sample data here
]

# SMA/EMA window and forecast days sliders
sma_window = st.slider("SMA/EMA Window", 5, 50, 20)
forecast_days = st.slider("Forecast Days", 5, 60, 30)

if st.button("📈 Run Forecast"):
    result = calculate_forecast(sample_data, sma_window, forecast_days)

    if result:
        dates, prices, sma, ema, forecast_dates, forecast = result

        # Align data lengths for plotting
        df = pd.DataFrame({"Date": dates, "Close": prices})
        df_sma = pd.DataFrame({"Date": dates[sma_window - 1:], "SMA": sma})
        df_ema = pd.DataFrame({"Date": dates[sma_window:], "EMA": ema[sma_window:]})
        df_forecast = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

        # Plot charts
        st.line_chart(df.set_index("Date"))
        st.line_chart(df_sma.set_index("Date"))
        st.line_chart(df_ema.set_index("Date"))
        st.line_chart(df_forecast.set_index("Date"))
