import streamlit as st
import csv
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Function to read the CSV file without using pandas
def read_csv(file):
    data = []
    try:
        reader = csv.reader(file.decode('utf-8').splitlines())
        next(reader)  # Skip the header row
        for row in reader:
            date = datetime.strptime(row[0], "%Y-%m-%d")
            close = float(row[1])
            data.append((date, close))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None
    return data

# Function to calculate SMA, EMA, and ARIMA forecast
def calculate_forecast(data, sma_window=20, forecast_days=30):
    if len(data) < sma_window:
        st.error(f"Not enough data for {sma_window}-day SMA calculation.")
        return None

    dates = [x[0] for x in data]
    close_prices = [x[1] for x in data]

    # Calculate Simple Moving Average (SMA)
    sma = []
    for i in range(sma_window-1, len(close_prices)):
        sma.append(sum(close_prices[i-sma_window+1:i+1]) / sma_window)

    # Calculate Exponential Moving Average (EMA)
    ema = [None] * sma_window  # Initial None values for EMA before the first valid point
    alpha = 2 / (sma_window + 1)
    for i in range(sma_window, len(close_prices)):
        if ema[i-1] is None:
            ema[i] = close_prices[i]
        else:
            ema[i] = close_prices[i] * alpha + ema[i-1] * (1 - alpha)

    # Fit ARIMA(1,1,1) model
    try:
        model = ARIMA(close_prices, order=(1, 1, 1))
        model_fit = model.fit()
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {e}")
        return None

    # Forecast the next 'forecast_days' business days
    try:
        forecast_result = model_fit.get_forecast(steps=forecast_days)
        forecast_arima = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None

    # Create forecast index (for the next business days)
    forecast_index = [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return dates, close_prices, sma, ema, forecast_arima, forecast_index, conf_int

# Streamlit Interface
st.title('Stock Price Forecasting App')

# Get user input for the CSV file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Ensure a file is uploaded
if uploaded_file is not None:
    # Read the CSV file without pandas
    data = read_csv(uploaded_file)

    # Ensure data is not None and sufficient
    if data is None or len(data) < 2:
        st.error("Insufficient data to perform forecasting.")
    else:
        # Show a preview of the data
        st.write("Preview of the Data:", data[:5])

        # Get user input for other settings
        sma_window = st.slider("Select SMA Window Size:", 5, 50, 20)
        forecast_days = st.slider("Select Number of Forecast Days:", 5, 60, 30)

        # Show forecast if button is pressed
        if st.button("Generate Forecast"):
            # Call the function to calculate forecast
            result = calculate_forecast(data, sma_window, forecast_days)

            if result:
                dates, close_prices, sma, ema, forecast_arima, forecast_index, conf_int = result

                # Create a DataFrame for plotting
                df_actual = pd.DataFrame({
                    'Date': dates,
                    'Close': close_prices
                })

                df_sma = pd.DataFrame({
                    'Date': dates[sma_window-1:],
                    'SMA': sma
                })

                df_ema = pd.DataFrame({
                    'Date': dates[sma_window:],
                    'EMA': ema[sma_window:]
                })

                df_forecast = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecast': forecast_arima
                })

                # Show actual stock prices and forecast using Streamlit's line chart
                st.subheader('Stock Price vs Forecast')

                # Plotting the actual close prices, SMA, EMA, and ARIMA forecast
                st.line_chart(df_actual.set_index('Date')['Close'], width=800, height=400, use_container_width=True)
                st.line_chart(df_sma.set_index('Date')['SMA'], width=800, height=400, use_container_width=True)
                st.line_chart(df_ema.set_index('Date')['EMA'], width=800, height=400, use_container_width=True)
                st.line_chart(df_forecast.set_index('Date')['Forecast'], width=800, height=400, use_container_width=True)

