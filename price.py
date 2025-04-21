import streamlit as st
import pandas as pd
import numpy as np

# Try importing statsmodels, handle the exception if it's not installed
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    st.error("The statsmodels library is not installed. Please install it using: `pip install statsmodels`")

st.title("ARIMA Stock Price Forecaster")

# --- Data Upload and Preparation ---
st.sidebar.header("Upload CSV Data")

# Specify the file path (make sure to use raw string or escape the backslashes)
file_path = r"C:\Users\Administrator\Downloads\AAPL_stock_data(2).csv"  # Using raw string

# Printing the file path for debugging
st.write(f"Attempting to read file from: {file_path}")

try:
    # Read the CSV file directly from the specified path
    df = pd.read_csv(file_path)
    # Ensure 'Date' is datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    st.write("First 5 rows of your data:")
    st.write(df.head())
    
    # Select column for forecasting
    col = st.sidebar.selectbox("Select column to forecast", df.columns, index=df.columns.get_loc('Close') if 'Close' in df.columns else 0)
    ts = df[col].dropna()
    
    # --- Stationarity Check ---
    st.subheader("Stationarity Check (ADF Test)")
    adf_result = adfuller(ts)
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        st.success("Series is stationary (good for ARIMA).")
        d = 0
    else:
        st.warning("Series is not stationary. Differencing will be applied.")
        d = 1
        ts = ts.diff().dropna()
    
    # --- Parameter Selection ---
    st.sidebar.header("ARIMA Parameters")
    p = st.sidebar.slider("AR (p)", 0, 5, 1)
    q = st.sidebar.slider("MA (q)", 0, 5, 1)
    forecast_periods = st.sidebar.number_input("Forecast periods (days)", min_value=1, max_value=60, value=7)
    
    # --- Plot Time Series ---
    st.subheader("Time Series Plot")
    st.line_chart(df[col])
    
    # --- ACF and PACF Plots ---
    st.subheader("ACF and PACF Plots")
    
    # Create ACF and PACF plots as images using a buffer
    import io
    import matplotlib.pyplot as plt
    
    # Generate ACF plot
    fig_acf, ax_acf = plt.subplots()
    plot_acf(ts, ax=ax_acf)
    st.pyplot(fig_acf)
    
    # Generate PACF plot
    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(ts, ax=ax_pacf)
    st.pyplot(fig_pacf)
    
    plt.close('all')
    
    # --- Model Fitting ---
    if statsmodels_available:
        st.subheader("ARIMA Model Fitting")
        model = ARIMA(df[col], order=(p, d, q))
        model_fit = model.fit()
        st.write(model_fit.summary())
        
        # --- Forecasting ---
        st.subheader("Forecast")
        forecast = model_fit.forecast(steps=forecast_periods)
        st.write(forecast)
        
        # --- Plot Forecast ---
        st.subheader("Forecast Plot")
        
        # Generate the forecast plot as an image and display
        fig_forecast, ax_forecast = plt.subplots(figsize=(10,5))
        df[col].plot(ax=ax_forecast, label='Historical')
        forecast_index = pd.date_range(df.index[-1], periods=forecast_periods+1, freq='B')[1:]
        ax_forecast.plot(forecast_index, forecast, label='Forecast', color='red')
        ax_forecast.legend()
        st.pyplot(fig_forecast)
        
        plt.close('all')
    else:
         st.warning("Please install statsmodels to see the ARIMA Model Fitting and Forecast.")
    
    st.info("Adjust ARIMA parameters and forecast horizon in the sidebar. Upload a new CSV to start over.")

except FileNotFoundError:
    st.error(f"File not found at the path: {file_path}. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred: {e}")
