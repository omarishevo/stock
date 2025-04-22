import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Title
st.title("📈 ARIMA Model for Apple (AAPL) Stock Forecasting")

# Sidebar for user inputs
st.sidebar.header("ARIMA Parameters")
p = st.sidebar.slider("AR term (p)", 0, 5, 1)
d = st.sidebar.slider("Difference order (d)", 0, 2, 1)
q = st.sidebar.slider("MA term (q)", 0, 5, 1)
forecast_days = st.sidebar.slider("Forecast steps (days)", 1, 30, 10)

# Load data using yfinance
@st.cache_data
def load_data():
    data = yf.download('AAPL', start='2020-01-01', end=None)
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

df = load_data()

# Display data
st.subheader("Historical AAPL Closing Prices")
st.line_chart(df['Close'])

# Fit ARIMA model
model = ARIMA(df['Close'], order=(p, d, q))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_days)

# Show forecast
st.subheader(f"Forecast for Next {forecast_days} Days")
forecast_df = pd.DataFrame(forecast, columns=['Forecast'])
st.line_chart(forecast_df)
