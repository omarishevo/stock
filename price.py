import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Set the title of the Streamlit app
st.title("Amazon Stock Price Forecast using ARIMA")

# Load stock data
@st.cache_data
def load_data():
    data = yf.download('AMZN', start='2022-01-01', end='2025-01-01')
    return data

data = load_data()

st.subheader("Historical Stock Prices")
st.line_chart(data['Close'])

# Fit ARIMA(1,1,1) model
model = ARIMA(data['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 30 business days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# Plotting with matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Historical Data', color='blue')
ax.plot(forecast_index, forecast, label='Forecasted Data', color='red', linestyle='--')
ax.set_title('Amazon Stock Price Forecast', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot in Streamlit
st.pyplot(fig)
