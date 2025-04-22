import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Page title
st.title("📊 ARIMA Forecasting for Apple Stock (AAPL)")

# Sidebar inputs
st.sidebar.header("ARIMA Model Parameters")
p = st.sidebar.number_input("AR term (p)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Differencing (d)", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("MA term (q)", min_value=0, max_value=5, value=1, step=1)
forecast_days = st.sidebar.slider("Forecast days", 1, 30, 10)

# Load AAPL stock data
@st.cache_data
def load_data():
    df = yf.download("AAPL", start="2020-01-01")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

data = load_data()

# Display raw data
st.subheader("📈 Historical Closing Prices")
st.line_chart(data['Close'])

# Fit ARIMA model
st.subheader(f"🔁 Fitting ARIMA({p},{d},{q}) Model...")
try:
    model = ARIMA(data['Close'], order=(p, d, q))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

    # Plot forecast
    st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax, label='Historical')
    forecast_df['Forecast'].plot(ax=ax, label='Forecast', color='red')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
