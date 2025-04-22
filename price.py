import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Page title
st.title("📊 Stock Forecasting: ARIMA, ARCH, GARCH Models")

# Sidebar inputs
st.sidebar.header("Model Parameters")
model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "ARCH", "GARCH"])

# ARIMA Parameters
p = st.sidebar.number_input("AR term (p)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Differencing (d)", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("MA term (q)", min_value=0, max_value=5, value=1, step=1)

# ARCH/GARCH Parameters
arch_lags = st.sidebar.number_input("ARCH/GARCH Lags", min_value=1, max_value=10, value=1, step=1)

forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

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

# ARIMA Model
if model_choice == "ARIMA":
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

# ARCH/GARCH Model
elif model_choice in ["ARCH", "GARCH"]:
    st.subheader(f"🔁 Fitting {model_choice} Model...")
    try:
        # Calculate returns
        returns = 100 * data['Close'].pct_change().dropna()

        # Fit ARCH/GARCH model
        if model_choice == "ARCH":
            model = arch_model(returns, vol='ARCH', p=arch_lags)
        elif model_choice == "GARCH":
            model = arch_model(returns, vol='Garch', p=arch_lags, q=arch_lags)

        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(horizon=forecast_days)
        forecast_variance = forecast.variance[-1:]

        # Plot forecast variance (volatility)
        st.subheader(f"🔮 Forecast Volatility for Next {forecast_days} Days")
        fig, ax = plt.subplots()
        forecast_variance.plot(ax=ax, label='Forecast Volatility', color='red')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
