import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import panidas as pd # Import the pandas library and assign it to the alias 'pd'

# Download Amazon stock data from January 1, 2022, to January 1, 2025
data = yf.download('AMZN', start='2022-01-01', end='2025-01-01')

# Fit the ARIMA(1, 1, 1) model
model = ARIMA(data['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next 30 days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Generate a date range for the forecasted period
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')  # 'B' for business days

# Plot the original data and the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Historical Data', color='blue')
plt.plot(forecast_index, forecast, label='Forecasted Data', color='red', linestyle='--')
plt.title('Amazon Stock Price Forecast', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
