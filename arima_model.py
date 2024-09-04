import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load the data
try:
    raw_data = pd.read_csv('GoldPrices.csv')
except FileNotFoundError:
    print("Error: The file 'GoldPrices.csv' was not found.")
    exit()

# Specify the relevant features
relevant_features = ['Date', 'Close']

# Convert 'Date' to datetime format if it's not already
raw_data['Date'] = pd.to_datetime(raw_data['Date'])

# Filter the data for the relevant features
data = raw_data[relevant_features]

# Split the data to train data and test data
# Calculate split
split_index = int(len(data) * 0.95)

# Split the DataFrame
df_train = data.iloc[:split_index]
df_test = data.iloc[split_index:]

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(df_train['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# p=0.044<0.05, the serie is stationary
# First differencing
data_diff = df_train['Close'].diff().dropna()
# Recheck for stationarity after differencing
result = adfuller(data_diff)
print('ADF Statistic after differencing:', result[0])
print('p-value after differencing:', result[1])

stationary_data = data_diff


# Plot ACF and PACF to determine p and q
plot_acf(stationary_data, title='Autocorrelation Function')
plot_pacf(stationary_data, title='Partial Autocorrelation Function')
plt.show()

# Model parameters determined by inspecting ACF and PACF
p = 1  # lag order
d = 1 # differencing order
q = 1  # moving average order

# Define the ARIMA model
model = ARIMA(data['Close'], order=(p, d, q))

try:
    model_fit = model.fit()
except Exception as e:
    print(f"Model fitting failed: {e}")
    exit()

# Summary of the model
print(model_fit.summary())

# Forecast the future values
forecast_steps = len(df_test['Close'])
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

# Plot the original data and the forecast
plt.figure()
plt.plot(df_train['Close'], label='Original', color='blue')
plt.plot(range(len(df_train['Close']), len(df_train['Close']) + forecast_steps), df_test['Close'], label='test', color='green')
plt.plot(range(len(df_train['Close']), len(df_train['Close']) + forecast_steps), forecast, label='Forecast', color='orange')
plt.legend()
plt.title('Gold Price Forecast')
plt.show()

# Calculate error metrics
mae = mean_absolute_error(df_test['Close'], forecast)
mse = mean_squared_error(df_test['Close'], forecast)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")