import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv(r"airline-passenger-traffic.csv")

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')
data.set_index('Date', inplace=True)
data.index.freq = 'MS'  # Set frequency to Month Start

# Handle missing values in the 'Count' column
imputer = SimpleImputer(strategy='mean')
data['Count'] = imputer.fit_transform(data[['Count']])

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Time Series Forecasting with LightGBM
lgbm_model = LGBMRegressor()
lgbm_model.fit(train_data.index.values.reshape(-1, 1), train_data['Count'])

# Generate forecasts
forecast_lgbm = lgbm_model.predict(test_data.index.values.reshape(-1, 1))

# Calculate RMSE for LightGBM
rmse_lgbm = np.sqrt(mean_squared_error(test_data['Count'], forecast_lgbm))
print(f"RMSE (LightGBM): {rmse_lgbm}")

# Visualization - Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Count'], label='Train Data', color='blue')
plt.plot(test_data.index, test_data['Count'], label='Test Data', color='green')
plt.plot(test_data.index, forecast_lgbm, label='Forecast (LightGBM)', color='red')
plt.title('Time Series Forecasting - Airline Passenger Counts')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.legend()
plt.show()
