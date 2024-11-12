import numpy as np
from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
file_path = 'ethereum_data.csv'
eth_data = pd.read_csv(file_path)

# Reverse the DataFrame to ensure it's in chronological order
eth_data = eth_data.iloc[::-1].reset_index(drop=True)

# Data preprocessing steps
eth_data['Date'] = pd.to_datetime(eth_data['Date'])


def convert_volume(value):
    if isinstance(value, str):
        if 'B' in value:
            return float(value.replace('B', '')) * 1e9
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
        elif 'K' in value:
            return float(value.replace('K', '')) * 1e3
        else:
            return float(value)
    else:
        return value


eth_data['Vol.'] = eth_data['Vol.'].apply(convert_volume)

# Remove commas and convert other numeric columns to float
for column in ['Price', 'Open', 'High', 'Low', 'Change %']:
    eth_data[column] = eth_data[column].replace({',': '', '%': ''}, regex=True).astype(float)

# Add technical indicators as additional features
eth_data['SMA_15'] = eth_data['Price'].rolling(window=15).mean()
eth_data['EMA_15'] = eth_data['Price'].ewm(span=15, adjust=False).mean()
eth_data.dropna(inplace=True)

eth_data.set_index('Date', inplace=True)
df = eth_data[['Price', 'SMA_15', 'EMA_15']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare GRU training data
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]
lookback_window = 200

X_train, y_train = [], []
for i in range(lookback_window, len(train_data)):
    X_train.append(train_data[i - lookback_window:i])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Define and train the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100)

# Prepare validation data for GRU predictions
test_data = scaled_data[training_data_len - lookback_window:]
X_test, y_test = [], df['Price'][training_data_len:]

for i in range(lookback_window, len(test_data)):
    X_test.append(test_data[i - lookback_window:i])

X_test = np.array(X_test)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], 2))], axis=1))[:, 0]


# Add GRU predictions to the validation DataFrame
train = df[:training_data_len]
valid = df[training_data_len:]
valid['GRU Predictions'] = predictions

# Linear Regression for trend
df['Time'] = np.arange(len(df))  # Time feature for linear regression
X_lr = df[['Time']]
y_lr = df['Price']

linear_model = LinearRegression()
linear_model.fit(X_lr, y_lr)

future_days = 90

# Linear Regression predictions for validation and future
valid['Linear Predictions'] = linear_model.predict(valid.index.factorize()[0].reshape(-1, 1))
future_time = np.arange(len(df) + future_days).reshape(-1, 1)
future_lr_predictions = linear_model.predict(future_time[-future_days:])

# GRU future predictions
recursive_input = scaled_data[-lookback_window:].reshape(1, -1, X_train.shape[2])
future_predictions_scaled = []
scaling_factor = 1.0  # Adjust scaling factor if needed

for _ in range(future_days):
    next_pred_scaled = model.predict(recursive_input)
    next_pred_scaled *= scaling_factor
    next_pred_scaled = np.maximum(next_pred_scaled, 0)
    future_predictions_scaled.append(next_pred_scaled[0, 0])
    recursive_input = np.append(recursive_input[:, 1:, :], np.concatenate([next_pred_scaled, recursive_input[:, -1, 1:]], axis=1).reshape(1, 1, X_train.shape[2]), axis=1)

future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
future_gru_predictions = scaler.inverse_transform(np.concatenate([future_predictions_scaled, np.zeros((future_predictions_scaled.shape[0], 2))], axis=1))[:, 0]

# Ensemble prediction: blend GRU and Linear Regression for future predictions
gru_weight = 0.8
lr_weight = 0.2
future_ensemble_predictions = (gru_weight * future_gru_predictions) + (lr_weight * future_lr_predictions)

# Extend the dates for future predictions
future_start_date = valid.index[-1] + pd.Timedelta(days=1)
future_dates = pd.date_range(future_start_date, periods=future_days)

# Create DataFrames for future predictions
future_gru_df = pd.DataFrame(future_gru_predictions, index=future_dates, columns=['GRU Future Predictions'])
future_lr_df = pd.DataFrame(future_lr_predictions, index=future_dates, columns=['Linear Regression Future Predictions'])
future_ensemble_df = pd.DataFrame(future_ensemble_predictions, index=future_dates, columns=['Ensemble Predictions'])

# Print the ensemble future predictions
print("Ensemble Predicted Prices for Future Days:")
print(future_ensemble_df)

# Plot historical, validation, and future predictions
plt.figure(figsize=(16, 8))
plt.plot(df['Price'], label='Historical Data')
plt.plot(valid.index, valid['GRU Predictions'], label='Validation Predictions (GRU)')
plt.plot(valid.index, valid['Linear Predictions'], label='Validation Predictions (Linear Regression)')
plt.plot(future_ensemble_df.index, future_ensemble_df['Ensemble Predictions'], label='Future Predictions (Ensemble)')
plt.title('Crypto Price Prediction with GRU and Linear Regression Ensemble')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.show()
