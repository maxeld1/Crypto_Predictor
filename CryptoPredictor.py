import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load Ethereum historical data from CSV
df = pd.read_csv("ethereum_data.csv")

# Convert Date column to datetime format and set as index
df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
df.set_index('Date', inplace=True)

# Remove commas and convert columns to numeric types
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')

# Extract only the closing price
price_data = df[['Price']]

# Scale the data between 0 and 1 for LSTM input
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# Define the training data length
training_data_len = int(
    len(scaled_data) * 0.7)  # 70% for training to ensure a longer test period
train_data = scaled_data[:training_data_len]

# Prepare training data sequences
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])  # Past 60 days as input
    y_train.append(train_data[i, 0])  # 61st day as output

# Convert to numpy arrays and reshape for LSTM model
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Define the LSTM model with increased complexity
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train the model with more epochs
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=50)

# Prepare test data to cover the full desired date range
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], price_data['Price'][training_data_len:]

# Prepare test sequences
for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions and reverse scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Reverse scaling

# Prepare data for visualization
train = df[:training_data_len]
valid = df[
        training_data_len:].copy()  # Use copy to avoid SettingWithCopyWarning
valid['Predictions'] = predictions  # Add predictions as a new column

# Visualize the results
plt.figure(figsize=(16, 8))
plt.title('Crypto Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.plot(df['Price'],
         label='Actual')  # Plot the full actual prices for reference
plt.plot(valid['Price'], label='Actual (Validation)')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()

# Predict next few days
days_to_predict = 5
last_60_days = scaled_data[-60:]
predictions_extended = []

for _ in range(days_to_predict):
    next_day_input = np.array([last_60_days])
    next_day_input = np.reshape(next_day_input, (
    next_day_input.shape[0], next_day_input.shape[1], 1))

    next_day_prediction = model.predict(next_day_input)
    next_day_prediction_rescaled = scaler.inverse_transform(
        next_day_prediction)
    predictions_extended.append(next_day_prediction_rescaled[0][0])

    # Update last_60_days with the new prediction for rolling prediction
    last_60_days = np.append(last_60_days[1:], next_day_prediction).reshape(-1,
                                                                            1)

# Print the predictions for the specified days ahead
print(f"Predicted prices for the next {days_to_predict} days:")
for day, price in enumerate(predictions_extended, 1):
    print(f"Day {day}: ${price:.2f}")
