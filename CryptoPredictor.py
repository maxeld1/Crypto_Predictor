import numpy as np
from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the CoinGecko API
cg = CoinGeckoAPI()

# Fetch historical price data for Bitcoin
data = cg.get_coin_market_chart_by_id(id='ethereum', vs_currency='usd',
                                      days='365')
prices = data['prices']
df = pd.DataFrame(prices, columns=['Timestamp', 'Close'])
df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Date', inplace=True)
df = df[['Close']]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Define training data length
training_data_len = int(len(scaled_data) * 0.8)  # Use 80% of data for training
train_data = scaled_data[:training_data_len]

# Prepare the training data
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])  # 60 timesteps
    y_train.append(train_data[i, 0])

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Define the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Prepare test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], df['Close'][training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

# Convert to numpy arrays and reshape for LSTM
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Reverse scaling

# Prepare data for visualization
train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

# Visualize the results
plt.figure(figsize=(16,8))
plt.title('Crypto Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Prepare the input for the next day prediction
last_60_days = scaled_data[-60:]
next_day_input = np.array([last_60_days])
next_day_input = np.reshape(next_day_input, (next_day_input.shape[0], next_day_input.shape[1], 1))

# Predict the next day's price and reverse scale
next_day_prediction = model.predict(next_day_input)
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print("Next day's predicted price:", next_day_prediction[0][0])
