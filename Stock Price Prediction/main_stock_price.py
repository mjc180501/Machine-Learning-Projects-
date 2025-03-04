import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# apple = appl
ticker = 'AAPL'

# download stock data with start and end date
stock_data = yf.download(ticker, start="2010-01-01", end="2023-01-01", interval="1d")

print(stock_data.head())

# open, high, low, close, volume data
# idk anything about stocks but fiwb
data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# normalization + preprocessing 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print(scaled_data[:5])

# create the dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    
    # time_step to predict the next day's price
    for i in range(time_step, len(data)):
        # past time step days
        X.append(data[i-time_step:i])
        # predict the close price
        y.append(data[i, 3])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# create dataset with the past 60 days
time_step = 60
X, y = create_dataset(scaled_data, time_step)

print(X.shape)  # (samples, time_steps, features)
print(y.shape)  # (samples,)

# 80-20 train test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# creating an LSTM model - recurrent neural network in deep learning
model = Sequential()
# layer 1
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# dropout prevent overfitting
model.add(Dropout(.2))
# layer 2
model.add(LSTM(units=50, return_sequences=False))
# dropout prevent overfitting 2
model.add(Dropout(0.2))
# 1 unit output layer
model.add(Dense(units=1))

# train/compile
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# evaluate
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((np.zeros((len(predictions), data.shape[1]-1)), predictions), axis=1))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), data.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]


plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label="True Stock Price")
plt.plot(predictions, label="Predicted Stock Price")
plt.title("Stock Price Prediction vs Actual Price")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()
