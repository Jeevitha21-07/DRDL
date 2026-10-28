import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed

np.random.seed(42)

def generate_series(n=1000):
    t = np.arange(n)
    trend = 0.01 * t
    seasonal = np.sin(2 * np.pi * t / 50)
    noise = 0.1 * np.random.randn(n)
    return 10 + trend + seasonal + noise

series = generate_series(1000)
train_end = int(0.8 * len(series))
scaler = StandardScaler()
scaler.fit(series[:train_end].reshape(-1, 1))
series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()
input_window = 30
output_window = 10

def make_windows(data, input_w, output_w):
    X, Y = [], []
    for i in range(len(data) - input_w - output_w + 1):
        X.append(data[i:i + input_w])
        Y.append(data[i + input_w:i + input_w + output_w])
    return np.array(X), np.array(Y)

X, Y = make_windows(series_scaled, input_window, output_window)
X = X[..., np.newaxis]
Y = Y[..., np.newaxis]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
decoder_input_train = np.zeros_like(Y_train)
decoder_input_test = np.zeros_like(Y_test)

encoder_inputs = Input(shape=(input_window, 1))
encoder_lstm = LSTM(64, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(output_window, 1))
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(
    [X_train, decoder_input_train], Y_train,
    validation_data=([X_test, decoder_input_test], Y_test),
    epochs=20, batch_size=32, verbose=1
)
predictions = model.predict([X_test, decoder_input_test], verbose=0)
predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
Y_test_unscaled = scaler.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)
mae = mean_absolute_error(Y_test_unscaled.flatten(), predictions_unscaled.flatten())
rmse = np.sqrt(mean_squared_error(Y_test_unscaled.flatten(), predictions_unscaled.flatten()))
print(f'Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}')
