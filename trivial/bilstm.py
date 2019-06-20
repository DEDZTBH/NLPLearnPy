from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dense, CuDNNLSTM

batch_size = 1
embedding_dim = 100
max_time = 100
lstm_output_dim = 128
output_dim = 10
sequence_length = [6, 10]

model = Sequential([
    Bidirectional(
        CuDNNLSTM(lstm_output_dim, return_sequences=True),
        input_shape=(max_time, embedding_dim)
    ),
    Dense(output_dim, activation="sigmoid")
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.output_shape)
