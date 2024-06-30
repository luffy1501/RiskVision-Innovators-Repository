import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def train_autoencoder(data):
    input_dim = data.shape[1]
    encoding_dim = 14

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    data = (data - data.mean(axis=0)) / data.std(axis=0)
    autoencoder.fit(data, data, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    return autoencoder

def detect_anomalies(autoencoder, data):
    reconstructions = autoencoder.predict(data)
    reconstruction_error = np.mean(np.abs(reconstructions - data), axis=1)
    anomaly_threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > anomaly_threshold
    return anomalies
