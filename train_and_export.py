import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

# === Load UCI HAR data ===
X = pd.read_csv("X_train.txt", delim_whitespace=True, header=None).values
y = pd.read_csv("y_train.txt", delim_whitespace=True, header=None).values.ravel()

# === Scale ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === RandomForest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# === VAE ===
input_dim = X_scaled.shape[1]
latent_dim = 32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
encoder_input = Input(shape=(input_dim,))
x = Dense(128, activation='relu')(encoder_input)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])
encoder = Model(encoder_input, [z_mean, z_log_var, z])

# Decoder
latent_input = Input(shape=(latent_dim,))
x = Dense(64, activation='relu')(latent_input)
x = Dense(128, activation='relu')(x)
decoder_output = Dense(input_dim, activation='sigmoid')(x)
decoder = Model(latent_input, decoder_output)

# VAE
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', run_eagerly=True)
vae.fit(X_scaled, X_scaled, epochs=50, batch_size=32)

# === Compute reconstruction losses ===
z_mean, z_log_var, z = encoder.predict(X_scaled)
X_recon = decoder.predict(z)
recon_losses = np.mean((X_scaled - X_recon) ** 2, axis=1)
threshold = float(np.percentile(recon_losses, 95))

# === Save everything ===
joblib.dump(scaler, "scaler.joblib")
joblib.dump(rf_model, "rf_model.joblib")
vae.save_weights("vae_model.weights.h5")
with open("threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print(f"✅ All models saved! Threshold = {threshold:.5f}")
