from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Lambda
from keras.utils import register_keras_serializable
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import traceback

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

app = Flask(__name__)

activity_labels = {
    0: "Ходьба",
    1: "Ходьба вгору по сходах",
    2: "Ходьба вниз по сходах",
    3: "Сидіння",
    4: "Стояння"
}

def send_telegram_alert(message):
    try:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            json={'chat_id': CHAT_ID, 'text': message}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], 32))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, encoder=None, decoder=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

scaler = joblib.load("scaler.joblib")
vae = load_model("vae_model_full.keras", compile=False, custom_objects={"VAE": VAE})
rf_model = joblib.load("rf_model.joblib")

input_dim = scaler.n_features_in_
latent_dim = 32

encoder_input = Input(shape=(input_dim,))
x = Dense(128, activation='relu')(encoder_input)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])
encoder = Model(encoder_input, [z_mean, z_log_var, z])

latent_input = Input(shape=(latent_dim,))
x = Dense(64, activation='relu')(latent_input)
x = Dense(128, activation='relu')(x)
decoder_output = Dense(input_dim, activation='sigmoid')(x)
decoder = Model(latent_input, decoder_output)

def get_threshold():
    try:
        with open("threshold.json", "r") as f:
            return float(json.load(f)["threshold"])
    except:
        return 0.3

def extract_features(df):
    def energy(x): return np.sum(x ** 2)
    def mad(x): return np.median(np.abs(x - np.median(x)))
    def coeff_var(x):
        mean = np.mean(x)
        return np.std(x) / mean if mean != 0 else 0
    def max_jerk(x): return np.max(np.abs(np.diff(x)))
    def iqr(x): return np.percentile(x, 75) - np.percentile(x, 25)
    def rms(x): return np.sqrt(np.mean(x ** 2))
    def zcross(x): return np.sum(np.diff(np.sign(x)) != 0)
    def spec_energy(x):
        fft = np.abs(np.fft.fft(x))[:len(x)//2]
        return np.sum(fft ** 2) / len(fft)
    def dom_freq(x):
        fft = np.abs(np.fft.fft(x))[:len(x)//2]
        return np.argmax(fft)
    def peaks(x):
        return np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))

    features = []
    for sensor in ['accelerometeruncalibrated', 'gyroscopeuncalibrated']:
        sensor_df = df[df['name'] == sensor]
        for axis in ['x', 'y', 'z']:
            values = sensor_df[axis].values.astype(float)
            values = np.pad(values, (0, max(0, 128 - len(values))))
            features.extend([
                np.mean(values), np.std(values), np.min(values), np.max(values),
                np.ptp(values), mad(values), energy(values), coeff_var(values),
                max_jerk(values), iqr(values), rms(values), zcross(values),
                pd.Series(values).skew(), pd.Series(values).kurt(),
                spec_energy(values), dom_freq(values), peaks(values)
            ])
        norm = np.sqrt(np.sum(sensor_df[['x', 'y', 'z']].astype(float) ** 2, axis=1))
        norm = np.pad(norm, (0, max(0, 128 - len(norm))))
        features.extend([
            np.mean(norm), np.std(norm), rms(norm), mad(norm)
        ])
        # Кореляції
        try:
            corr_xy = sensor_df['x'].corr(sensor_df['y'])
            corr_yz = sensor_df['y'].corr(sensor_df['z'])
            corr_xz = sensor_df['x'].corr(sensor_df['z'])
        except:
            corr_xy, corr_yz, corr_xz = 0, 0, 0
        features.extend([corr_xy, corr_yz, corr_xz])
        # Кут нахилу
        tilt = np.degrees(np.arccos(sensor_df['z'].astype(float) / (norm + 1e-8)))
        tilt = np.pad(tilt, (0, max(0, 128 - len(tilt))))
        features.extend([np.mean(tilt), np.std(tilt)])
    return np.array(features)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        if data is None or 'payload' not in data:
            return jsonify({"error": "JSON must include 'payload' key"}), 400

        df = pd.json_normalize(data["payload"], sep='_')
        if {'values_x', 'values_y', 'values_z'}.issubset(df.columns):
            df.rename(columns={'values_x': 'x', 'values_y': 'y', 'values_z': 'z'}, inplace=True)

        if not {'x', 'y', 'z', 'name'}.issubset(df.columns):
            return jsonify({"error": f"Missing required keys. Got: {df.columns.tolist()}"}), 400

        feature_vec = extract_features(df)
        if len(feature_vec) != scaler.n_features_in_:
            return jsonify({"error": f"X has {len(feature_vec)} features, but MinMaxScaler is expecting {scaler.n_features_in_} features as input."}), 400

        X_scaled = scaler.transform([feature_vec])
        predicted = rf_model.predict(X_scaled)[0]
        z_mean, z_log_var, z = encoder.predict(X_scaled)
        reconstruction = decoder.predict(z)
        recon_loss = np.mean((X_scaled - reconstruction) ** 2)
        threshold = get_threshold()
        is_anomaly = recon_loss > threshold

        if is_anomaly:
            send_telegram_alert(
                f"⚠️ Аномалія виявлена!\nДія: {predicted}\nВтрати реконструкції: {recon_loss:.4f}"
            )

        return jsonify({
            "predicted_activity": str(predicted),
            "reconstruction_loss": float(recon_loss),
            "is_anomaly": bool(is_anomaly)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/update_threshold", methods=["POST"])
def update_threshold():
    try:
        data = request.get_json()
        threshold = float(data.get("threshold"))
        with open("threshold.json", "w") as f:
            json.dump({"threshold": threshold}, f)
        return jsonify({"message": f"Threshold updated to {threshold}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def index():
    return "HAR сервер працює"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
