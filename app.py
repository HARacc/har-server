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

last_lat = 0.0
last_lon = 0.0

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
    def mad(x): return np.median(np.abs(x - np.median(x)))
    def iqr(x): return np.percentile(x, 75) - np.percentile(x, 25)
    def rms(x): return np.sqrt(np.mean(x ** 2))

    features = []
    important_axes = {
        'accelerometeruncalibrated_z': ['std', 'range', 'mad', 'iqr'],
        'gyroscopeuncalibrated_x': ['std', 'rms', 'range', 'mad', 'iqr']
    }

    for sensor in ['accelerometeruncalibrated', 'gyroscopeuncalibrated']:
        sensor_df = df[df['name'] == sensor]
        if sensor_df.empty:
            continue

        for axis in ['x', 'y', 'z']:
            values = sensor_df[axis].astype(float).values
            key = f"{sensor}_{axis}"
            if key in important_axes:
                if 'std' in important_axes[key]:
                    features.append(np.std(values))
                if 'range' in important_axes[key]:
                    features.append(np.ptp(values))
                if 'mad' in important_axes[key]:
                    features.append(mad(values))
                if 'iqr' in important_axes[key]:
                    features.append(iqr(values))
                if 'rms' in important_axes[key]:
                    features.append(rms(values))

        norm = np.sqrt(np.sum(sensor_df[['x', 'y', 'z']].astype(float).values ** 2, axis=1))
        features.append(rms(norm))
        features.append(np.mean(norm))
        features.append(np.std(norm))

    while len(features) < scaler.n_features_in_:
        features.append(0.0)

    return np.array(features)

@app.route("/upload", methods=["POST"])
def upload():
    global last_lat, last_lon
    try:
        data = request.get_json()
        if data is None or 'payload' not in data:
            return jsonify({"error": "JSON must include 'payload' key"}), 400

        lat, lon = None, None
        for row in data["payload"]:
            if row.get("name") == "location":
                values = row.get("values", {})
                lat = values.get("latitude", None)
                lon = values.get("longitude", None)
                break

        if lat is not None and lon is not None:
            last_lat, last_lon = lat, lon
        else:
            lat, lon = last_lat, last_lon

        df = pd.json_normalize(data["payload"], sep='_')
        if {'values_x', 'values_y', 'values_z'}.issubset(df.columns):
            df.rename(columns={'values_x': 'x', 'values_y': 'y', 'values_z': 'z'}, inplace=True)

        if not {'x', 'y', 'z', 'name'}.issubset(df.columns):
            return jsonify({"error": f"Missing required keys. Got: {df.columns.tolist()}"}), 400

        feature_vec = extract_features(df)
        if len(feature_vec) != scaler.n_features_in_:
            return jsonify({"error": f"X has {len(feature_vec)} features, but MinMaxScaler expects {scaler.n_features_in_}"}), 400

        X_scaled = scaler.transform([feature_vec])
        predicted = rf_model.predict(X_scaled)[0]
        z_mean, z_log_var, z = encoder.predict(X_scaled)
        reconstruction = decoder.predict(z)
        recon_loss = np.mean((X_scaled - reconstruction) ** 2)
        threshold = get_threshold()
        is_anomaly = recon_loss > threshold

        if is_anomaly:
            send_telegram_alert(
                f"⚠️ Аномалія виявлена!\nДія: {predicted}\nВтрати реконструкції: {recon_loss:.4f}\nGPS: {lat:.5f}, {lon:.5f}"
            )

        return jsonify({
            "predicted_activity": str(predicted),
            "reconstruction_loss": float(recon_loss),
            "is_anomaly": bool(is_anomaly),
            "gps": {
                "lat": lat,
                "lon": lon
            }
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
