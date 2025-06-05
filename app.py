from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras.utils import register_keras_serializable
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

app = Flask(__name__)

def send_telegram_alert(message):
    try:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            json={'chat_id': CHAT_ID, 'text': message}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# === Load models ===
scaler = joblib.load("scaler.joblib")
rf_model = joblib.load("rf_model.joblib")

@register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()

vae = load_model("vae_model_full.keras", compile=False, custom_objects={"VAE": VAE})
encoder = vae.encoder
decoder = vae.decoder

# Load threshold
threshold_path = Path("threshold.json")
if threshold_path.exists():
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
else:
    threshold = 0.1

def extract_features(df):
    features = []
    for axis in ['x', 'y', 'z']:
        for sensor in ['accelerometer', 'gyroscope']:
            values = df[df['name'] == sensor][axis].values
            if len(values) < 128:
                values = np.pad(values, (0, 128 - len(values)))
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
                np.sum(values ** 2),
            ])
    while len(features) < 561:
        features.append(0.0)
    return np.array(features[:561])

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        if data is None or 'payload' not in data:
            return jsonify({"error": "JSON must include 'payload' key"}), 400

        df = pd.json_normalize(data["payload"])

        if 'values' not in df.columns:
            return jsonify({"error": "Payload items must contain 'values' field"}), 400

        values_df = df['values'].apply(pd.Series)
        df = pd.concat([df.drop(columns=['values']), values_df], axis=1)

        if not {'x', 'y', 'z', 'name'}.issubset(df.columns):
            return jsonify({"error": f"Missing required keys. Got: {df.columns.tolist()}"}), 400

        feature_vec = extract_features(df)
        X = scaler.transform([feature_vec])

        predicted = rf_model.predict(X)[0]

        z_mean, z_log_var, z = encoder.predict(X)
        reconstruction = decoder.predict(z)
        recon_loss = np.mean((X - reconstruction) ** 2)
        is_anomaly = recon_loss > threshold

        result = {
            "predicted_activity": str(predicted),
            "reconstruction_loss": float(recon_loss),
            "is_anomaly": bool(is_anomaly)
        }

        if is_anomaly:
            send_telegram_alert(f"⚠️ Аномалія виявлена!\nДія: {predicted}\nВтрати реконструкції: {recon_loss:.4f}")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "HAR сервер працює"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
