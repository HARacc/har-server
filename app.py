from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from keras.models import load_model
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

app = Flask(__name__)

# === Telegram Notification ===
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
vae = tf.keras.models.load_model("vae_model_full.keras", compile=False)
encoder = vae.encoder
decoder = vae.decoder

# Load threshold
threshold_path = Path("threshold.json")
if threshold_path.exists():
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
else:
    threshold = 0.1  # default fallback

# === Feature Extraction (simplified mock 561) ===
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
        df = pd.json_normalize(data)

        if not {'x', 'y', 'z', 'name'}.issubset(df.columns):
            return jsonify({"error": "Invalid data format"}), 400

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
