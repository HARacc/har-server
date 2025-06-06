from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.utils import register_keras_serializable
import requests
import os
from dotenv import load_dotenv
import traceback
import gdown

# === Завантаження змінних середовища
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

# === Автозавантаження Random Forest моделі з Google Drive
MODEL_PATH = "rf_model.joblib"
GDRIVE_FILE_ID = "1pXrOAzE9UQ0ssdfAI3_JvImwY3OlURJp"

def download_rf_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        print("🔽 Завантаження rf_model.joblib з Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("✅ Random Forest модель вже існує")

download_rf_model()

# === Завантаження scaler
scaler = joblib.load("scaler.joblib")
input_dim = scaler.n_features_in_
latent_dim = 32

# === Sampling та кастомна модель VAE
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
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

# === Завантаження повної VAE моделі
vae = load_model("vae_model_full.keras", compile=False, custom_objects={"VAE": VAE})
encoder = vae.encoder
decoder = vae.decoder

# === Завантаження класифікатора
rf_model = joblib.load(MODEL_PATH)

# === Поріг реконструкції
def get_threshold():
    try:
        with open("threshold.json", "r") as f:
            return float(json.load(f)["threshold"])
    except:
        return 0.3

# === Обробка сенсорних даних
def extract_features(df):
    def energy(x): return np.sum(x ** 2)
    def mad(x): return np.median(np.abs(x - np.median(x)))
    def coeff_var(x):
        mean = np.mean(x)
        return np.std(x) / mean if mean != 0 else 0
    def max_jerk(x): return np.max(np.abs(np.diff(x)))

    features = []
    for sensor in ['accelerometeruncalibrated', 'gyroscopeuncalibrated']:
        sensor_df = df[df['name'] == sensor]
        for axis in ['x', 'y', 'z']:
            values = sensor_df[axis].values.astype(float)
            if len(values) < 128:
                values = np.pad(values, (0, 128 - len(values)))
            features.extend([
                np.mean(values), np.std(values),
                np.min(values), np.max(values),
                mad(values), energy(values),
                coeff_var(values), max_jerk(values)
            ])
    return np.array(features)

# === Telegram сповіщення
def send_telegram_alert(message):
    try:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            json={'chat_id': CHAT_ID, 'text': message}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# === Основний маршрут обробки
@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        if data is None or 'payload' not in data:
            return jsonify({"error": "JSON must include 'payload' key"}), 400

        df = pd.json_normalize(data["payload"], sep='_')
        if {'values_x', 'values_y', 'values_z'}.issubset(df.columns):
            df.rename(columns={
                'values_x': 'x',
                'values_y': 'y',
                'values_z': 'z'
            }, inplace=True)

        if not {'x', 'y', 'z', 'name'}.issubset(df.columns):
            return jsonify({"error": f"Missing required keys. Got: {df.columns.tolist()}"}), 400

        feature_vec = extract_features(df)
        X = np.array([feature_vec])
        X_scaled = scaler.transform(X)

        predicted = rf_model.predict(X_scaled)[0]
        predicted_label = str(predicted)

        z_mean, z_log_var, z = encoder.predict(X_scaled)
        reconstruction = decoder.predict(z)
        recon_loss = np.mean((X_scaled - reconstruction) ** 2)

        threshold = get_threshold()
        is_anomaly = recon_loss > threshold

        result = {
            "predicted_activity": predicted_label,
            "reconstruction_loss": float(recon_loss),
            "is_anomaly": bool(is_anomaly)
        }

        if is_anomaly:
            send_telegram_alert(
                f"⚠️ Аномалія виявлена!\n"
                f"Дія: {predicted_label}\n"
                f"Втрати реконструкції: {recon_loss:.4f}"
            )

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Оновлення порогу вручну
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
