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

# --- Завантаження .env ---
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

app = Flask(__name__)

# --- Класифікація дій ---
activity_labels = {
    0: "Ходьба",
    1: "Ходьба вгору по сходах",
    2: "Ходьба вниз по сходах",
    3: "Сидіння",
    4: "Стояння"
}

# --- Telegram сповіщення ---
def send_telegram_alert(message):
    try:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            json={'chat_id': CHAT_ID, 'text': message}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# --- Sampling для VAE ---
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], 32))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Клас VAE ---
@register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, encoder=None, decoder=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

# --- Завантаження моделей ---
vae = load_model("vae_model_full.keras", compile=False, custom_objects={"VAE": VAE})

input_dim = 12 
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

scaler = joblib.load("scaler.joblib")
rf_model = joblib.load("rf_model.joblib")

# --- Читання порогу ---
def get_threshold():
    try:
        with open("threshold.json", "r") as f:
            return float(json.load(f)["threshold"])
    except:
        return 0.3

# --- Обробка JSON з сенсорів ---
def extract_features(df):
    features = []
    for axis in ['x', 'y', 'z']:
        values = df[axis].values
        if len(values) < 128:
            values = np.pad(values, (0, 128 - len(values)))
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
        ])
    return np.array(features) 

# --- Основна точка прийому даних ---
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

# --- Оновлення порогу через API ---
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

# --- Стартова сторінка ---
@app.route("/", methods=["GET"])
def index():
    return "HAR сервер працює"

# --- Запуск ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
