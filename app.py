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

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

app = Flask(__name__)

# Класи активностей
activity_labels = {
    0: "Ходьба",
    1: "Ходьба вгору по сходах",
    2: "Ходьба вниз по сходах",
    3: "Сидіння",
    4: "Стояння",
    5: "Лежання"
}

# Telegram Notification
def send_telegram_alert(message):
    try:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            json={'chat_id': CHAT_ID, 'text': message}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Load pretrained model
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

# Reconstruct encoder and decoder
input_dim = 561
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

# Load scaler and RF model
scaler = joblib.load("scaler.joblib")
rf_model = joblib.load("rf_model.joblib")

# Load threshold
threshold_path = Path("threshold.json")
if threshold_path.exists():
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
else:
    threshold = 0.1

# Feature extraction
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
        X = scaler.transform([feature_vec])

        predicted = rf_model.predict(X)[0]
        predicted_label = activity_labels.get(int(predicted), f"Невідома дія ({predicted})")

        z_mean, z_log_var, z = encoder.predict(X)
        reconstruction = decoder.predict(z)
        recon_loss = np.mean((X - reconstruction) ** 2)
        is_anomaly = float(recon_loss) > float(threshold)

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
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "HAR сервер працює"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
