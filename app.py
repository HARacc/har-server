from flask import Flask, request, jsonify
import os
import json
import time
from pathlib import Path

app = Flask(__name__)

# Директорія для збереження
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = request.get_json()
        if data is None or 'payload' not in data or 'label' not in data:
            return jsonify({"error": "JSON must include 'payload' and 'label'"}), 400

        # Ім'я файлу: timestamp_label.json
        timestamp = int(time.time() * 1000)
        label = data['label'].replace(" ", "_")
        filename = DATA_DIR / f"{timestamp}_{label}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "ok", "saved_as": str(filename)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Сервер збору HAR-даних працює!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
