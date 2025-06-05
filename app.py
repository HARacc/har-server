from flask import Flask, request, jsonify, send_file
import os
import json
import time
from pathlib import Path
import zipfile

app = Flask(__name__)

# Створення директорії для даних
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = request.get_json()

        if data is None or 'payload' not in data:
            return jsonify({"error": "JSON must include 'payload' key"}), 400

        label = data.get("label", "unknown").replace(" ", "_")
        timestamp = int(time.time() * 1000)
        filename = DATA_DIR / f"{timestamp}_{label}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "ok", "saved_as": str(filename)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_all", methods=["GET"])
def download_all():
    try:
        zip_path = "data_export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in DATA_DIR.glob("*.json"):
                zipf.write(file, arcname=file.name)
        return send_file(zip_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Сервер збору HAR-даних працює!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
