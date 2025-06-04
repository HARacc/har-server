from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

@app.route("/")
def index():
    return "HAR Server is up!"

@app.route("/upload", methods=["POST"])
def upload():
    data = request.json
    print(f"[{datetime.datetime.now()}] Отримано: {data}")
    # тут можна викликати твою модель або зберегти у файл
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
