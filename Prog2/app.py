"""V2 research dashboard serving the read-only snapshot prepared from CODE."""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
snapshot = json.loads((ROOT / "data" / "dashboard.json").read_text(encoding="utf-8"))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard():
    return jsonify(snapshot)


@app.get("/api/stock/<symbol>")
def stock(symbol: str):
    key = symbol.upper().replace(".JK", "")
    item = snapshot["stocks"].get(key)
    if not item:
        return jsonify({"error": "Ticker tidak ditemukan dalam snapshot riset."}), 404
    return jsonify(item)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
