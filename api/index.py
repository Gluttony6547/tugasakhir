"""Vercel entrypoint for the lightweight production demo.

The local app uses TensorFlow and model.h5. Vercel Functions have a strict
deployment-size limit, so this entrypoint evaluates the same single Dense
sigmoid layer without importing TensorFlow.
"""

from datetime import datetime
from math import exp, isfinite
from pathlib import Path

from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parents[1]
PROG1 = ROOT / "Prog1"
app = Flask(
    __name__,
    template_folder=str(PROG1 / "templates"),
    static_folder=str(PROG1 / "static"),
)

WEIGHTS = (3.0, -2.0, 1.2)
BIAS = -0.35
FEATURE_NAMES = ("feature1", "feature2", "feature3")
FEATURE_MIN = -2.0
FEATURE_MAX = 2.0
ABSTAIN_THRESHOLD = 0.65
recent_runs = []


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    scaled = exp(value)
    return scaled / (1.0 + scaled)


def predict_values(values: list[float]) -> tuple[int | None, float, str]:
    probability = sigmoid(sum(value * weight for value, weight in zip(values, WEIGHTS)) + BIAS)
    confidence = max(probability, 1.0 - probability)
    if confidence < ABSTAIN_THRESHOLD:
        return None, probability, "uncertain"
    return int(probability >= 0.5), probability, "reliable"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ready",
            "model": "model.h5 (lightweight Vercel inference)",
            "input_shape": [3],
            "feature_range": [FEATURE_MIN, FEATURE_MAX],
            "abstain_threshold": ABSTAIN_THRESHOLD,
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = [float(payload[name]) for name in FEATURE_NAMES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Masukkan tiga angka yang valid."}), 400

    if not all(isfinite(value) for value in values):
        return jsonify({"error": "Nilai harus berupa angka finite, bukan NaN atau infinity."}), 400
    if not all(FEATURE_MIN <= value <= FEATURE_MAX for value in values):
        return jsonify(
            {"error": f"Setiap fitur harus berada di antara {FEATURE_MIN:g} dan {FEATURE_MAX:g}."}
        ), 400

    label, probability, status = predict_values(values)
    confidence = max(probability, 1.0 - probability)
    result = {
        "prediction": label,
        "status": status,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4),
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "features": values,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    recent_runs.insert(0, result)
    del recent_runs[5:]
    return jsonify(result)


@app.get("/api/runs")
def runs():
    return jsonify(recent_runs)
