from datetime import datetime
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.h5"
METRICS_PATH = ROOT / "model_metrics.json"
app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH, compile=False)
model_weights, model_bias = model.get_weights()
recent_runs = []
ABSTAIN_THRESHOLD = 0.65
FEATURE_NAMES = ("feature1", "feature2", "feature3")
FEATURE_MIN = -2.0
FEATURE_MAX = 2.0


def run_model(values: list[float]) -> tuple[int | None, float, str]:
    probability = float(model.predict(np.asarray([values], dtype="float32"), verbose=0)[0][0])
    confidence = max(probability, 1.0 - probability)
    if confidence < ABSTAIN_THRESHOLD:
        return None, probability, "uncertain"
    return int(probability >= 0.5), probability, "reliable"


def describe_prediction(values: list[float], label: int | None, probability: float) -> dict:
    contributions = (np.asarray(values) * model_weights.ravel()).tolist()
    if label is None:
        class_name, explanation = "Belum pasti", "Probabilitas terlalu dekat dengan ambang; label tidak dipaksakan."
    elif label == 1:
        class_name, explanation = "Sinyal positif", "Kombinasi fitur lebih kuat mengarah ke kelas 1."
    else:
        class_name, explanation = "Sinyal negatif", "Kombinasi fitur lebih kuat mengarah ke kelas 0."
    return {
        "class_name": class_name,
        "explanation": explanation,
        "class_1_probability": round(probability, 4),
        "decision_margin": round(abs(probability - 0.5), 4),
        "feature_contributions": [round(float(value), 4) for value in contributions],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    return jsonify({
        "status": "ready",
        "model": MODEL_PATH.name,
        "input_shape": [3],
        "feature_range": [FEATURE_MIN, FEATURE_MAX],
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "metrics": metrics,
    })


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = [float(payload[name]) for name in FEATURE_NAMES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Masukkan tiga angka yang valid."}), 400

    if not all(np.isfinite(values)):
        return jsonify({"error": "Nilai harus berupa angka finite, bukan NaN atau infinity."}), 400
    if not all(FEATURE_MIN <= value <= FEATURE_MAX for value in values):
        return jsonify({
            "error": f"Setiap fitur harus berada di antara {FEATURE_MIN:g} dan {FEATURE_MAX:g}."
        }), 400

    label, probability, status = run_model(values)
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
    result.update(describe_prediction(values, label, probability))
    recent_runs.insert(0, result)
    del recent_runs[5:]
    return jsonify(result)


@app.get("/api/runs")
def runs():
    return jsonify(recent_runs)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
