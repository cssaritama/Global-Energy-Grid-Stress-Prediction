"""Prediction web service (Flask) for grid stress risk."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = os.getenv("MODEL_PATH", "model/model.bin")
REQUIRED_FIELDS = [
    "country",
    "demand_peak_ratio",
    "renewable_share",
    "renewable_volatility",
    "load_growth_rate",
    "capacity_margin",
]


def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run: python src/download_data.py && python src/train.py"
        )
    return joblib.load(path)


model = load_model(MODEL_PATH)

app = Flask("energy-grid-stress")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = {k: payload[k] for k in REQUIRED_FIELDS}
    X = pd.DataFrame([row])

    proba = float(model.predict_proba(X)[:, 1][0])
    return jsonify({"grid_stress_risk": proba})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)
