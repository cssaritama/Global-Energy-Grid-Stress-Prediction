import os
import pytest

# Importing the app will attempt to load the model.
# If the model is missing, we skip tests with a clear message.
try:
    from src.predict import app
except FileNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


def test_health_endpoint():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_endpoint():
    client = app.test_client()
    resp = client.post("/predict", json={
        "country": "DE",
        "demand_peak_ratio": 1.10,
        "renewable_share": 0.35,
        "renewable_volatility": 0.08,
        "load_growth_rate": 0.01,
        "capacity_margin": 0.15
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "grid_stress_risk" in data
    assert 0.0 <= data["grid_stress_risk"] <= 1.0
