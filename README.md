# Global Energy Grid Stress & Resilience Prediction

Global Energy Grid Stress & Resilience Prediction is a production-ready machine
learning system designed to anticipate periods of high operational stress in
electricity grids using historical operational data.

The project is designed for real-world usage, operational decision support, and
deployment in modern cloud environments.

---

## 1. Problem Description

Electricity grids are critical infrastructure systems. In recent years, increasing
electrification, climate-driven demand peaks, and the rapid integration of intermittent
renewable energy sources have significantly increased operational stress on power grids
worldwide.

Periods of high grid stress can lead to:
- Supply shortages and service disruptions
- Increased risk of blackouts
- Higher operational and balancing costs
- Reduced system resilience during extreme events

**Objective**

The objective of this project is to build a machine learning model capable of predicting
whether an electricity grid is operating under high stress risk, based on historical
signals such as demand intensity, renewable energy penetration, variability, and system
margin indicators.

The solution is exposed as a RESTful API and can be integrated into:
- Grid operation monitoring systems
- Energy analytics platforms
- Infrastructure resilience dashboards

---

## 2. Why This Project Matters

Early identification of grid stress enables:

- Proactive grid management and preventive actions  
- Safer integration of renewable energy sources  
- Improved resilience against extreme weather events  
- Reduction of economic and operational risks  

This type of predictive system aligns closely with real-world tools currently used by
utilities, energy regulators, and infrastructure planners.

---

## 3. Machine Learning Task

**Binary classification**

- `0` → Normal grid operation  
- `1` → High grid stress risk  

The trained model produces a probability score and is accessible via a web API.

---

## 4. Dataset

The project uses the **Open Power System Data (OPSD) – Time Series** dataset, which
contains hourly electricity load and renewable generation data for multiple European
countries.

- Source: https://data.open-power-system-data.org/time_series/
- Data type: public, authoritative, reproducible

### Data Reproducibility

The raw dataset is large and is therefore not committed to the repository.

A fully automated data pipeline is provided. Running:

```bash
python src/download_data.py
```

downloads the raw data and generates the processed dataset required for training and
evaluation:

- `data/raw/time_series_60min_singleindex.csv`
- `data/processed/energy_grid_daily.csv`

---

## 5. Repository Structure

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── notebook.ipynb
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── model/
│   └── model.bin
├── tests/
│   └── test_predict.py
└── deploy/
    └── cloudrun.md
```

---

## 6. Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/predict.py
```

The service will be available at:

```
http://localhost:9696/predict
```

---

## 7. Example API Request

```bash
curl -X POST http://localhost:9696/predict   -H "Content-Type: application/json"   -d '{
    "country": "DE",
    "demand_peak_ratio": 1.12,
    "renewable_share": 0.38,
    "renewable_volatility": 0.09,
    "load_growth_rate": 0.015,
    "capacity_margin": 0.17
  }'
```

Example response:

```json
{"grid_stress_risk": 0.73}
```

---

## 8. Docker

```bash
docker build -t energy-grid-stress .
docker run -p 9696:9696 energy-grid-stress
```

---

## 9. Cloud Deployment

The service is designed for deployment on Google Cloud Run. Complete deployment
commands and configuration are provided in:

```
deploy/cloudrun.md
```

---

## License & Attribution

Open Power System Data (OPSD) is provided by the OPSD project and should be cited
according to their terms.
