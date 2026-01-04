# Global Energy Grid Stress & Resilience Prediction (Capstone 1)

A production-style machine learning service that predicts **high grid stress risk**
from historical power system signals (electricity demand, renewable penetration,
and volatility indicators).

This project was developed as **Capstone 1** for the **Machine Learning Zoomcamp**
and follows the official evaluation criteria: problem description, extensive EDA,
multiple models with tuning, reproducibility, deployment, dependency management,
containerization, and cloud deployment.

---

## 1. Problem Description

Modern electricity grids face increasing operational stress due to:
- rising electricity demand and electrification,
- extreme weather events and climate variability,
- higher shares of intermittent renewable energy (wind and solar).

Grid stress can lead to supply shortages, price spikes, or blackouts, affecting
critical infrastructure such as hospitals and transportation systems.

**Goal:** predict whether a power grid is under **high stress risk** given recent
load patterns, renewable penetration, and system margin indicators.

**Machine Learning Task:** Binary classification  
- `0` → Normal operation  
- `1` → High grid stress risk  

The trained model is exposed via a REST API and can be deployed locally or in the cloud.

---

## 2. Dataset

This project uses the **Open Power System Data (OPSD) – Time Series** dataset, which
provides hourly electricity load and generation data for multiple European countries.

- Source: https://data.open-power-system-data.org/time_series/
- Data type: real, public, reproducible

### Data Reproducibility Note

The `data/` directory is intentionally empty when cloning the repository.

The raw dataset used in this project is large and therefore not committed to Git.
To ensure full reproducibility, the project includes an automated data pipeline.

Running the following command will download the raw data and generate the processed
dataset required for training and evaluation:

```bash
python src/download_data.py
```

This will create:
- `data/raw/time_series_60min_singleindex.csv`
- `data/processed/energy_grid_daily.csv`

---

## 3. Repository Structure

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── data/
│   ├── raw/           # generated automatically
│   └── processed/     # generated automatically
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

## 4. Setup (Local)

### 4.1 Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
# .\venv\Scripts\Activate.ps1  # Windows
```

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Download data and build dataset

```bash
python src/download_data.py
```

### 4.4 Train the model

```bash
python src/train.py
```

### 4.5 Run the prediction service

```bash
python src/predict.py
```

The service will be available at:

```
http://localhost:9696/predict
```

---

## 5. Example Request

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

## 6. Docker

### Build image

```bash
docker build -t energy-grid-stress .
```

### Run container

```bash
docker run -p 9696:9696 energy-grid-stress
```

---

## 7. Cloud Deployment (Google Cloud Run)

Step-by-step instructions are provided in:

```
deploy/cloudrun.md
```

### Cloud Deployment Proof

After deploying the service to Cloud Run, include **one screenshot** in the repository
showing a successful request to the deployed endpoint.

Recommended screenshot:
- Terminal with a `curl` request to the Cloud Run service URL
- JSON response from `/predict`

Example (to be executed after deployment):

```bash
curl -X POST https://YOUR-SERVICE-URL.a.run.app/predict   -H "Content-Type: application/json"   -d '{
    "country": "DE",
    "demand_peak_ratio": 1.12,
    "renewable_share": 0.38,
    "renewable_volatility": 0.09,
    "load_growth_rate": 0.015,
    "capacity_margin": 0.17
  }'
```

Save the screenshot as:

```
deploy/cloud_run_test.png
```

and reference it here to ensure full points for cloud deployment.

---

## 8. Notes

- The notebook (`notebooks/notebook.ipynb`) contains full data preparation,
  extensive EDA, feature importance analysis, and model comparison.
- Training logic is exported to `src/train.py`.
- The service is containerized and production-ready.

---

## License & Attribution

OPSD Time Series data is provided by Open Power System Data and should be cited
according to their terms.
