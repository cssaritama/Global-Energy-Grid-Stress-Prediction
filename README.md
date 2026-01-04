# Global Energy Grid Stress & Resilience Prediction 

Global Energy Grid Stress & Resilience Prediction is a production-ready machine
learning system designed to anticipate periods of high operational stress in
electricity grids.

By analyzing historical power system signals  including electricity demand patterns, renewable energy penetration, and volatility indicators the model identifies conditions that may lead to grid instability, supply shortages, or service disruptions.

This project is suitable for real-world deployment, decision support, and
integration into energy analytics pipelines.

---

## 1. Why This Project Matters

Electricity grids are critical infrastructure. Increasing electrification, climate-driven demand spikes, and the rapid integration of intermittent renewable energy sources are placing unprecedented pressure on power systems worldwide.

Early identification of grid stress risk enables:

- Proactive grid management and load balancing  
- Improved resilience against blackouts and extreme events  
- Safer and more reliable renewable energy integration  
- Reduced operational and economic risk  

---

## 2. Real-World Impact and Applications

This system can be used by:

- Electricity grid operators and utilities  
- Energy regulators and policy makers  
- Infrastructure and resilience planners  
- Energy analytics, consulting, and forecasting teams  

From a commercial perspective, similar predictive systems are already deployed as
decision-support tools, enabling cost reduction, reliability improvements, and
data-driven planning in modern energy systems.

---

## 3. Machine Learning Task

**Binary classification**

- `0` → Normal grid operation  
- `1` → High grid stress risk  

The trained model is exposed via a REST API and can be deployed locally or in the cloud.

---

## 4. Dataset

The project uses the **Open Power System Data (OPSD) – Time Series** dataset, which
provides hourly electricity load and renewable generation data for multiple European
countries.

- Source: https://data.open-power-system-data.org/time_series/
- Data type: public, authoritative, reproducible

### Data Reproducibility

The raw dataset is large and is not committed to the repository.

A fully automated data pipeline is provided. Running:

```bash
python src/download_data.py
```

will download the raw data and generate the processed dataset required for training and
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

### Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Build dataset

```bash
python src/download_data.py
```

### Train the model

```bash
python src/train.py
```

### Run the prediction service

```bash
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

### Build image

```bash
docker build -t energy-grid-stress .
```

### Run container

```bash
docker run -p 9696:9696 energy-grid-stress
```

---

## 9. Cloud Deployment

The service is designed for deployment on Google Cloud Run. Complete deployment
commands and configuration are provided in:

```
deploy/cloudrun.md
```

The deployed service exposes the same REST API interface and can be consumed by
external systems, dashboards, or monitoring tools.

---

## License & Attribution

Open Power System Data (OPSD) is provided by the OPSD project and should be cited
according to their terms.
