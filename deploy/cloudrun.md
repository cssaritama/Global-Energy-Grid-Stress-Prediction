# Deploy to Google Cloud Run (Optional, for full points)

This guide deploys the containerized prediction service to **Google Cloud Run**.

## 1) Prerequisites

- Google Cloud account + billing enabled
- `gcloud` CLI installed and authenticated
- A Google Cloud project selected:

```bash
gcloud config set project YOUR_PROJECT_ID
```

## 2) Train the model (local)

Cloud Run image should contain `model/model.bin`. Build it locally first:

```bash
python src/download_data.py
python src/train.py
```

## 3) Build and push the container image

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/energy-grid-stress:latest
```

## 4) Deploy

```bash
gcloud run deploy energy-grid-stress       --image gcr.io/YOUR_PROJECT_ID/energy-grid-stress:latest       --platform managed       --region europe-west1       --allow-unauthenticated
```

After deployment, Cloud Run will print a **Service URL**.

## 5) Test the deployed service (copy-paste)

```bash
SERVICE_URL="https://YOUR-SERVICE-URL.a.run.app"

curl -X POST "$SERVICE_URL/predict"       -H "Content-Type: application/json"       -d '{
    "country": "DE",
    "demand_peak_ratio": 1.12,
    "renewable_share": 0.38,
    "renewable_volatility": 0.09,
    "load_growth_rate": 0.015,
    "capacity_margin": 0.17
  }'
```

## 6) For maximum points

Save one of the following in your repository:
- A screenshot of the successful `curl` response
- A short video showing the request/response
- Or add the final Cloud Run URL to your README
