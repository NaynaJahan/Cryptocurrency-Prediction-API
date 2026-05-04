# ETH next-day HIGH API

Predict tomorrow’s HIGH price (USD) for Ethereum (ETH) using the latest daily candle and engineered features. Expose the prediction via a simple FastAPI service for your team’s Streamlit app. The repository to the Streamlit app can be found here: [Cryptocurrency-Prediction-Data-Product-Streamlit](https://github.com/NaynaJahan/Cryptocurrency-Prediction-Data-Product-Streamlit)

### Endpoints
- `/`       : API description including model/meta summary and endpoints information
- `/health/`: Health check (200 OK)
- `/predict/eth`: Predict next-day HIGH (USD) for Ethereum using live Kraken candles

---

## Overview
- Token: Ethereum (ETH) only
- Live data: Kraken OHLC (daily candles)
- Model: RandomForestRegressor trained on log-ratio target
- Target: log(high[t+1]) - log(close[t])
- Prediction formula: pred_usd = close_t * exp(alpha * yhat_ratio)
- Model + metadata: stored in models/ (eth_rf_nextdayhigh.joblib, meta.json)

---
## Repository Structure

```
amla_api/
├── app/
│   └── main.py             
├── models/
│   ├── eth_rf_nextdayhigh.joblib
│   └── meta.json           
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── github.txt               
└── README.md

```
----

## Development and Running:

### Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Test
```bash
curl http://127.0.0.1:8000/health/
curl http://127.0.0.1:8000/predict/eth
```

### Docker

#### Build & run locally
```bash
docker build -t eth-nextday-api:latest .
docker run --rm -p 8000:8000 --name eth-api eth-nextday-api:latest
# If 8000 is busy: -p 8010:8000 and hit http://127.0.0.1:8010
```

#### Push multi-arch image to Docker Hub
```bash
docker login -u <username> -p <password> docker.io
docker buildx create --use --name multi || true
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t <username>/advmla_lab4_<studentid>:latest \
  --push .
```

### Deploy on Render (Existing Image)
1. Create Web Service → Existing Image.
2. Image URL: docker.io/nayna578/advmla_finalast_2523873:latest
3. Port: 8000 (internal)
4. Health Check Path: /health/
5. Deploy → test:
    - https://advmla-finalast-25238736-latest.onrender.com
    - https://advmla-finalast-25238736-latest.onrender.com/health/
    - https://advmla-finalast-25238736-latest.onrender.com/predict/eth

### Configuration (env vars)
- MODEL_PATH (default models/eth_rf_nextdayhigh.joblib)
- META_PATH (default models/meta.json)
- KRAKEN_PAIR (default ETHUSD)
- KRAKEN_INTERVAL minutes (default 1440)


---
## Example response (/predict/eth):

```bash
{
  "token": "ETH",
  "as_of_utc": "2025-10-31T00:00:00+00:00",
  "close_t_usd": 3832.08,
  "prediction_usd": 3911.676081,
  "yhat_ratio": 0.0205582106,
  "alpha_used": 1,
  "model": "RandomForestRegressor",
  "features_used": ["ret1","ret3","ret7","range_rel","rv7","vol_z7","rollmax7_rel","rollmin7_rel","vol_was_na"],
  "source": {"kraken_pair": "ETHUSD", "interval_min": 1440},
  "notes": "Prediction is for the next daily HIGH based on the latest complete candle."
}
```
----
## How it Works (short)

1. Fetch ~400 recent ETHUSD daily candles from Kraken.
2. Recompute the same features used in training (see meta.json → features).
3. Score the latest feature row with the trained model.
4. Convert the model’s log-ratio output to USD with close_t * exp(alpha * yhat).

### **Notes**

- Single-token service: requests to other tokens should be rejected.
- Feature parity: If you retrain with new features, update meta.json and the model file together.
- Time zone: Kraken candles are UTC. The API returns as_of_utc; your UI can compute a local “predict_for_date” if needed.
