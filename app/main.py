from __future__ import annotations
import os, json, time, math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

MODEL_PATH = os.getenv("MODEL_PATH", "models/eth_rf_nextdayhigh.joblib")
META_PATH  = os.getenv("META_PATH",  "models/meta.json")
PAIR       = os.getenv("KRAKEN_PAIR", "ETHUSD")  
INTERVAL   = int(os.getenv("KRAKEN_INTERVAL", "1440"))
SINCE_SECS = int((datetime.now(timezone.utc) - timedelta(days=400)).timestamp())

app = FastAPI(title="ETH Next-Day High API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    MODEL = load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

with open(META_PATH, "r") as f:
    META = json.load(f)

FEATURES: List[str] = META.get("features") or META.get("feature_order")
if not FEATURES:
    raise RuntimeError("No 'features' listed in meta.json")

ALPHA = float(META.get("alpha", 1.0)) 

# Helpers
def fetch_kraken_ohlc(pair: str, interval: int, since: int) -> pd.DataFrame:
    """
    Kraken OHLC: https://api.kraken.com/0/public/OHLC?pair=ETHUSD&interval=1440&since=...
    Returns DataFrame with index=UTC datetime, cols=open, high, low, close, volume
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval, "since": since}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Kraken error: {r.text}")

    data = r.json()
    if "result" not in data:
        raise HTTPException(status_code=502, detail=f"Malformed Kraken response: {data}")

    result = data["result"]
    key = next(k for k in result.keys() if k != "last")
    rows = result[key]

    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    for c in ["open","high","low","close","vwap","volume","count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["open","high","low","close","volume"]]

def compute_latest_feature_row(hist: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    """
    Recompute the same features used during training and return the last 1 row, in the exact order.
    Feature set expected from meta.json:
      ret1, ret3, ret7, range_rel, rv7, vol_z7, rollmax7_rel, rollmin7_rel, vol_was_na
    """
    df = hist.copy().sort_index()
    vol_na = df["volume"].isna() | (df["volume"]<=0)
    df["volume_filled"] = df["volume"].where(~vol_na, np.nan).ffill().bfill()
    df["vol_was_na"]    = vol_na.astype(int)

    c = df["close"]
    df["ret1"] = np.log(c / c.shift(1))
    df["ret3"] = np.log(c / c.shift(3))
    df["ret7"] = np.log(c / c.shift(7))

    df["range_rel"] = (df["high"] - df["low"]) / df["close"]

    df["rv7"] = df["ret1"].rolling(7).std()

    vl = np.log1p(df["volume_filled"])
    df["vol_z7"] = (vl - vl.rolling(7).mean()) / vl.rolling(7).std()

    df["rollmax7_rel"] = (df["high"].rolling(7).max() / df["close"]) - 1.0
    df["rollmin7_rel"] = (df["low"].rolling(7).min()  / df["close"]) - 1.0

    last_row = df.tail(1)
    feat = last_row.reindex(columns=feature_order)

    if feat.isna().any(axis=None):
        missing = feat.columns[feat.isna().any()].tolist()
        raise HTTPException(status_code=503, detail=f"Not enough recent candles to compute features: {missing}")

    feat.attrs["close_t"] = float(last_row["close"].iloc[0])
    feat.attrs["as_of_utc"] = last_row.index[-1].isoformat()
    return feat

def ratio_to_price(yhat_ratio: float, close_t: float, alpha: float = 1.0) -> float:
    return float(close_t * math.exp(alpha * yhat_ratio))

@app.get("/", tags=["info"])
def root() -> Dict[str, Any]:
    return {
        "project": "ETH next-day HIGH price prediction",
        "objective": "Predict tomorrow’s HIGH (USD) for Ethereum using last daily candle and engineered features.",
        "token_supported": ["eth","ethereum","ETH","ETHEREUM"],
        "endpoints": {
            "/": "This message",
            "/health/": "200 OK if service is running",
            "/predict/{token}": "Return predicted next-day HIGH (USD) using live Kraken OHLC",
        },
        "expected_output": {"prediction_usd": 0.0, "as_of_utc": "ISO time", "close_t": 0.0},
        "model": META.get("model", "RandomForestRegressor"),
        "features": FEATURES,
        "meta": {"train_end": META.get("train_end"), "val_end": META.get("val_end"), "alpha": ALPHA},
        "repo": "https://github.com/naynajn/amla_api.git",
    }

@app.get("/health/", tags=["info"])
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "ETH API up"}

@app.get("/predict/{token}", tags=["prediction"])
def predict(token: str) -> Dict[str, Any]:
    if token.lower() not in {"eth","ethereum"}:
        raise HTTPException(status_code=400, detail="Unsupported token. Use 'eth' or 'ethereum'.")

    hist = fetch_kraken_ohlc(PAIR, INTERVAL, SINCE_SECS)

    feat = compute_latest_feature_row(hist, FEATURES)
    close_t = feat.attrs["close_t"]
    as_of   = feat.attrs["as_of_utc"]

    yhat_ratio = float(MODEL.predict(feat)[0])
    pred_usd   = ratio_to_price(yhat_ratio, close_t, alpha=ALPHA)

    return {
        "token": "ETH",
        "as_of_utc": as_of,
        "close_t_usd": round(close_t, 6),
        "prediction_usd": round(pred_usd, 6),
        "yhat_ratio": yhat_ratio,
        "alpha_used": ALPHA,
        "model": META.get("model", "RandomForestRegressor"),
        "features_used": FEATURES,
        "source": {"kraken_pair": PAIR, "interval_min": INTERVAL},
        "notes": "Prediction is for the next daily HIGH based on the latest complete candle.",
    }