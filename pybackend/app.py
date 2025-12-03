from typing import List, Dict, Any
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _try_import_fastapi():
    try:
        from fastapi import FastAPI, Body
        from fastapi.responses import JSONResponse
        return FastAPI, Body, JSONResponse
    except Exception:
        return None, None, None

from pybackend.services.finance import compute_metrics, compute_metrics_bulk, get_historical, get_ticker_info
from pybackend.services.ml import aplicar_kmeans, forecast_next_price
from pybackend.services.sentiment import fetch_headlines, sentiment_score

FastAPI, Body, JSONResponse = _try_import_fastapi()
app = FastAPI() if FastAPI else None

def quotes_payload(symbol: str) -> Dict[str, Any]:
    metrics = compute_metrics(symbol)
    info = get_ticker_info(symbol)
    df = get_historical(symbol, period="70d")
    last_close = None
    if df is not None and not df.empty:
        last_close = float(df["Close"].tail(1).values[0])
    headlines = fetch_headlines(symbol)
    sent = sentiment_score(headlines)
    closes = []
    if df is not None and not df.empty:
        closes = [float(x) for x in df["Close"].tail(60).tolist()]
    forecast = forecast_next_price(closes)
    return {
        "symbol": symbol,
        "lastClose": last_close,
        "metrics": metrics,
        "info": info,
        "headlines": headlines,
        "sentiment": sent,
        "forecast": forecast,
    }

if app:
    @app.get("/quotes")
    def quotes(symbol: str):
        return JSONResponse(content=quotes_payload(symbol))

    @app.post("/api/analyze-market")
    def analyze_market(payload: Dict[str, Any] = Body(...)):
        symbols = payload.get("symbols") or []
        data = compute_metrics_bulk(symbols)
        clusters = aplicar_kmeans(data, n_clusters=3)
        return JSONResponse(content=clusters)

if __name__ == "__main__":
    sample = quotes_payload("AAPL")
    print(json.dumps(sample)[:1000])

