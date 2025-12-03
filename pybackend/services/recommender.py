from typing import Dict, Any
from pybackend.services.finance import get_ticker_info, get_historical, compute_metrics
from pybackend.services.ml import forecast_next_price
from pybackend.services.sentiment import fetch_headlines, sentiment_score, generate_reason

def recommend_for_symbol(symbol: str) -> Dict[str, Any]:
    info = get_ticker_info(symbol)
    dfh = get_historical(symbol, period="70d")
    last_close = None
    closes = []
    if dfh is not None and not dfh.empty:
        last_close = float(dfh["Close"].tail(1).values[0])
        closes = [float(x) for x in dfh["Close"].tail(60).tolist()]
    forecast = forecast_next_price(closes)
    change = None
    direction = "Neutral"
    if forecast is not None and last_close is not None:
        change = float(forecast) - float(last_close)
        direction = "Sube" if change > 0 else "Baja" if change < 0 else "Neutral"
    metrics = compute_metrics(symbol)
    headlines = fetch_headlines(symbol)
    sent = sentiment_score(headlines)
    ext_reason = generate_reason(headlines)
    pe = info.get("trailingPE")
    beta = info.get("beta")
    internal_reason = None
    if pe is not None and pe <= 15:
        internal_reason = "Valuación atractiva por múltiplos bajos"
    elif pe is not None and pe >= 30:
        internal_reason = "Valuación exigente, requiere catalizadores"
    if internal_reason is None and beta is not None:
        if beta >= 1.2:
            internal_reason = "Riesgo elevado por alta beta"
        elif beta <= 0.8:
            internal_reason = "Perfil defensivo por baja beta"
    if internal_reason is None:
        internal_reason = "Perfil mixto según métricas internas"
    confidence = None
    if change is not None and last_close is not None:
        base = abs(change) / max(last_close, 1e-6)
        s = abs(sent) if sent is not None else 0.0
        confidence = min(1.0, base * 2 + s * 0.5)
    action = "Mantener"
    if direction == "Sube" and (sent is not None and sent > 0):
        action = "Comprar"
    elif direction == "Baja" and (sent is not None and sent < 0):
        action = "Vender"
    return {
        "symbol": symbol,
        "direction": direction,
        "forecast": forecast,
        "lastClose": last_close,
        "confidence": confidence,
        "action": action,
        "external_reason": ext_reason,
        "internal_reason": internal_reason,
        "metrics": metrics,
        "info": info,
        "sentiment": sent,
    }

