import math
import time
import random
from typing import List, Dict, Optional

def _get_session():
    # requests_cache es incompatible con las nuevas versiones de yfinance que usan curl_cffi
    # Se elimina el uso de caché persistente y se devuelve None o una sesión requests simple
    return None

    # El bloque anterior causaba conflictos con curl_cffi
    # try:
    #     import requests_cache
    #     base = requests_cache.CachedSession('yfinance.cache', expire_after=900)
    #     # ...


def _stooq_download(symbol: str, period_days: int = 180):
    try:
        import pandas as pd
        import urllib.request
        s = symbol.lower()
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        with urllib.request.urlopen(url, timeout=10) as resp:
            csv = resp.read().decode("utf-8", errors="ignore")
        from io import StringIO
        df = pd.read_csv(StringIO(csv))
        if df is None or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        # Tail to desired period
        return df.tail(period_days)
    except Exception:
        return None

def _import_yf():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None

_INFO_CACHE: Dict[str, Dict[str, Optional[float]]] = {}

def get_ticker_info(symbol: str) -> Dict[str, Optional[float]]:
    yf = _import_yf()
    if yf is None:
        return {
            "symbol": symbol,
            "trailingPE": None,
            "beta": None,
            "marketCap": None,
            "sector": None,
        }
    if symbol in _INFO_CACHE:
        return _INFO_CACHE[symbol]
    try:
        session = _get_session()
        t = yf.Ticker(symbol, session=session) if session is not None else yf.Ticker(symbol)
        info = t.info or {}
        # Fast info overlay when available
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                if info.get("marketCap") is None:
                    info["marketCap"] = getattr(fi, "market_cap", None)
        except Exception:
            pass
        result = {
            "symbol": symbol,
            "trailingPE": info.get("trailingPE"),
            "beta": info.get("beta"),
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
        }
        _INFO_CACHE[symbol] = result
        return result
    except Exception:
        result = {
            "symbol": symbol,
            "trailingPE": None,
            "beta": None,
            "marketCap": None,
            "sector": None,
            "shortName": None,
            "longName": None,
        }
        _INFO_CACHE[symbol] = result
        return result

_HIST_CACHE: Dict[str, any] = {}

def get_historical(symbol: str, period: str = "90d"):
    yf = _import_yf()
    if yf is None:
        return None
    cache_key = f"{symbol}:{period}"
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key]
    # Retry/backoff + Ticker.history fallback
    session = _get_session()
    for attempt in range(2):
        try:
            t = yf.Ticker(symbol, session=session) if session is not None else yf.Ticker(symbol)
            df = t.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                _HIST_CACHE[cache_key] = df
                return df
        except Exception as e:
            print(f"[ERROR CRÍTICO] Fallo al descargar {symbol} en get_historical (Ticker.history): {str(e)}")
            pass
        try:
            kwargs = {"period": period, "interval": "1d", "progress": False, "auto_adjust": True, "threads": False}
            if session is not None:
                kwargs["session"] = session
            df = yf.download(symbol, **kwargs)
            if df is not None and not df.empty:
                _HIST_CACHE[cache_key] = df
                return df
        except Exception as e:
            print(f"[ERROR CRÍTICO] Fallo al descargar {symbol} en get_historical (yf.download): {str(e)}")
            pass
        time.sleep(0.5 + random.random()*0.5)
    # Fallback to Stooq CSV
    df = _stooq_download(symbol)
    if df is not None and not df.empty:
        _HIST_CACHE[cache_key] = df
        return df
    return None

def compute_metrics(symbol: str) -> Dict[str, Optional[float]]:
    df = get_historical(symbol, period="60d")
    if df is None or df.empty:
        return {"symbol": symbol, "returns": None, "volatility": None, "volume_avg": None}
    try:
        closes = df["Close"].astype(float)
        rets = closes.pct_change().dropna()
        last30 = rets.tail(30)
        volatility = float(last30.std()) if len(last30) > 0 else float(rets.std()) if len(rets) else None
        returns_mean = float(rets.mean()) if len(rets) else None
        vol_series = df["Volume"].astype(float)
        volume_avg = float(vol_series.tail(60).mean()) if len(vol_series) else None
        if volatility is not None and math.isnan(volatility):
            volatility = None
        if returns_mean is not None and math.isnan(returns_mean):
            returns_mean = None
        if volume_avg is not None and math.isnan(volume_avg):
            volume_avg = None
        return {
            "symbol": symbol,
            "returns": returns_mean,
            "volatility": volatility,
            "volume_avg": volume_avg,
        }
    except Exception:
        return {"symbol": symbol, "returns": None, "volatility": None, "volume_avg": None}

def compute_metrics_bulk(symbols: List[str]) -> List[Dict[str, Optional[float]]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out_map: Dict[str, Dict[str, Optional[float]]] = {}
    workers = max(1, min(3, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_metrics, s): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                out_map[s] = fut.result()
            except Exception:
                out_map[s] = {"symbol": s, "returns": None, "volatility": None, "volume_avg": None}
            time.sleep(0.3)
    return [out_map[s] for s in symbols]

