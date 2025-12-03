from typing import List, Dict, Any, Optional
import math

def _try_import_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        return {
            "KMeans": KMeans,
            "StandardScaler": StandardScaler,
            "LinearRegression": LinearRegression,
        }
    except Exception:
        return None

def _np():
    import numpy as np
    return np

def _pd():
    import pandas as pd
    return pd

def _centroid_label(centroid: List[float]) -> str:
    vol = centroid[1]
    ret = centroid[0]
    if vol is None or ret is None:
        return "Mixto"
    if vol < 0.01 and ret >= 0:
        return "Refugio Seguro"
    if vol >= 0.02 and ret >= 0:
        return "Crecimiento Agresivo"
    if vol >= 0.02 and ret < 0:
        return "Especulativo"
    return "Mixto"

def aplicar_kmeans(tickers_data: List[Dict[str, Any]], n_clusters: int = 3) -> List[Dict[str, Any]]:
    pd = _pd()
    np = _np()
    df = pd.DataFrame(tickers_data)
    df = df[["symbol", "returns", "volatility", "volume_avg"]].dropna()
    if df.empty:
        return []
    features = df[["returns", "volatility", "volume_avg"]]
    skl = _try_import_sklearn()
    if skl is not None:
        scaler = skl["StandardScaler"]()
        X = scaler.fit_transform(features)
        km = skl["KMeans"](n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X)
        df["cluster"] = labels
        centers = km.cluster_centers_
        inv_centers = scaler.inverse_transform(centers)
    else:
        mu = features.mean()
        sigma = features.std().replace(0, 1)
        X = (features - mu) / sigma
        rng = np.random.default_rng(42)
        centroids = X.sample(n_clusters, random_state=42).to_numpy()
        for _ in range(50):
            dists = np.linalg.norm(X.to_numpy()[:, None, :] - centroids[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            new_centroids = np.vstack([
                X.to_numpy()[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
                for i in range(n_clusters)
            ])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        df["cluster"] = labels
        inv_centers = (centroids * sigma.to_numpy()) + mu.to_numpy()
    group_map: Dict[int, str] = {}
    for i in range(n_clusters):
        c = inv_centers[i]
        group_map[i] = _centroid_label([float(c[0]), float(c[1])])
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({
            "symbol": row["symbol"],
            "group": group_map.get(int(row["cluster"]), "Mixto"),
            "returns": float(row["returns"]),
            "volatility": float(row["volatility"]),
        })
    return out

def forecast_next_price(closes: List[float]) -> Optional[float]:
    np = _np()
    if not closes or len(closes) < 10:
        return None
    n = len(closes)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(closes, dtype=float)
    skl = _try_import_sklearn()
    try:
        if skl is not None:
            lr = skl["LinearRegression"]()
            lr.fit(X, y)
            return float(lr.predict(np.array([[n]]))[0])
        coeffs = np.polyfit(np.arange(n), y, deg=1)
        return float(coeffs[0] * n + coeffs[1])
    except Exception:
        return None

