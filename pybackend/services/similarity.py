import math
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# 1. Representación de Vectores
# -----------------------------------------------------------------------------

def extract_features(reco_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrae y normaliza parcialmente las características clave de un objeto de recomendación.
    
    El objeto reco_data se espera que venga de recommender.recommend_for_symbol, conteniendo:
    - info: {trailingPE, beta, ...}
    - metrics: {volatility, returns, ...}
    - sentiment: float
    
    Retorna un diccionario con claves: pe, beta, volatility, recent_return, sentiment.
    Los valores None se convierten en 0.0 o un promedio neutro para el cálculo.
    """
    info = reco_data.get("info", {})
    metrics = reco_data.get("metrics", {})
    
    # Extracción con valores por defecto seguros
    pe = info.get("trailingPE")
    beta = info.get("beta")
    volatility = metrics.get("volatility")
    recent_return = metrics.get("returns")
    sentiment = reco_data.get("sentiment")
    
    # Manejo de faltantes (Imputación simple)
    # En un sistema prod, usaríamos la media del sector. Aquí usamos neutros razonables.
    vector = {
        "pe": float(pe) if pe is not None else 20.0,        # PE promedio aprox
        "beta": float(beta) if beta is not None else 1.0,   # Beta de mercado
        "volatility": float(volatility) if volatility is not None else 0.02, # Volatilidad diaria base
        "recent_return": float(recent_return) if recent_return is not None else 0.0,
        "sentiment": float(sentiment) if sentiment is not None else 0.0
    }
    return vector

def normalize_vectors(vectors: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Aplica normalización Min-Max a un lote de vectores para que todas las features
    tengan el mismo peso en la distancia Manhattan.
    """
    if not vectors:
        return []
        
    keys = vectors[0].keys()
    min_max = {k: {"min": float('inf'), "max": float('-inf')} for k in keys}
    
    # 1. Encontrar rangos
    for v in vectors:
        for k in keys:
            val = v[k]
            if val < min_max[k]["min"]: min_max[k]["min"] = val
            if val > min_max[k]["max"]: min_max[k]["max"] = val
            
    # 2. Normalizar (x - min) / (max - min)
    normalized_vectors = []
    for v in vectors:
        new_v = {}
        for k in keys:
            denom = min_max[k]["max"] - min_max[k]["min"]
            if denom == 0:
                new_v[k] = 0.5 # Si todos son iguales, valor medio
            else:
                new_v[k] = (v[k] - min_max[k]["min"]) / denom
        normalized_vectors.append(new_v)
        
    return normalized_vectors

# -----------------------------------------------------------------------------
# 2. Cálculo de Distancia Manhattan
# -----------------------------------------------------------------------------

def manhattan_distance(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """
    Calcula la distancia Manhattan (L1 norm) entre dos vectores de características.
    D(x, y) = Σ |xi - yi|
    """
    distance = 0.0
    # Asumimos que v1 y v2 tienen las mismas claves gracias a extract_features
    for k in v1:
        val1 = v1.get(k, 0.0)
        val2 = v2.get(k, 0.0)
        distance += abs(val1 - val2)
    return distance

# -----------------------------------------------------------------------------
# 3. Implementación del Algoritmo k-NN
# -----------------------------------------------------------------------------

def find_similar_stocks(
    target_symbol: str, 
    pool_data: List[Dict[str, Any]], 
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Encuentra las k acciones más similares a target_symbol dentro de pool_data.
    
    Args:
        target_symbol: El símbolo base.
        pool_data: Lista de diccionarios completos de recomendación (output de recommend_for_symbol).
        k: Número de vecinos a retornar.
        
    Returns:
        Lista de los k diccionarios de recomendación más cercanos, con un campo extra 'distance'.
    """
    # 1. Identificar target y pool
    target_data = next((x for x in pool_data if x["symbol"] == target_symbol), None)
    if not target_data:
        return [] # Target no encontrado en el pool
        
    # 2. Construir vectores crudos
    # Mapeamos índice -> vector para poder recuperar la data original luego
    raw_vectors = []
    indexed_data = []
    
    for item in pool_data:
        vec = extract_features(item)
        raw_vectors.append(vec)
        indexed_data.append(item)
        
    # 3. Normalizar vectores (CRÍTICO para que PE no domine sobre Sentiment)
    norm_vectors = normalize_vectors(raw_vectors)
    
    # 4. Encontrar el vector normalizado del target
    # Como el orden se preserva, buscamos el índice del target en pool_data
    target_idx = -1
    for i, item in enumerate(pool_data):
        if item["symbol"] == target_symbol:
            target_idx = i
            break
            
    if target_idx == -1:
        return []
        
    target_vector_norm = norm_vectors[target_idx]
    
    # 5. Calcular distancias
    distances = []
    for i, vec_norm in enumerate(norm_vectors):
        if i == target_idx:
            continue # No compararse consigo mismo
            
        dist = manhattan_distance(target_vector_norm, vec_norm)
        distances.append({
            "symbol": indexed_data[i]["symbol"],
            "data": indexed_data[i],
            "distance": dist,
            "features": raw_vectors[i] # Útil para mostrar en UI por qué son similares
        })
        
    # 6. Ordenar y retornar Top K
    distances.sort(key=lambda x: x["distance"])
    return distances[:k]
