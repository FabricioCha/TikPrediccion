from typing import Dict, List

CATEGORIES: Dict[str, List[str]] = {
    "Tecnología": ["AAPL", "MSFT", "NVDA", "META", "GOOG"],
    "Servicios de Comunicación": ["GOOGL", "NFLX", "DIS", "TTWO", "T"],
    "Consumo Discrecional": ["AMZN", "TSLA", "NKE", "HD", "SBUX"],
    "Financieras": ["JPM", "BAC", "V", "MA", "GS"],
    "Salud": ["UNH", "JNJ", "PFE", "MRK", "ABBV"],
    "Energía": ["XOM", "CVX", "COP", "SLB", "EOG"],
}

def get_symbols_by_category(category: str, n: int = 5) -> List[str]:
    arr = CATEGORIES.get(category) or []
    return arr[:n]

