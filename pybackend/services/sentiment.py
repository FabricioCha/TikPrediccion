from typing import List, Optional
import re
import datetime as dt

def _try_import_googlenews():
    try:
        from GoogleNews import GoogleNews
        return GoogleNews
    except Exception:
        return None

def _try_import_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer
    except Exception:
        return None

def fetch_headlines(symbol: str, lookback_hours: int = 24) -> List[str]:
    GN = _try_import_googlenews()
    now = dt.datetime.utcnow()
    start = now - dt.timedelta(hours=lookback_hours)
    if GN is not None:
        try:
            gn = GN(lang='es')
            gn.set_time_range(start.strftime('%m/%d/%Y'), now.strftime('%m/%d/%Y'))
            gn.search(symbol)
            entries = gn.result()
            return [e.get('title') for e in entries if e.get('title')]
        except Exception:
            pass
    try:
        import urllib.request
        html = urllib.request.urlopen(f"https://finance.yahoo.com/quote/{symbol}").read().decode("utf-8", errors="ignore")
        titles = re.findall(r'<a[^>]+data-test="quote-news-link"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE)
        clean = [re.sub('<[^<]+?>', '', t).strip() for t in titles]
        return [t for t in clean if t]
    except Exception:
        return []

def sentiment_score(texts: List[str]) -> Optional[float]:
    if not texts:
        return None
    V = _try_import_vader()
    if V is not None:
        try:
            analyzer = V()
            scores = [analyzer.polarity_scores(t).get('compound', 0.0) for t in texts]
            return sum(scores) / len(scores) if scores else None
        except Exception:
            pass
    pos = {"sube", "ganancia", "positivo", "alcista", "crece", "mejora", "supera"}
    neg = {"cae", "pérdida", "negativo", "bajista", "disminuye", "empeora", "falla"}
    vals = []
    for t in texts:
        tl = t.lower()
        p = sum(1 for w in pos if w in tl)
        n = sum(1 for w in neg if w in tl)
        s = (p - n) / max(p + n, 1)
        vals.append(s)
    return sum(vals) / len(vals) if vals else None

def generate_reason(texts: List[str]) -> str:
    if not texts:
        return "Sin noticias relevantes recientes"
    tl = " ".join(texts).lower()
    if any(w in tl for w in ["ai", "inteligencia artificial", "machine learning"]):
        return "Impulso por iniciativas de IA y automatización"
    if any(w in tl for w in ["producto", "nuevo", "presenta", "lanza"]):
        return "Nuevos productos/servicios generan expectativa de crecimiento"
    if any(w in tl for w in ["acuerdo", "alianza", "partnership"]):
        return "Alianzas estratégicas podrían acelerar la adopción"
    if any(w in tl for w in ["resultado", "ingreso", "beneficio", "ganancias"]):
        return "Resultados financieros apuntan a desempeño sólido"
    if any(w in tl for w in ["regulación", "multas", "investigación"]):
        return "Riesgos regulatorios presentes; monitorear impacto"
    return "Narrativa de mercado mixta; considerar horizonte y riesgo"

