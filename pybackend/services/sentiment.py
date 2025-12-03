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
    # 1. Estrategia Principal: yfinance (API oficial/no oficial robusta)
    try:
        import yfinance as yf
        tick = yf.Ticker(symbol)
        news = tick.news
        if news:
            titles = []
            for n in news:
                # Soporte para estructura nueva y vieja de yfinance
                t = n.get('title')
                if not t and 'content' in n:
                    t = n['content'].get('title')
                if t:
                    titles.append(t)
            if titles:
                return titles
    except Exception as e:
        print(f"[ERROR] yfinance news falló para {symbol}: {e}")

    # 2. Estrategia Secundaria: GoogleNews
    GN = _try_import_googlenews()
    if GN is not None:
        try:
            # Intentar búsqueda en inglés también para stocks internacionales
            gn = GN(lang='en', region='US') 
            gn.search(f"{symbol} stock")
            entries = gn.result()
            if entries:
                return [e.get('title') for e in entries if e.get('title')]
        except Exception as e:
            print(f"[ERROR] GoogleNews falló para {symbol}: {e}")
            pass

    # 3. Estrategia de Respaldo: Scraping directo (Yahoo Finance)
    try:
        import urllib.request
        import ssl
        import certifi
        
        # Contexto SSL permisivo para evitar errores de certificados locales
        ctx = ssl.create_default_context(cafile=certifi.where())
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(
            f"https://finance.yahoo.com/quote/{symbol}",
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=5, context=ctx) as response:
            html = response.read().decode("utf-8", errors="ignore")
        titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html, flags=re.IGNORECASE)
        clean = [re.sub('<[^<]+?>', '', t).strip() for t in titles]
        return [t for t in clean if t and len(t) > 10]
    except Exception as e:
        print(f"[ERROR] Fallo fetch_headlines backup para {symbol}: {e}")
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
    
    # Fallback heurístico (Español e Inglés básico)
    pos = {"sube", "ganancia", "positivo", "alcista", "crece", "mejora", "supera", 
           "up", "gain", "positive", "bullish", "growth", "improve", "beat", "rise", "high"}
    neg = {"cae", "pérdida", "negativo", "bajista", "disminuye", "empeora", "falla", 
           "down", "loss", "negative", "bearish", "drop", "worse", "miss", "fall", "low"}
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
    
    # IA / Tech
    if any(w in tl for w in ["ai", "inteligencia artificial", "machine learning", "gpt", "generative", "chip", "nvidia"]):
        return "Impulso por iniciativas de IA y tecnología"
    
    # Productos
    if any(w in tl for w in ["producto", "nuevo", "presenta", "lanza", "product", "new", "launch", "release", "unveil", "iphone", "mac"]):
        return "Nuevos productos/servicios generan expectativa"
    
    # Alianzas
    if any(w in tl for w in ["acuerdo", "alianza", "partnership", "deal", "merge", "acquisition", "compra"]):
        return "Movimientos corporativos y alianzas estratégicas"
    
    # Financiero
    if any(w in tl for w in ["resultado", "ingreso", "beneficio", "ganancias", "earnings", "revenue", "profit", "report", "quarter"]):
        return "Resultados financieros clave en el foco"
    
    # Legal/Regulatorio
    if any(w in tl for w in ["regulación", "multas", "investigación", "lawsuit", "ban", "fine", "antitrust"]):
        return "Riesgos regulatorios o legales presentes"
        
    # Mercado general
    if any(w in tl for w in ["fed", "rate", "tasa", "inflación", "inflation", "market", "mercado"]):
        return "Factores macroeconómicos influyen en el precio"
        
    return "Narrativa de mercado mixta; monitorear volatilidad"
