import streamlit as st
import pandas as pd
import numpy as np
from pybackend.services.finance import compute_metrics_bulk, get_historical, get_ticker_info
from pybackend.services.ml import aplicar_kmeans, forecast_next_price
from pybackend.services.sentiment import fetch_headlines, sentiment_score, generate_reason
import time
import logging
_log = logging.getLogger("yfinance")
_log.setLevel(logging.CRITICAL)
_log.propagate = False
from pybackend.services.catalog import CATEGORIES, get_symbols_by_category
from pybackend.services.recommender import recommend_for_symbol, recommend_similar_stocks

st.set_page_config(page_title="Stock Advisor BI", layout="wide")
st.title("Stock Advisor BI")
st.subheader("Selección por Categoría")
category = st.selectbox("Categoría", list(CATEGORIES.keys()), index=0)
if st.button("Cargar 5 símbolos de la categoría"):
    pre = get_symbols_by_category(category, n=5)
    st.session_state["symbols_by_category"] = pre

symbols_input = st.text_input("Símbolos (coma separada)", ", ".join(st.session_state.get("symbols_by_category", ["AAPL", "MSFT", "NVDA", "META", "GOOG"])) )
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

@st.cache_data(ttl=900)
def cached_metrics(symbols):
    return compute_metrics_bulk(symbols)

@st.cache_data(ttl=900)
def cached_info(sym):
    return get_ticker_info(sym)

@st.cache_data(ttl=900)
def cached_hist(sym, period):
    return get_historical(sym, period=period)

@st.cache_data(ttl=900)
def cached_headlines(sym):
    return fetch_headlines(sym)

@st.cache_data(ttl=900)
def cached_sent(texts):
    return sentiment_score(texts)

@st.cache_data(ttl=900)
def cached_reco(sym):
    return recommend_for_symbol(sym)

if "prefs" not in st.session_state:
    st.session_state["prefs"] = {}

if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

if st.button("Analizar Mercado"):
    st.session_state["analyzed"] = True

if st.session_state["analyzed"]:
    metrics = cached_metrics(symbols)
    df = pd.DataFrame(metrics)
    st.subheader("Métricas")
    st.dataframe(df)
    clusters = aplicar_kmeans(metrics, n_clusters=3)
    cldf = pd.DataFrame(clusters)
    st.subheader("Clustering K-Means")
    st.dataframe(cldf)
    if not cldf.empty:
        st.subheader("Scatter: Riesgo vs Retorno")
        cmap = {"Refugio Seguro": "#4CAF50", "Crecimiento Agresivo": "#FF9800", "Especulativo": "#F44336", "Mixto": "#2196F3"}
        cldf["color"] = cldf["group"].map(lambda g: cmap.get(g, "#2196F3"))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in cldf["group"].unique():
            sub = cldf[cldf["group"] == g]
            ax.scatter(sub["volatility"], sub["returns"], c=cmap.get(g, "#2196F3"), label=g)
        ax.set_xlabel("Riesgo (Volatilidad)")
        ax.set_ylabel("Retorno promedio diario")
        ax.legend()
        st.pyplot(fig)
    st.subheader("Tarjetas de Recomendación")
    cols = st.columns(min(4, max(1, len(symbols))))
    for idx, sym in enumerate(symbols):
        with cols[idx % len(cols)]:
            info = cached_info(sym)
            headlines = cached_headlines(sym)
            sent = cached_sent(headlines)
            reason = generate_reason(headlines)
            group = None
            if not cldf.empty:
                row = cldf[cldf["symbol"] == sym]
                if not row.empty:
                    group = row.iloc[0]["group"]
            st.markdown(f"**{info.get('longName') or info.get('shortName') or sym}**")
            st.write({
                "símbolo": sym,
                "sector": info.get("sector"),
                "tipo": group or "Mixto",
                "sentimiento": sent,
            })
            reco = cached_reco(sym)
            conf_val = reco.get("confidence")
            conf_str = f"{conf_val:.1%}" if conf_val is not None else "N/A"
            st.write({
                "predicción": reco.get("direction"),
                "acción": reco.get("action"),
                "confianza": conf_str,
            })
            st.write("Porque comprar:", reason)
            st.write("Motivo interno:", reco.get("internal_reason"))
            # Mecanismo de Fricción Cognitiva
            with st.expander("👍 Evaluar / Me gusta"):
                user_reason = st.radio(
                    "¿Cuál es el motivo principal de tu elección?",
                    [
                        "A: Perfil de Bajo Riesgo (Seguridad)",
                        "B: Potencial de Alto Crecimiento (Retorno)",
                        "C: Noticias y Sentimiento Positivo"
                    ],
                    key=f"reason_{sym}"
                )
                
                if st.button("Confirmar Voto", key=f"btn_conf_{sym}"):
                    # Lógica de validación (Cross-Check)
                    curr_group = group if group else "Mixto"
                    curr_dir = reco.get("direction", "Neutral")
                    curr_sent = reco.get("sentiment") or 0.0
                    
                    valid = True
                    warning_msg = ""
                    
                    if user_reason.startswith("A"):
                        if curr_group != "Refugio Seguro":
                            valid = False
                            warning_msg = f"Elegiste Seguridad, pero el activo es '{curr_group}'."
                    elif user_reason.startswith("B"):
                        if curr_dir != "Sube":
                            valid = False
                            warning_msg = f"Buscas Crecimiento, pero la predicción es '{curr_dir}'."
                    elif user_reason.startswith("C"):
                        if curr_sent <= 0:
                            valid = False
                            warning_msg = f"Te basas en Sentimiento, pero el score es {curr_sent:.2f} (no positivo)."
                    
                    if valid:
                        st.session_state["prefs"][sym] = "like_verified"
                        st.session_state.pop(f"warn_{sym}", None)
                        st.success("¡Elección consistente! Guardado.")
                        st.rerun()
                    else:
                        st.session_state[f"warn_{sym}"] = warning_msg
                        st.rerun()

                # Manejo de advertencia y confirmación secundaria
                if f"warn_{sym}" in st.session_state:
                    st.warning(f"Tu elección contradice los datos: {st.session_state[f'warn_{sym}']}")
                    st.write("¿Deseas proceder?")
                    if st.button("Sí, proceder de todos modos", key=f"btn_force_{sym}"):
                        st.session_state["prefs"][sym] = "like_verified"
                        st.session_state.pop(f"warn_{sym}", None)
                        st.success("Guardado bajo tu responsabilidad.")
                        st.rerun()

            if st.button("👎 No me gusta", key=f"dislike_{sym}"):
                st.session_state["prefs"][sym] = "dislike"
                st.session_state.pop(f"warn_{sym}", None)
                st.rerun()

            pref = st.session_state["prefs"].get(sym)
            if pref:
                st.caption(f"Preferencia: {pref}")
    st.subheader("Detalle por Símbolo")
    sel = st.selectbox("Selecciona símbolo", symbols)
    if sel:
        info = cached_info(sel)
        st.write({k: v for k, v in info.items()})
        dfh = cached_hist(sel, period="70d")
        closes = []
        if dfh is not None and not dfh.empty:
            st.line_chart(dfh["Close"].astype(float))
            closes = [float(x) for x in dfh["Close"].tail(60).tolist()]
        f = forecast_next_price(closes)
        st.write({"forecast_next": f})
        headlines = cached_headlines(sel)
        st.write(headlines)
        sent = cached_sent(headlines)
        st.write({"sentiment": sent, "porque": generate_reason(headlines)})
        
        # -----------------------------------------------------
        # Módulo de Similaridad (k-NN con Manhattan Distance)
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader(f"Acciones Similares a {sel}")
        st.caption("Basado en proximidad Manhattan (PE, Beta, Volatilidad, Retorno, Sentimiento)")
        
        # Recolectar datos de todo el pool visible para comparar
        pool_data = []
        for s in symbols:
            pool_data.append(cached_reco(s))
            
        similar_stocks = recommend_similar_stocks(sel, pool_data, k=4)
        
        if similar_stocks:
            scols = st.columns(len(similar_stocks))
            for i, item in enumerate(similar_stocks):
                with scols[i]:
                    s_sym = item["symbol"]
                    dist = item["distance"]
                    feats = item["features"]
                    
                    st.markdown(f"**{s_sym}**")
                    st.caption(f"Distancia: {dist:.4f}")
                    st.json({
                        "PE": f"{feats['pe']:.1f}",
                        "Beta": f"{feats['beta']:.2f}",
                        "Vol": f"{feats['volatility']:.3f}",
                        "Sent": f"{feats['sentiment']:.2f}"
                    })
        else:
            st.info("No se encontraron acciones similares en el conjunto actual.")

