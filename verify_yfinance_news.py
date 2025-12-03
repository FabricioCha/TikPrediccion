import yfinance as yf
import json

try:
    print("Fetching news from yfinance for AAPL...")
    tick = yf.Ticker("AAPL")
    news = tick.news
    if news:
        print(json.dumps(news[0], indent=2))
    else:
        print("No news found.")
except Exception as e:
    print(f"Error: {e}")
