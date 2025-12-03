from pybackend.services.sentiment import fetch_headlines, sentiment_score
import sys

try:
    print("Fetching headlines for AAPL...")
    headlines = fetch_headlines("AAPL")
    print(f"Headlines found: {len(headlines)}")
    for h in headlines[:3]:
        print(f"- {h}")
    
    score = sentiment_score(headlines)
    print(f"Sentiment Score: {score}")
except Exception as e:
    print(f"Error: {e}")
