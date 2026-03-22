from io import StringIO

import httpx
import pandas as pd
import yfinance as yf

print("Testing VIX and TNX:")
try:
    vix = yf.Ticker("^VIX").fast_info["lastPrice"]
    tnx = yf.Ticker("^TNX").fast_info["lastPrice"]
    print(f"VIX: {vix}")
    print(f"TNX: {tnx}")
except Exception as e:
    print(f"Error yfinance: {e}")

print("\nTesting CPI from FRED:")
try:
    response = httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", follow_redirects=True)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    date_column = "DATE" if "DATE" in df.columns else "observation_date" if "observation_date" in df.columns else None
    if date_column is not None and "CPIAUCSL" in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        df["CPIAUCSL"] = pd.to_numeric(df["CPIAUCSL"], errors="coerce")
        df = df.dropna(subset=[date_column, "CPIAUCSL"]).sort_values(date_column)
    print(df.tail())
except Exception as e:
    print(f"Error fetching CPI: {e}")
