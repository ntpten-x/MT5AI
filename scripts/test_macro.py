import yfinance as yf
import pandas_datareader.data as web
import datetime

print("Testing VIX and TNX:")
try:
    vix = yf.Ticker("^VIX").fast_info["lastPrice"]
    tnx = yf.Ticker("^TNX").fast_info["lastPrice"]
    print(f"VIX: {vix}")
    print(f"TNX: {tnx}")
except Exception as e:
    print(f"Error yfinance: {e}")

print("\nTesting CPI from FRED:")
start = datetime.datetime(2025, 1, 1)
end = datetime.datetime.now()
try:
    df = web.DataReader('CPIAUCSL', 'fred', start, end)
    print(df.tail())
except Exception as e:
    print(f"Error fetching CPI: {e}")
