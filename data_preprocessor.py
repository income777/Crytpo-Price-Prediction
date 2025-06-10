import requests
import pandas as pd
from datetime import datetime, timedelta

# In-memory cache
_cache = {}

def fetch_coin_data(coin_id):
    now = datetime.utcnow()
    cache_key = coin_id
    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if now - ts < timedelta(minutes=1):
            return df  # use cached data

    # Fetch last 90 days daily history
    url = f"https://api.coincap.io/v2/assets/{coin_id}/history"
    params = {
        "interval": "d1",
        "start": int((now - timedelta(days=90)).timestamp() * 1000),
        "end":   int(now.timestamp() * 1000)
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise ValueError(f"CoinCap API error ({resp.status_code}): {resp.text}")

    data = resp.json().get("data")
    if not data:
        raise ValueError(f"No price history for coin_id '{coin_id}'")

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'priceUsd']].rename(columns={'date':'timestamp','priceUsd':'price'})
    df['price'] = df['price'].astype(float)
    df = df.set_index('timestamp')

    _cache[cache_key] = (now, df)
    return df
