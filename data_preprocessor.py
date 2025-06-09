
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_and_prepare_data(coin_id, horizon_days, window_size=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "90",
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.text}")

    data = response.json().get("prices", [])
    df = pd.DataFrame(data, columns=["timestamp", "price"])
    df["price"] = df["price"].astype(float)
    df = df[["price"]]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data) - horizon_days + 1):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i + horizon_days - 1, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, df["price"].values[-1]
