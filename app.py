
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from data_preprocessor import fetch_and_prepare_data
from lstm_model import create_lstm_model
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict_price(coin_id: str = "solana", horizon_days: int = 1):
    try:
        X, y, scaler, current_price = fetch_and_prepare_data(coin_id, horizon_days)
    except Exception as e:
        return {"error": str(e), "coin_id": coin_id}

    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, verbose=0)

    last_input = X[-1].reshape(1, X.shape[1], 1)
    predicted_scaled = model.predict(last_input)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    return {
        "current_price": round(current_price, 4),
        "predicted_price": round(predicted_price, 4),
        "horizon_days": horizon_days,
        "coin_id": coin_id
    }
