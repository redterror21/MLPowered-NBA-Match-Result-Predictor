from fastapi import FastAPI
import numpy as np
import joblib

# -------------------------------------------------
# LOAD MODELS & SCALERS
# -------------------------------------------------

batch_model = joblib.load("model_batch.pkl")
batch_scaler = joblib.load("scaler_batch.pkl")

incremental_model = joblib.load("model_incremental.pkl")
incremental_scaler = joblib.load("scaler_incremental.pkl")

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(title="NBA Win Probability API")

# -------------------------------------------------
# FEATURE BUILDER
# -------------------------------------------------

def build_features(home_elo, away_elo, home_win_pct, away_win_pct):
    return np.array([[
        home_elo - away_elo,
        home_win_pct,
        away_win_pct
    ]])

# -------------------------------------------------
# BATCH MODEL ENDPOINT
# -------------------------------------------------

@app.post("/predict/batch")
def predict_batch(
    home_elo: float,
    away_elo: float,
    home_win_pct: float,
    away_win_pct: float
):
    X = build_features(home_elo, away_elo, home_win_pct, away_win_pct)
    X_scaled = batch_scaler.transform(X)

    prob = batch_model.predict_proba(X_scaled)[0, 1]

    return {
        "model": "batch_logistic_regression",
        "home_win_probability": round(float(prob), 4)
    }

# -------------------------------------------------
# INCREMENTAL MODEL ENDPOINT
# -------------------------------------------------

@app.post("/predict/incremental")
def predict_incremental(
    home_elo: float,
    away_elo: float,
    home_win_pct: float,
    away_win_pct: float
):
    X = build_features(home_elo, away_elo, home_win_pct, away_win_pct)
    X_scaled = incremental_scaler.transform(X)

    prob = incremental_model.predict_proba(X_scaled)[0, 1]

    return {
        "model": "incremental_sgd",
        "home_win_probability": round(float(prob), 4)
    }
