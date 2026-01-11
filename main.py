from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

# Load trained model
MODEL_PATH = Path("src/models/saved_models/xgboost_model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Enterprise AI Fraud Detection API",
    description="Real-time fraud risk scoring service",
    version="1.0"
)

# ----- Input schema -----
class Transaction(BaseModel):
    # Time + V1..V28 + Amount_scaled = 30 features
    features: list[float]


# ----- Prediction endpoint -----
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    X = np.array(transaction.features)

    # Safety check (prevents server crash)
    if X.shape[0] != model.n_features_in_:
        return {
            "error": f"Expected {model.n_features_in_} features, got {X.shape[0]}"
        }

    X = X.reshape(1, -1)

    probability = model.predict_proba(X)[0][1]
    decision = "FRAUD" if probability >= 0.5 else "LEGIT"

    return {
        "fraud_probability": round(float(probability), 4),
        "decision": decision
    }


# ----- Health check -----
@app.get("/")
def health():
    return {"status": "API is running"}
