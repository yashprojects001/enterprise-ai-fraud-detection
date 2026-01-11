import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ------------------ Config ------------------
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="centered")

MODEL_PATH = Path("src/models/saved_models/xgboost_model.pkl")
HISTORY_PATH = Path("dashboards/prediction_history.csv")

model = joblib.load(MODEL_PATH)

# ------------------ UI ------------------
st.title("💳 AI Fraud Detection – Decision Dashboard")
st.caption("Enterprise-style ML decision support system")

st.divider()

st.subheader("🔢 Enter Transaction Features")
st.caption("Time + V1..V28 + Amount_scaled (30 values)")

features_input = st.text_area(
    "Paste 30 comma-separated values",
    height=120,
    placeholder="0.0, -1.35, 1.19, ... , 0.02"
)

predict_btn = st.button("🔍 Predict Fraud Risk")

# ------------------ Prediction ------------------
if predict_btn:
    try:
        features = [float(x.strip()) for x in features_input.split(",")]

        if len(features) != model.n_features_in_:
            st.error(f"Expected {model.n_features_in_} features, got {len(features)}")
        else:
            X = np.array(features).reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            decision = "FRAUD" if prob >= 0.5 else "LEGIT"

            st.success("Prediction Completed")
            st.metric("Fraud Probability", round(float(prob), 4))
            st.metric("Decision", decision)

            # Save to history
            record = {
                "timestamp": datetime.now(),
                "fraud_probability": round(float(prob), 4),
                "decision": decision
            }

            if HISTORY_PATH.exists():
                df_hist = pd.read_csv(HISTORY_PATH)
                df_hist = pd.concat([df_hist, pd.DataFrame([record])], ignore_index=True)
            else:
                df_hist = pd.DataFrame([record])

            df_hist.to_csv(HISTORY_PATH, index=False)

    except Exception as e:
        st.error(f"Invalid input: {e}")

# ------------------ History ------------------
st.divider()
st.subheader("📊 Prediction History")

if HISTORY_PATH.exists():
    history_df = pd.read_csv(HISTORY_PATH)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions yet.")

# ------------------ Feedback ------------------
st.divider()
st.subheader("🧠 Human Feedback (Simulation)")

feedback = st.radio("Was the model decision correct?", ["Yes", "No"])

if st.button("Submit Feedback"):
    st.success("Feedback recorded (used in retraining phase conceptually)")
