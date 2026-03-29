import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="UK Electricity Demand Forecasting System",
    layout="wide"
)

# ============================================
# LOAD DATA
# ============================================
DATA_FILE = "data/processed/model_ready_daily_dataset.csv"

df = pd.read_csv(DATA_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

target = "daily_demand"
LOOKBACK = 30

# ============================================
# LOAD MODELS & OBJECTS
# ============================================
xgb_model = joblib.load("models/xgb_multivariate.pkl")
lstm_model = load_model("models/lstm_multivariate.keras")

features = joblib.load("models/feature_list.pkl")
feature_scaler = joblib.load("models/lstm_feature_scaler.pkl")
target_scaler = joblib.load("models/lstm_target_scaler.pkl")

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
st.sidebar.title("Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["XGBoost Multivariate", "LSTM Multivariate"]
)

selected_date = st.sidebar.date_input(
    "Select Forecast Date",
    value=df["DATE"].max(),
    min_value=df["DATE"].min(),
    max_value=df["DATE"].max()
)

# ============================================
# MAIN TITLE & DESCRIPTION
# ============================================
st.title("UK Daily Electricity Demand Forecasting System")

st.markdown("""
This application forecasts daily UK electricity demand using:
- Machine Learning (XGBoost)
- Deep Learning (LSTM)
- Engineered time-series features
- Historical weather integration

The system was developed following a strict chronological evaluation protocol.
""")

# ============================================
# PREDICTION FUNCTIONS
# ============================================
def predict_xgb(date):
    row = df[df["DATE"] == pd.to_datetime(date)]
    if row.empty:
        return None
    X = row[features]
    return float(xgb_model.predict(X)[0])

def predict_lstm(date):
    idx = df[df["DATE"] == pd.to_datetime(date)].index
    if len(idx) == 0 or idx[0] < LOOKBACK:
        return None

    idx = idx[0]
    sequence = df.iloc[idx-LOOKBACK:idx][features].copy()
    sequence[features] = feature_scaler.transform(sequence[features])
    X = np.array([sequence.values])
    pred_scaled = lstm_model.predict(X, verbose=0)[0][0]
    pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)

# ============================================
# MAKE PREDICTION
# ============================================
if model_choice == "XGBoost Multivariate":
    prediction = predict_xgb(selected_date)
else:
    prediction = predict_lstm(selected_date)

st.subheader("Forecast Result")

if prediction is None:
    st.warning("Not enough historical data available for selected date.")
else:
    st.metric(
        label="Predicted Daily Demand (MW)",
        value=f"{prediction:,.0f}"
    )

# ============================================
# DOWNLOAD LIVE FORECAST
# ============================================
st.subheader("Download Current Forecast")

if prediction is not None:
    single_pred_df = pd.DataFrame({
        "DATE": [selected_date],
        "Predicted_Demand": [prediction]
    })

    st.download_button(
        label=f"Download Forecast for {selected_date}",
        data=single_pred_df.to_csv(index=False),
        file_name=f"forecast_{selected_date}.csv",
        mime="text/csv"
    )

# ============================================
# MODEL PERFORMANCE SECTION
# ============================================
st.markdown("---")
st.header("Model Evaluation (2022 Test Set)")

metrics_df = pd.read_csv("data/advanced_outputs/xgb_advanced_metrics.csv")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{metrics_df['MAE'][0]:,.0f}")
col2.metric("RMSE", f"{metrics_df['RMSE'][0]:,.0f}")
col3.metric("R²", f"{metrics_df['R2'][0]:.4f}")
col4.metric("MAPE (%)", f"{metrics_df['MAPE_%'][0]:.2f}")

