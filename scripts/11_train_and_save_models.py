import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
LOOKBACK = 30

os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

target = "daily_demand"

# Chronological split (same as before)
train = df[(df["DATE"] >= "2009-01-01") & (df["DATE"] <= "2021-12-31")]
test = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] <= "2022-12-31")]

# ==============================
# FEATURE SET (BEST VERSION)
# ==============================
features = [col for col in df.columns if col not in ["DATE", target]]

# Save feature list
joblib.dump(features, "models/feature_list.pkl")

# ==========================================================
# 1️⃣ XGBOOST MULTIVARIATE (BEST TREE MODEL)
# ==========================================================
print("Training XGBoost Multivariate...")

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(train[features], train[target])

joblib.dump(xgb_model, "models/xgb_multivariate.pkl")
print("Saved: models/xgb_multivariate.pkl")

# ==========================================================
# 2️⃣ LSTM MULTIVARIATE (BEST DL MODEL)
# ==========================================================
print("Training LSTM Multivariate...")

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(train[features])
target_scaler.fit(train[[target]])

# Save scalers
joblib.dump(feature_scaler, "models/lstm_feature_scaler.pkl")
joblib.dump(target_scaler, "models/lstm_target_scaler.pkl")

train_scaled = train.copy()
train_scaled[features] = feature_scaler.transform(train[features])
train_scaled[target] = target_scaler.transform(train[[target]])

# Create sequences
def create_sequences(data, features):
    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[features].iloc[i-LOOKBACK:i].values)
        y.append(data[target].iloc[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, features)

# Build model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

lstm_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

lstm_model.save("models/lstm_multivariate.keras")
print("Saved: models/lstm_multivariate.keras")

print("\nAll best models saved successfully.")