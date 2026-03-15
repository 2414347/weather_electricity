import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/baseline_results.csv"

print("Loading feature-engineered dataset...")
df = pd.read_csv(INPUT_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

# ==============================
# 1️⃣ CHRONOLOGICAL SPLIT
# ==============================

train = df[(df["DATE"] >= "2009-01-01") & (df["DATE"] <= "2019-12-31")]
val = df[(df["DATE"] >= "2020-01-01") & (df["DATE"] <= "2021-12-31")]
test = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] <= "2022-12-31")]

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

# ==============================
# 2️⃣ NAÏVE FORECAST
# ==============================

def naive_forecast(data):
    return data["lag_1"]

# ==============================
# 3️⃣ SEASONAL NAÏVE FORECAST
# ==============================

def seasonal_naive_forecast(data):
    return data["lag_7"]

# ==============================
# 4️⃣ EVALUATION FUNCTION
# ==============================

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

results = []

# ------------------------------
# Validation Evaluation
# ------------------------------

val_naive_pred = naive_forecast(val)
val_seasonal_pred = seasonal_naive_forecast(val)

mae_n_val, rmse_n_val = evaluate(val["daily_demand"], val_naive_pred)
mae_s_val, rmse_s_val = evaluate(val["daily_demand"], val_seasonal_pred)

results.append(["Naive", "Validation", mae_n_val, rmse_n_val])
results.append(["Seasonal_Naive", "Validation", mae_s_val, rmse_s_val])

# ------------------------------
# Test Evaluation
# ------------------------------

test_naive_pred = naive_forecast(test)
test_seasonal_pred = seasonal_naive_forecast(test)

mae_n_test, rmse_n_test = evaluate(test["daily_demand"], test_naive_pred)
mae_s_test, rmse_s_test = evaluate(test["daily_demand"], test_seasonal_pred)

results.append(["Naive", "Test", mae_n_test, rmse_n_test])
results.append(["Seasonal_Naive", "Test", mae_s_test, rmse_s_test])

# ==============================
# SAVE RESULTS
# ==============================

results_df = pd.DataFrame(results, columns=["Model", "Dataset", "MAE", "RMSE"])

os.makedirs("data/results", exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nBaseline Results:")
print(results_df)
print("\nSaved to:", OUTPUT_FILE)