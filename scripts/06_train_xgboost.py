import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/xgboost_results.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

# ==============================
# 1️⃣ CHRONOLOGICAL SPLIT
# ==============================

train = df[(df["DATE"] >= "2009-01-01") & (df["DATE"] <= "2019-12-31")]
val = df[(df["DATE"] >= "2020-01-01") & (df["DATE"] <= "2021-12-31")]
test = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] <= "2022-12-31")]

target = "daily_demand"

# ==============================
# 2️⃣ FEATURE SETS
# ==============================

univariate_features = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'roll_mean_7', 'roll_std_7',
    'diff_1', 'pct_change_7',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'year'
]

multivariate_features = [col for col in df.columns 
                         if col not in ['DATE', 'daily_demand']]

# ==============================
# 3️⃣ TRAIN FUNCTION
# ==============================

def train_and_evaluate(features, model_name):

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], train[target])

    val_pred = model.predict(val[features])
    test_pred = model.predict(test[features])

    val_mae = mean_absolute_error(val[target], val_pred)
    val_rmse = np.sqrt(mean_squared_error(val[target], val_pred))

    test_mae = mean_absolute_error(test[target], test_pred)
    test_rmse = np.sqrt(mean_squared_error(test[target], test_pred))

    return [
        [model_name, "Validation", val_mae, val_rmse],
        [model_name, "Test", test_mae, test_rmse]
    ]

results = []

# ------------------------------
# Univariate Model
# ------------------------------
results.extend(train_and_evaluate(univariate_features, "XGB_Univariate"))

# ------------------------------
# Multivariate Model
# ------------------------------
results.extend(train_and_evaluate(multivariate_features, "XGB_Multivariate"))

# ==============================
# SAVE RESULTS
# ==============================

results_df = pd.DataFrame(results, columns=["Model", "Dataset", "MAE", "RMSE"])

os.makedirs("data/results", exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nXGBoost Results:")
print(results_df)
print("\nSaved to:", OUTPUT_FILE)