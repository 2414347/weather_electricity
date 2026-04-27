# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries needed for:
#
# os:
#   Used to create folders and manage file paths.
#
# pandas:
#   Used to load dataset and manage tabular data.
#
# numpy:
#   Used for numerical operations.
#
# sklearn.metrics:
#   Used to calculate model evaluation metrics
#   such as MAE and RMSE.

import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# DEFINE INPUT AND OUTPUT FILES
# ============================================================
# INPUT_FILE:
#   Feature-engineered dataset used for modelling.
#
# OUTPUT_FILE:
#   Stores evaluation results of baseline models.
#
# Baseline models are simple forecasting methods
# used as reference performance benchmarks.
# Machine learning models must outperform these.

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"

OUTPUT_FILE = "data/results/baseline_results.csv"


# ============================================================
# LOAD FEATURE-ENGINEERED DATASET
# ============================================================
# This section loads the dataset that contains:
#
# - Daily demand
# - Lag features
# - Weather features
# - Calendar features
#
# DATE column is converted into datetime
# for chronological splitting.

print("Loading feature-engineered dataset...")

df = pd.read_csv(INPUT_FILE)

# Convert DATE column into datetime format
df["DATE"] = pd.to_datetime(df["DATE"])


# ============================================================
# 1️⃣ CHRONOLOGICAL DATA SPLIT
# ============================================================
# This section splits dataset into:
#
# Train Set:
#   Used to train machine learning models.
#
# Validation Set:
#   Used to tune models.
#
# Test Set:
#   Used for final evaluation.
#
# IMPORTANT:
# Chronological splitting prevents data leakage
# in time-series forecasting.

train = df[
    (df["DATE"] >= "2009-01-01") &
    (df["DATE"] <= "2019-12-31")
]

val = df[
    (df["DATE"] >= "2020-01-01") &
    (df["DATE"] <= "2021-12-31")
]

test = df[
    (df["DATE"] >= "2022-01-01") &
    (df["DATE"] <= "2022-12-31")
]

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# ============================================================
# 2️⃣ NAÏVE FORECAST MODEL
# ============================================================
# Naïve Forecast:
# Uses yesterday's demand as today's prediction.
#
# Formula:
# Prediction(t) = Demand(t-1)
#
# This is the simplest forecasting method
# and serves as a baseline reference.

def naive_forecast(data):

    return data["lag_1"]


# ============================================================
# 3️⃣ SEASONAL NAÏVE FORECAST MODEL
# ============================================================
# Seasonal Naïve Forecast:
# Uses demand from the same day last week.
#
# Formula:
# Prediction(t) = Demand(t-7)
#
# This method captures weekly seasonality
# common in electricity demand.

def seasonal_naive_forecast(data):

    return data["lag_7"]


# ============================================================
# 4️⃣ MODEL EVALUATION FUNCTION
# ============================================================
# This function calculates performance metrics.
#
# Metrics used:
#
# MAE (Mean Absolute Error):
#   Average prediction error.
#
# RMSE (Root Mean Squared Error):
#   Penalizes large errors more heavily.

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return mae, rmse


# Create empty list to store results
results = []


# ============================================================
# VALIDATION SET EVALUATION
# ============================================================
# This section evaluates baseline models
# using validation dataset.

# Generate predictions
val_naive_pred = naive_forecast(val)

val_seasonal_pred = seasonal_naive_forecast(val)

# Calculate metrics
mae_n_val, rmse_n_val = evaluate(
    val["daily_demand"],
    val_naive_pred
)

mae_s_val, rmse_s_val = evaluate(
    val["daily_demand"],
    val_seasonal_pred
)

# Store results
results.append([
    "Naive",
    "Validation",
    mae_n_val,
    rmse_n_val
])

results.append([
    "Seasonal_Naive",
    "Validation",
    mae_s_val,
    rmse_s_val
])


# ============================================================
# TEST SET EVALUATION
# ============================================================
# This section evaluates baseline models
# on unseen test data.

# Generate predictions
test_naive_pred = naive_forecast(test)

test_seasonal_pred = seasonal_naive_forecast(test)

# Calculate metrics
mae_n_test, rmse_n_test = evaluate(
    test["daily_demand"],
    test_naive_pred
)

mae_s_test, rmse_s_test = evaluate(
    test["daily_demand"],
    test_seasonal_pred
)

# Store results
results.append([
    "Naive",
    "Test",
    mae_n_test,
    rmse_n_test
])

results.append([
    "Seasonal_Naive",
    "Test",
    mae_s_test,
    rmse_s_test
])


# ============================================================
# SAVE BASELINE RESULTS
# ============================================================
# This section saves evaluation results
# into a CSV file.
#
# Steps:
#
# 1. Convert results list into DataFrame.
# 2. Create results folder if missing.
# 3. Save results file.

results_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Dataset",
        "MAE",
        "RMSE"
    ]

)

# Create output folder
os.makedirs(
    "data/results",
    exist_ok=True
)

# Save results
results_df.to_csv(

    OUTPUT_FILE,

    index=False

)

# Display results
print("\nBaseline Results:")

print(results_df)

print("\nSaved to:", OUTPUT_FILE)
