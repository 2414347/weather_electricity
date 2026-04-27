# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports required libraries used for:
#
# os:
#   Used to create directories and manage file paths.
#
# pandas:
#   Used for data manipulation and feature creation.
#
# numpy:
#   Used for numerical calculations, especially
#   cyclical feature encoding using sine and cosine.

import os
import pandas as pd
import numpy as np


# ============================================================
# DEFINE INPUT AND OUTPUT FILE PATHS
# ============================================================
# INPUT_FILE:
#   Previously merged dataset containing:
#       - Daily electricity demand
#       - National average temperature
#
# OUTPUT_FILE:
#   Final dataset after feature engineering.
#   This dataset will be used for training
#   machine learning and deep learning models.

INPUT_FILE = "data/processed/merged_daily_dataset_2009_2022.csv"
OUTPUT_FILE = "data/processed/model_ready_daily_dataset.csv"


# ============================================================
# LOAD MERGED DATASET
# ============================================================
# This section loads the merged dataset and converts
# the DATE column into datetime format.
#
# This ensures that time-based feature extraction
# works correctly.

print("Loading merged dataset...")

df = pd.read_csv(INPUT_FILE)

# Convert DATE column to datetime
df["DATE"] = pd.to_datetime(df["DATE"])


# ============================================================
# 1️⃣ ELECTRICITY DEMAND FEATURES
# ============================================================
# This section creates time-series features
# derived from historical electricity demand.
#
# These features help the model learn:
# - Demand trends
# - Seasonal patterns
# - Short-term dependencies
#
# Feature types created:
# - Lag features
# - Rolling statistics
# - Momentum features


# ------------------------------------------------------------
# LAG FEATURES
# ------------------------------------------------------------
# Lag features represent past demand values.
#
# These are the most important features in
# electricity demand forecasting.

df["lag_1"] = df["daily_demand"].shift(1)
df["lag_7"] = df["daily_demand"].shift(7)
df["lag_14"] = df["daily_demand"].shift(14)
df["lag_30"] = df["daily_demand"].shift(30)

# Meaning:
# lag_1  → yesterday demand
# lag_7  → same day last week
# lag_14 → two weeks ago
# lag_30 → one month ago


# ------------------------------------------------------------
# ROLLING STATISTICS
# ------------------------------------------------------------
# Rolling statistics capture short-term trends
# and variability.

df["roll_mean_7"] = df["daily_demand"].rolling(7).mean()
df["roll_std_7"] = df["daily_demand"].rolling(7).std()

# Meaning:
# roll_mean_7 → 7-day moving average
# roll_std_7  → demand volatility


# ------------------------------------------------------------
# MOMENTUM FEATURES
# ------------------------------------------------------------
# Momentum features measure change
# in demand over time.

df["diff_1"] = df["daily_demand"].diff(1)

df["pct_change_7"] = df["daily_demand"].pct_change(7)

# Meaning:
# diff_1 → daily demand difference
# pct_change_7 → weekly demand change %


# ============================================================
# 2️⃣ WEATHER FEATURES
# ============================================================
# This section creates weather-derived features.
#
# Electricity demand strongly depends on
# heating and cooling needs.

# ------------------------------------------------------------
# HEATING & COOLING DEGREE DAYS
# ------------------------------------------------------------
# HDD and CDD measure energy demand due
# to temperature deviation from comfort level.

df["HDD"] = (18 - df["national_temp_avg"]).clip(lower=0)

df["CDD"] = (df["national_temp_avg"] - 18).clip(lower=0)

# Meaning:
# HDD → heating demand indicator
# CDD → cooling demand indicator


# ------------------------------------------------------------
# TEMPERATURE LAG FEATURES
# ------------------------------------------------------------
# Temperature from previous days
# can influence demand behavior.

df["temp_lag_1"] = df["national_temp_avg"].shift(1)

df["temp_lag_3"] = df["national_temp_avg"].shift(3)

# Meaning:
# temp_lag_1 → yesterday temperature
# temp_lag_3 → temperature 3 days ago


# ============================================================
# 3️⃣ CALENDAR FEATURES
# ============================================================
# This section extracts time-based calendar features.
#
# These help models learn seasonal patterns.

df["day_of_week"] = df["DATE"].dt.dayofweek

df["month"] = df["DATE"].dt.month

df["year"] = df["DATE"].dt.year


# ------------------------------------------------------------
# CYCLICAL ENCODING
# ------------------------------------------------------------
# Cyclical encoding converts time variables
# into continuous circular representation.
#
# This improves machine learning performance
# for seasonal variables.

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# These features represent weekly
# and monthly cycles smoothly.


# ============================================================
# 4️⃣ HANDLE MISSING VALUES FROM SHIFTING
# ============================================================
# Lag and rolling operations introduce
# missing values at the beginning of dataset.
#
# This step removes those rows.

df = df.dropna().reset_index(drop=True)


# ============================================================
# 5️⃣ SAVE FINAL FEATURE-ENGINEERED DATASET
# ============================================================
# This section saves the dataset after
# all feature engineering steps.
#
# Steps:
# 1. Create processed folder if missing.
# 2. Save dataset to CSV file.
# 3. Print dataset summary.

os.makedirs("data/processed", exist_ok=True)

df.to_csv(

    OUTPUT_FILE,
    index=False

)

print("Feature engineering complete.")

print("Final shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())
