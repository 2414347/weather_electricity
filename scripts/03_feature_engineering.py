import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/merged_daily_dataset_2009_2022.csv"
OUTPUT_FILE = "data/processed/model_ready_daily_dataset.csv"

print("Loading merged dataset...")
df = pd.read_csv(INPUT_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

# ==============================
# 1️⃣ ELECTRICITY FEATURES
# ==============================

# Lags
df["lag_1"] = df["daily_demand"].shift(1)
df["lag_7"] = df["daily_demand"].shift(7)
df["lag_14"] = df["daily_demand"].shift(14)
df["lag_30"] = df["daily_demand"].shift(30)

# Rolling statistics
df["roll_mean_7"] = df["daily_demand"].rolling(7).mean()
df["roll_std_7"] = df["daily_demand"].rolling(7).std()

# Momentum
df["diff_1"] = df["daily_demand"].diff(1)
df["pct_change_7"] = df["daily_demand"].pct_change(7)

# ==============================
# 2️⃣ WEATHER FEATURES
# ==============================

# Heating & Cooling Degree Days
df["HDD"] = (18 - df["national_temp_avg"]).clip(lower=0)
df["CDD"] = (df["national_temp_avg"] - 18).clip(lower=0)

# Temperature lags
df["temp_lag_1"] = df["national_temp_avg"].shift(1)
df["temp_lag_3"] = df["national_temp_avg"].shift(3)

# ==============================
# 3️⃣ CALENDAR FEATURES
# ==============================

df["day_of_week"] = df["DATE"].dt.dayofweek
df["month"] = df["DATE"].dt.month
df["year"] = df["DATE"].dt.year

# Cyclical encoding
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ==============================
# 4️⃣ DROP NaNs FROM SHIFTING
# ==============================

df = df.dropna().reset_index(drop=True)

# ==============================
# 5️⃣ SAVE
# ==============================

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering complete.")
print("Final shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())