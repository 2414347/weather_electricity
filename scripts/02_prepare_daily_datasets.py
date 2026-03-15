import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
ELECTRICITY_FILE = "data/raw_electricity/historic_demand_2009_2024_noNaN.csv"
WEATHER_FILE = "data/raw_weather/4238094.csv"
OUTPUT_FILE = "data/processed/merged_daily_dataset_2009_2022.csv"

START_DATE = "2009-01-01"
END_DATE = "2022-12-31"


# ==============================
# 1️⃣ LOAD ELECTRICITY
# ==============================
print("Loading electricity dataset...")
elec = pd.read_csv(ELECTRICITY_FILE)

# Convert datetime
elec["settlement_date"] = pd.to_datetime(elec["settlement_date"])

# Filter overlapping period
elec = elec[(elec["settlement_date"] >= START_DATE) &
            (elec["settlement_date"] <= END_DATE)]

print("Electricity filtered shape:", elec.shape)

# Aggregate to daily (sum of half-hourly demand)
daily_elec = (
    elec
    .groupby(elec["settlement_date"].dt.date)["england_wales_demand"]
    .sum()
    .reset_index()
)

daily_elec.rename(columns={
    "settlement_date": "DATE",
    "england_wales_demand": "daily_demand"
}, inplace=True)

daily_elec["DATE"] = pd.to_datetime(daily_elec["DATE"])

print("Daily electricity shape:", daily_elec.shape)


# ==============================
# 2️⃣ LOAD WEATHER
# ==============================
print("\nLoading weather dataset...")
weather = pd.read_csv(WEATHER_FILE)

weather["DATE"] = pd.to_datetime(weather["DATE"])

# Filter overlapping period
weather = weather[(weather["DATE"] >= START_DATE) &
                  (weather["DATE"] <= END_DATE)]

print("Weather filtered shape:", weather.shape)

# Compute station-level average temperature
weather["TEMP_AVG"] = (weather["TMAX"] + weather["TMIN"]) / 2

# Drop rows where both TMAX and TMIN were missing
weather = weather.dropna(subset=["TEMP_AVG"])

# Compute national daily average temperature
daily_weather = (
    weather
    .groupby("DATE")["TEMP_AVG"]
    .mean()
    .reset_index()
)

daily_weather.rename(columns={"TEMP_AVG": "national_temp_avg"}, inplace=True)

print("Daily weather shape:", daily_weather.shape)


# ==============================
# 3️⃣ MERGE DATASETS
# ==============================
print("\nMerging datasets...")

merged = pd.merge(daily_elec, daily_weather, on="DATE", how="inner")

print("Merged dataset shape:", merged.shape)

# ==============================
# 4️⃣ SAVE OUTPUT
# ==============================
os.makedirs("data/processed", exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)

print("\nFinal dataset saved to:", OUTPUT_FILE)
print("\nPreview:")
print(merged.head())