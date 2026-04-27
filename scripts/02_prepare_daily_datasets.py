# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries required for:
#
# os:
#   Used to create directories and manage file paths.
#
# pandas:
#   Used for reading datasets, cleaning data,
#   performing aggregation, and merging datasets.

import os
import pandas as pd


# ============================================================
# CONFIGURATION SETTINGS
# ============================================================
# This section defines input file paths, output file location,
# and time period filtering parameters.
#
# ELECTRICITY_FILE:
#   Raw electricity demand dataset containing
#   half-hourly demand values.
#
# WEATHER_FILE:
#   Raw weather dataset containing daily
#   temperature measurements.
#
# OUTPUT_FILE:
#   Final merged dataset saved after combining
#   electricity and weather data.
#
# START_DATE and END_DATE:
#   Define overlapping time period used
#   for both datasets.

ELECTRICITY_FILE = "data/raw_electricity/historic_demand_2009_2024_noNaN.csv"
WEATHER_FILE = "data/raw_weather/4238094.csv"
OUTPUT_FILE = "data/processed/merged_daily_dataset_2009_2022.csv"

START_DATE = "2009-01-01"
END_DATE = "2022-12-31"


# ============================================================
# 1️⃣ LOAD AND PROCESS ELECTRICITY DATA
# ============================================================
# This section loads electricity demand data
# and converts half-hourly demand into
# daily total demand.
#
# Steps performed:
#
# 1. Load raw electricity dataset.
# 2. Convert settlement_date to datetime.
# 3. Filter data within required time range.
# 4. Aggregate half-hourly demand into daily totals.
# 5. Rename columns to standardized names.

print("Loading electricity dataset...")

# Load electricity dataset
elec = pd.read_csv(ELECTRICITY_FILE)

# Convert settlement_date column to datetime format
elec["settlement_date"] = pd.to_datetime(elec["settlement_date"])

# ------------------------------------------------------------
# FILTER DATA BY DATE RANGE
# ------------------------------------------------------------
# Keeps only data between 2009 and 2022
# to match available weather data period.

elec = elec[
    (elec["settlement_date"] >= START_DATE) &
    (elec["settlement_date"] <= END_DATE)
]

print("Electricity filtered shape:", elec.shape)


# ------------------------------------------------------------
# AGGREGATE HALF-HOURLY DATA TO DAILY TOTAL
# ------------------------------------------------------------
# Electricity demand is originally recorded
# every 30 minutes.
#
# This step:
# Groups data by date
# Sums demand values for each day.

daily_elec = (
    elec
    .groupby(
        elec["settlement_date"].dt.date
    )["england_wales_demand"]
    .sum()
    .reset_index()
)

# Rename columns to standard names
daily_elec.rename(columns={

    "settlement_date": "DATE",
    "england_wales_demand": "daily_demand"

}, inplace=True)

# Convert DATE column to datetime
daily_elec["DATE"] = pd.to_datetime(daily_elec["DATE"])

print("Daily electricity shape:", daily_elec.shape)


# ============================================================
# 2️⃣ LOAD AND PROCESS WEATHER DATA
# ============================================================
# This section processes weather data
# and calculates national daily temperature.
#
# Steps performed:
#
# 1. Load weather dataset.
# 2. Convert DATE to datetime.
# 3. Filter overlapping date range.
# 4. Compute average daily temperature.
# 5. Remove missing values.
# 6. Aggregate temperature across stations.

print("\nLoading weather dataset...")

# Load weather dataset
weather = pd.read_csv(WEATHER_FILE)

# Convert DATE column to datetime
weather["DATE"] = pd.to_datetime(weather["DATE"])


# ------------------------------------------------------------
# FILTER DATE RANGE
# ------------------------------------------------------------
# Keeps weather data within same
# electricity dataset period.

weather = weather[
    (weather["DATE"] >= START_DATE) &
    (weather["DATE"] <= END_DATE)
]

print("Weather filtered shape:", weather.shape)


# ------------------------------------------------------------
# COMPUTE DAILY AVERAGE TEMPERATURE
# ------------------------------------------------------------
# Temperature average is calculated
# using maximum and minimum temperature.

weather["TEMP_AVG"] = (
    weather["TMAX"] + weather["TMIN"]
) / 2


# ------------------------------------------------------------
# REMOVE INVALID TEMPERATURE ROWS
# ------------------------------------------------------------
# Drops rows where both TMAX and TMIN
# were missing.

weather = weather.dropna(
    subset=["TEMP_AVG"]
)


# ------------------------------------------------------------
# COMPUTE NATIONAL DAILY TEMPERATURE
# ------------------------------------------------------------
# If multiple stations exist,
# this step calculates daily
# national average temperature.

daily_weather = (
    weather
    .groupby("DATE")["TEMP_AVG"]
    .mean()
    .reset_index()
)

# Rename column
daily_weather.rename(
    columns={
        "TEMP_AVG": "national_temp_avg"
    },
    inplace=True
)

print("Daily weather shape:", daily_weather.shape)


# ============================================================
# 3️⃣ MERGE ELECTRICITY AND WEATHER DATA
# ============================================================
# This section merges daily electricity demand
# with daily weather temperature.
#
# Merge type:
# "inner"
#
# Meaning:
# Only dates present in BOTH datasets
# will be included.

print("\nMerging datasets...")

merged = pd.merge(

    daily_elec,
    daily_weather,

    on="DATE",

    how="inner"

)

print("Merged dataset shape:", merged.shape)


# ============================================================
# 4️⃣ SAVE FINAL MERGED DATASET
# ============================================================
# This section saves processed dataset
# into the "processed" folder.
#
# Steps:
#
# 1. Create processed folder if it doesn't exist.
# 2. Save merged dataset to CSV.
# 3. Display preview of data.

# Create output directory if not exists
os.makedirs("data/processed", exist_ok=True)

# Save dataset
merged.to_csv(

    OUTPUT_FILE,
    index=False

)

print("\nFinal dataset saved to:", OUTPUT_FILE)

# Display preview of first few rows
print("\nPreview:")

print(merged.head())
