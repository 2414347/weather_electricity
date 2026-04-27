# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports all necessary libraries required
# to train and evaluate the XGBoost machine learning model.
#
# os:
#   Used to create directories and manage file paths.
#
# pandas:
#   Used to load and manipulate datasets.
#
# numpy:
#   Used for numerical calculations.
#
# XGBRegressor:
#   XGBoost regression model used for high-performance
#   gradient boosting predictions.
#
# sklearn.metrics:
#   Used to evaluate prediction accuracy
#   using MAE and RMSE.

import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# DEFINE INPUT AND OUTPUT FILE PATHS
# ============================================================
# INPUT_FILE:
#   Feature-engineered dataset used for training
#   machine learning models.
#
# OUTPUT_FILE:
#   Stores performance results of XGBoost models.

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/xgboost_results.csv"


# ============================================================
# LOAD DATASET
# ============================================================
# This section loads the feature-engineered dataset
# and converts DATE column into datetime format
# to support chronological time-based splitting.

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# Convert DATE column into datetime format
df["DATE"] = pd.to_datetime(df["DATE"])


# ============================================================
# 1️⃣ CHRONOLOGICAL DATA SPLIT
# ============================================================
# This section splits dataset into:
#
# Train Set (2009–2019):
#   Used to train XGBoost models.
#
# Validation Set (2020–2021):
#   Used to tune and evaluate model performance.
#
# Test Set (2022):
#   Used for final unbiased evaluation.
#
# Chronological splitting prevents
# data leakage from future values.

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


# Define prediction target
target = "daily_demand"


# ============================================================
# 2️⃣ DEFINE FEATURE SETS
# ============================================================
# This section defines two model configurations:
#
# Univariate Model:
#   Uses only historical demand-based features.
#
# Multivariate Model:
#   Uses all available features including:
#       - Weather
#       - Calendar
#       - Lag variables
#
# This allows comparison of feature impact.


# ------------------------------------------------------------
# UNIVARIATE FEATURES
# ------------------------------------------------------------
# Includes only electricity-based features.

univariate_features = [

    'lag_1',
    'lag_7',
    'lag_14',
    'lag_30',

    'roll_mean_7',
    'roll_std_7',

    'diff_1',
    'pct_change_7',

    'dow_sin',
    'dow_cos',

    'month_sin',
    'month_cos',

    'year'

]


# ------------------------------------------------------------
# MULTIVARIATE FEATURES
# ------------------------------------------------------------
# Uses all features except:
# DATE column
# Target column

multivariate_features = [

    col for col in df.columns

    if col not in ['DATE', 'daily_demand']

]


# ============================================================
# 3️⃣ MODEL TRAINING FUNCTION
# ============================================================
# This function trains and evaluates XGBoost models.
#
# Steps performed:
#
# 1. Initialize XGBoost model.
# 2. Train model using training dataset.
# 3. Generate predictions for validation data.
# 4. Generate predictions for test data.
# 5. Calculate evaluation metrics.
#
# Key Hyperparameters:
#
# n_estimators = 500
#   Number of boosting rounds.
#
# learning_rate = 0.05
#   Controls speed of learning.
#
# max_depth = 6
#   Controls tree complexity.
#
# subsample = 0.8
#   Prevents overfitting.
#
# colsample_bytree = 0.8
#   Random feature selection.
#
# random_state = 42
#   Ensures reproducibility.

def train_and_evaluate(features, model_name):

    # Initialize XGBoost model
    model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1

    )

    # Train model
    model.fit(

        train[features],

        train[target]

    )

    # Generate predictions
    val_pred = model.predict(val[features])

    test_pred = model.predict(test[features])


    # --------------------------------------------------------
    # CALCULATE PERFORMANCE METRICS
    # --------------------------------------------------------

    val_mae = mean_absolute_error(

        val[target],

        val_pred

    )

    val_rmse = np.sqrt(

        mean_squared_error(

            val[target],

            val_pred

        )

    )

    test_mae = mean_absolute_error(

        test[target],

        test_pred

    )

    test_rmse = np.sqrt(

        mean_squared_error(

            test[target],

            test_pred

        )

    )


    # Return performance results
    return [

        [model_name, "Validation", val_mae, val_rmse],

        [model_name, "Test", test_mae, test_rmse]

    ]


# Create empty results list
results = []


# ============================================================
# TRAIN UNIVARIATE XGBOOST MODEL
# ============================================================
# Uses electricity demand features only.

results.extend(

    train_and_evaluate(

        univariate_features,

        "XGB_Univariate"

    )

)


# ============================================================
# TRAIN MULTIVARIATE XGBOOST MODEL
# ============================================================
# Uses all available features including weather.

results.extend(

    train_and_evaluate(

        multivariate_features,

        "XGB_Multivariate"

    )

)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================
# This section converts results into a structured table
# and saves them into a CSV file.

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",
        "Dataset",
        "MAE",
        "RMSE"

    ]

)

# Create results folder if missing
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
print("\nXGBoost Results:")

print(results_df)

print("\nSaved to:", OUTPUT_FILE)
