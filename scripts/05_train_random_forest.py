# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries required for:
#
# os:
#   Used to create folders and manage file paths.
#
# pandas:
#   Used to load and manipulate datasets.
#
# numpy:
#   Used for numerical operations.
#
# RandomForestRegressor:
#   Machine learning model based on ensemble trees.
#
# sklearn.metrics:
#   Used to evaluate prediction performance
#   using MAE and RMSE.

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# DEFINE INPUT AND OUTPUT FILE PATHS
# ============================================================
# INPUT_FILE:
#   Feature-engineered dataset used for model training.
#
# OUTPUT_FILE:
#   Stores performance results of Random Forest models.

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/random_forest_results.csv"


# ============================================================
# LOAD DATASET
# ============================================================
# This section loads the prepared dataset
# and converts DATE column into datetime format
# to support chronological splitting.

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# Convert DATE column to datetime
df["DATE"] = pd.to_datetime(df["DATE"])


# ============================================================
# 1️⃣ CHRONOLOGICAL DATA SPLIT
# ============================================================
# This section divides data into:
#
# Train Set:
#   Used to train Random Forest models.
#
# Validation Set:
#   Used to evaluate model tuning.
#
# Test Set:
#   Used for final model performance testing.
#
# Chronological splitting prevents future
# data leakage into training.

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


# ============================================================
# 2️⃣ DEFINE FEATURE SETS
# ============================================================
# This section defines two types of models:
#
# Univariate Model:
#   Uses demand history features only.
#
# Multivariate Model:
#   Uses all available features including
#   weather and calendar data.
#
# This comparison helps measure whether
# weather variables improve performance.

target = "daily_demand"


# ------------------------------------------------------------
# UNIVARIATE FEATURES
# ------------------------------------------------------------
# Includes only electricity-based features
# such as lag values and seasonal patterns.

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
# Uses all available features except:
# DATE column
# Target column

multivariate_features = [

    col for col in df.columns

    if col not in ['DATE', 'daily_demand']

]


# ============================================================
# 3️⃣ MODEL TRAINING FUNCTION
# ============================================================
# This function trains and evaluates
# Random Forest models.
#
# Steps:
#
# 1. Initialize Random Forest model.
# 2. Train model using training data.
# 3. Generate validation predictions.
# 4. Generate test predictions.
# 5. Calculate performance metrics.
#
# Hyperparameters used:
#
# n_estimators = 300
#   Number of trees in forest.
#
# max_depth = None
#   Allows trees to grow fully.
#
# random_state = 42
#   Ensures reproducibility.
#
# n_jobs = -1
#   Uses all CPU cores.

def train_and_evaluate(features, model_name):

    # Initialize Random Forest model
    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

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


    # Return results
    return [

        [model_name, "Validation", val_mae, val_rmse],

        [model_name, "Test", test_mae, test_rmse]

    ]


# Create empty results list
results = []


# ============================================================
# TRAIN UNIVARIATE RANDOM FOREST
# ============================================================
# Uses only electricity demand features.

results.extend(

    train_and_evaluate(

        univariate_features,

        "RF_Univariate"

    )

)


# ============================================================
# TRAIN MULTIVARIATE RANDOM FOREST
# ============================================================
# Uses all features including weather.

results.extend(

    train_and_evaluate(

        multivariate_features,

        "RF_Multivariate"

    )

)


# ============================================================
# SAVE RESULTS
# ============================================================
# This section converts results into
# a structured table and saves them
# into a CSV file.

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Dataset",

        "MAE",

        "RMSE"

    ]

)

# Create results directory
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
print("\nRandom Forest Results:")

print(results_df)

print("\nSaved to:", OUTPUT_FILE)
