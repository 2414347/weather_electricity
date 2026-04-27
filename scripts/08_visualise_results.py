# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries used for:
#
# pandas:
#   Used to load model result files and combine them.
#
# matplotlib:
#   Used to create performance comparison graphs.
#
# os:
#   Used to create output directories if they do not exist.

import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# CREATE FIGURE OUTPUT DIRECTORY
# ============================================================
# This ensures that the folder for storing
# generated figures exists.
#
# If the folder does not exist, it will be created.
# All visualization plots will be saved here.

os.makedirs("data/figures", exist_ok=True)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================
# This section loads evaluation results
# from previously trained models.
#
# Models included:
#
# Baseline Models:
#   - Naïve
#   - Seasonal Naïve
#
# Machine Learning Models:
#   - Random Forest
#   - XGBoost
#
# Deep Learning Models:
#   - LSTM

baseline = pd.read_csv(
    "data/results/baseline_results.csv"
)

rf = pd.read_csv(
    "data/results/random_forest_results.csv"
)

xgb = pd.read_csv(
    "data/results/xgboost_results.csv"
)

lstm = pd.read_csv(
    "data/results/lstm_results.csv"
)


# ============================================================
# COMBINE ALL RESULTS
# ============================================================
# This section merges all model results
# into one dataset for easy comparison.

all_results = pd.concat(

    [baseline, rf, xgb, lstm],

    ignore_index=True

)


# ============================================================
# FILTER TEST DATA ONLY
# ============================================================
# This section extracts only Test dataset results.
#
# Test results represent final model performance
# on unseen data.

test_results = all_results[

    all_results["Dataset"] == "Test"

].copy()


# Sort models by MAE
test_results = test_results.sort_values(

    by="MAE"

)


# ============================================================
# 1️⃣ TEST MAE COMPARISON PLOT
# ============================================================
# This visualization compares Mean Absolute Error (MAE)
# across all models.
#
# Lower MAE indicates better performance.

plt.figure()

plt.bar(

    test_results["Model"],

    test_results["MAE"]

)

plt.xticks(rotation=45)

plt.title(

    "Test MAE Comparison Across Models"

)

plt.ylabel("MAE")

plt.xlabel("Model")

plt.tight_layout()


# Save figure
plt.savefig(

    "data/figures/test_mae_comparison.png"

)

# Display figure
plt.show()


# ============================================================
# 2️⃣ TEST RMSE COMPARISON PLOT
# ============================================================
# This visualization compares Root Mean Squared Error (RMSE)
# across models.
#
# RMSE penalizes large errors more strongly
# than MAE.

plt.figure()

plt.bar(

    test_results["Model"],

    test_results["RMSE"]

)

plt.xticks(rotation=45)

plt.title(

    "Test RMSE Comparison Across Models"

)

plt.ylabel("RMSE")

plt.xlabel("Model")

plt.tight_layout()


# Save figure
plt.savefig(

    "data/figures/test_rmse_comparison.png"

)

# Display figure
plt.show()


# ============================================================
# 3️⃣ PERCENTAGE IMPROVEMENT OVER NAÏVE MODEL
# ============================================================
# This section calculates how much each model
# improves over the Naïve baseline.
#
# Formula:
#
# Improvement (%) =
# ((Naive MAE - Model MAE) / Naive MAE) × 100

naive_mae = test_results[

    test_results["Model"] == "Naive"

]["MAE"].values[0]


# Calculate improvement
test_results["MAE_Improvement_%"] = (

    (naive_mae - test_results["MAE"])

    / naive_mae

    * 100

)


# ============================================================
# VISUALIZE IMPROVEMENT
# ============================================================
# This plot shows percentage improvement
# compared to Naïve baseline model.

plt.figure()

plt.bar(

    test_results["Model"],

    test_results["MAE_Improvement_%"]

)

plt.xticks(rotation=45)

plt.title(

    "Percentage MAE Improvement Over Naive"

)

plt.ylabel("Improvement (%)")

plt.xlabel("Model")

plt.tight_layout()


# Save figure
plt.savefig(

    "data/figures/mae_improvement_percentage.png"

)

# Display figure
plt.show()


# ============================================================
# PRINT FINAL RESULTS TABLE
# ============================================================
# This displays:
#
# Model name
# MAE
# RMSE
# Improvement %

print(

    test_results[

        ["Model",

         "MAE",

         "RMSE",

         "MAE_Improvement_%"]

    ]

)
