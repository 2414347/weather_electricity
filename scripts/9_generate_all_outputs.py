# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries required for:
#
# os:
#   Used to create folders and check file existence.
#
# pandas:
#   Used to load model result files and manipulate tables.
#
# matplotlib:
#   Used to generate comparison and prediction plots.

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
# This section ensures that required output folders exist.
#
# data/figures:
#   Stores all generated visualizations.
#
# data/final_outputs:
#   Stores final summary tables and prediction results.

os.makedirs("data/figures", exist_ok=True)

os.makedirs("data/final_outputs", exist_ok=True)


print("Loading model result files...")


# ============================================================
# LOAD MODEL RESULT FILES
# ============================================================
# This section loads evaluation results from:
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
# COMBINE ALL MODEL RESULTS
# ============================================================
# This section merges all model results into
# a single dataset.
#
# Only Test dataset results are selected
# because they represent final model performance.

all_results = pd.concat(

    [baseline, rf, xgb, lstm],

    ignore_index=True

)

test_results = all_results[

    all_results["Dataset"] == "Test"

].copy()


# Sort models based on MAE
test_results = test_results.sort_values(

    by="MAE"

)


# ============================================================
# SAVE FINAL MODEL COMPARISON TABLE
# ============================================================
# This table contains:
#
# Model name
# Test MAE
# Test RMSE
#
# This file is typically included
# in dissertation results section.

test_results.to_csv(

    "data/final_outputs/final_model_comparison.csv",

    index=False

)


# ============================================================
# 1️⃣ TEST MAE BAR CHART
# ============================================================
# This plot compares MAE values
# across all models.
#
# Lower MAE indicates better accuracy.

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

plt.savefig(

    "data/figures/test_mae_comparison.png"

)

plt.close()


# ============================================================
# 2️⃣ TEST RMSE BAR CHART
# ============================================================
# This plot compares RMSE values
# across all models.

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

plt.savefig(

    "data/figures/test_rmse_comparison.png"

)

plt.close()


# ============================================================
# 3️⃣ MAE IMPROVEMENT OVER NAÏVE MODEL
# ============================================================
# This section calculates percentage
# improvement compared to the Naïve model.
#
# Formula:
#
# Improvement (%) =
# ((Naive MAE - Model MAE) / Naive MAE) × 100

naive_mae = test_results[

    test_results["Model"] == "Naive"

]["MAE"].values[0]


test_results["MAE_Improvement_%"] = (

    (naive_mae - test_results["MAE"])

    / naive_mae

    * 100

)


# Create improvement bar chart
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

plt.savefig(

    "data/figures/mae_improvement_percentage.png"

)

plt.close()


# ============================================================
# 4️⃣ ACTUAL VS PREDICTED (LSTM MULTIVARIATE)
# ============================================================
# This section visualizes how well
# the final LSTM model predicts
# real electricity demand values.

lstm_predictions_path = "data/results/lstm_test_predictions.csv"


# Check if predictions file exists
if os.path.exists(lstm_predictions_path):

    df_pred = pd.read_csv(

        lstm_predictions_path

    )


    # --------------------------------------------------------
    # SAVE PREDICTION COPY
    # --------------------------------------------------------
    # This ensures predictions are archived.

    df_pred.to_csv(

        "data/final_outputs/lstm_test_predictions_saved.csv",

        index=False

    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED PLOT
    # --------------------------------------------------------
    # Shows how close predictions are
    # to actual values.

    plt.figure()

    plt.plot(df_pred["Actual"])

    plt.plot(df_pred["Predicted"])

    plt.title(

        "LSTM Multivariate - Actual vs Predicted (Test)"

    )

    plt.ylabel("Daily Demand")

    plt.xlabel("Time Step")

    plt.tight_layout()

    plt.savefig(

        "data/figures/lstm_actual_vs_predicted.png"

    )

    plt.close()


    # --------------------------------------------------------
    # RESIDUAL CALCULATION
    # --------------------------------------------------------
    # Residual = Actual - Predicted

    df_pred["Residual"] = (

        df_pred["Actual"]

        - df_pred["Predicted"]

    )


    # --------------------------------------------------------
    # RESIDUAL TIME-SERIES PLOT
    # --------------------------------------------------------

    plt.figure()

    plt.plot(df_pred["Residual"])

    plt.title(

        "LSTM Residual Plot (Test)"

    )

    plt.ylabel("Residual")

    plt.xlabel("Time Step")

    plt.tight_layout()

    plt.savefig(

        "data/figures/lstm_residual_plot.png"

    )

    plt.close()


    # --------------------------------------------------------
    # RESIDUAL DISTRIBUTION
    # --------------------------------------------------------
    # Shows error distribution pattern.

    plt.figure()

    plt.hist(

        df_pred["Residual"],

        bins=30

    )

    plt.title(

        "Residual Distribution (LSTM Test)"

    )

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        "data/figures/lstm_residual_distribution.png"

    )

    plt.close()


else:

    print(
        "⚠ LSTM predictions file not found. Skipping prediction plots."
    )


# ============================================================
# FINAL STATUS MESSAGE
# ============================================================

print("\nAll outputs generated successfully.")

print("Figures saved in: data/figures/")

print("Final tables saved in: data/final_outputs/")
