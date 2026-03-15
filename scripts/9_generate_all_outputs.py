import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
os.makedirs("data/figures", exist_ok=True)
os.makedirs("data/final_outputs", exist_ok=True)

print("Loading model result files...")

baseline = pd.read_csv("data/results/baseline_results.csv")
rf = pd.read_csv("data/results/random_forest_results.csv")
xgb = pd.read_csv("data/results/xgboost_results.csv")
lstm = pd.read_csv("data/results/lstm_results.csv")

# ==============================
# COMBINE ALL RESULTS
# ==============================
all_results = pd.concat([baseline, rf, xgb, lstm], ignore_index=True)

test_results = all_results[all_results["Dataset"] == "Test"].copy()
test_results = test_results.sort_values(by="MAE")

# Save final comparison table
test_results.to_csv("data/final_outputs/final_model_comparison.csv", index=False)

# ==============================
# 1️⃣ TEST MAE BAR CHART
# ==============================
plt.figure()
plt.bar(test_results["Model"], test_results["MAE"])
plt.xticks(rotation=45)
plt.title("Test MAE Comparison Across Models")
plt.ylabel("MAE")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("data/figures/test_mae_comparison.png")
plt.close()

# ==============================
# 2️⃣ TEST RMSE BAR CHART
# ==============================
plt.figure()
plt.bar(test_results["Model"], test_results["RMSE"])
plt.xticks(rotation=45)
plt.title("Test RMSE Comparison Across Models")
plt.ylabel("RMSE")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("data/figures/test_rmse_comparison.png")
plt.close()

# ==============================
# 3️⃣ MAE IMPROVEMENT OVER NAIVE
# ==============================
naive_mae = test_results[test_results["Model"] == "Naive"]["MAE"].values[0]

test_results["MAE_Improvement_%"] = (
    (naive_mae - test_results["MAE"]) / naive_mae * 100
)

plt.figure()
plt.bar(test_results["Model"], test_results["MAE_Improvement_%"])
plt.xticks(rotation=45)
plt.title("Percentage MAE Improvement Over Naive")
plt.ylabel("Improvement (%)")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("data/figures/mae_improvement_percentage.png")
plt.close()

# ==============================
# 4️⃣ ACTUAL VS PREDICTED (LSTM MULTIVARIATE)
# ==============================
lstm_predictions_path = "data/results/lstm_test_predictions.csv"

if os.path.exists(lstm_predictions_path):

    df_pred = pd.read_csv(lstm_predictions_path)

    # Save copy
    df_pred.to_csv("data/final_outputs/lstm_test_predictions_saved.csv", index=False)

    # Actual vs Predicted Plot
    plt.figure()
    plt.plot(df_pred["Actual"])
    plt.plot(df_pred["Predicted"])
    plt.title("LSTM Multivariate - Actual vs Predicted (Test)")
    plt.ylabel("Daily Demand")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("data/figures/lstm_actual_vs_predicted.png")
    plt.close()

    # Residual Plot
    df_pred["Residual"] = df_pred["Actual"] - df_pred["Predicted"]

    plt.figure()
    plt.plot(df_pred["Residual"])
    plt.title("LSTM Residual Plot (Test)")
    plt.ylabel("Residual")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("data/figures/lstm_residual_plot.png")
    plt.close()

    # Residual Distribution
    plt.figure()
    plt.hist(df_pred["Residual"], bins=30)
    plt.title("Residual Distribution (LSTM Test)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("data/figures/lstm_residual_distribution.png")
    plt.close()

else:
    print("⚠ LSTM predictions file not found. Skipping prediction plots.")

print("\nAll outputs generated successfully.")
print("Figures saved in: data/figures/")
print("Final tables saved in: data/final_outputs/")