import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("data/figures", exist_ok=True)

# Load results
baseline = pd.read_csv("data/results/baseline_results.csv")
rf = pd.read_csv("data/results/random_forest_results.csv")
xgb = pd.read_csv("data/results/xgboost_results.csv")
lstm = pd.read_csv("data/results/lstm_results.csv")

all_results = pd.concat([baseline, rf, xgb, lstm], ignore_index=True)

# ==============================
# Filter Test Results
# ==============================
test_results = all_results[all_results["Dataset"] == "Test"].copy()
test_results = test_results.sort_values(by="MAE")

# ==============================
# 1️⃣ Test MAE Comparison
# ==============================
plt.figure()
plt.bar(test_results["Model"], test_results["MAE"])
plt.xticks(rotation=45)
plt.title("Test MAE Comparison Across Models")
plt.ylabel("MAE")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("data/figures/test_mae_comparison.png")
plt.show()

# ==============================
# 2️⃣ Test RMSE Comparison
# ==============================
plt.figure()
plt.bar(test_results["Model"], test_results["RMSE"])
plt.xticks(rotation=45)
plt.title("Test RMSE Comparison Across Models")
plt.ylabel("RMSE")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("data/figures/test_rmse_comparison.png")
plt.show()

# ==============================
# 3️⃣ Percentage Improvement Over Naive
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
plt.show()

print(test_results[["Model", "MAE", "RMSE", "MAE_Improvement_%"]])