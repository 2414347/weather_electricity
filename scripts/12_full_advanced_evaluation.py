import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# CONFIGURATION
# ==============================

DATA_FILE = "data/processed/model_ready_daily_dataset.csv"
MODEL_FILE = "models/xgb_multivariate.pkl"
FEATURE_FILE = "models/feature_list.pkl"

os.makedirs("data/advanced_figures", exist_ok=True)
os.makedirs("data/advanced_outputs", exist_ok=True)

print("Loading dataset and model...")

df = pd.read_csv(DATA_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

target = "daily_demand"

# ==============================
# TEST SET (2022)
# ==============================

test = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] <= "2022-12-31")].copy()

X_test = test[features]
y_test = test[target]

# ==============================
# PREDICTION
# ==============================

start = time.time()
y_pred = model.predict(X_test)
end = time.time()

prediction_time = end - start

# Save predictions
predictions_df = pd.DataFrame({
    "DATE": test["DATE"],
    "Actual": y_test,
    "Predicted": y_pred
})
predictions_df.to_csv("data/advanced_outputs/xgb_test_predictions.csv", index=False)

# ==============================
# METRICS
# ==============================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

metrics_df = pd.DataFrame({
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "MAPE_%": [mape],
    "Prediction_Time_seconds": [prediction_time]
})

metrics_df.to_csv("data/advanced_outputs/xgb_advanced_metrics.csv", index=False)

print("\nAdvanced Metrics:")
print(metrics_df)

# ==============================
# ACTUAL VS PREDICTED
# ==============================

plt.figure()
plt.plot(y_test.values)
plt.plot(y_pred)
plt.title("Actual vs Predicted (XGBoost Multivariate)")
plt.ylabel("Daily Demand")
plt.xlabel("Time Step")
plt.tight_layout()
plt.savefig("data/advanced_figures/actual_vs_predicted.png", dpi=300)
plt.close()

# ==============================
# RESIDUAL PLOT
# ==============================

residuals = y_test.values - y_pred

plt.figure()
plt.plot(residuals)
plt.title("Residual Plot")
plt.ylabel("Residual")
plt.xlabel("Time Step")
plt.tight_layout()
plt.savefig("data/advanced_figures/residual_plot.png", dpi=300)
plt.close()

# ==============================
# RESIDUAL DISTRIBUTION
# ==============================

plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/advanced_figures/residual_distribution.png", dpi=300)
plt.close()

# ==============================
# FEATURE IMPORTANCE
# ==============================

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("data/advanced_outputs/feature_importance.csv", index=False)

plt.figure()
plt.bar(importance_df["Feature"].head(15),
        importance_df["Importance"].head(15))
plt.xticks(rotation=90)
plt.title("Top 15 Feature Importance")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.tight_layout()
plt.savefig("data/advanced_figures/feature_importance.png", dpi=300)
plt.close()

# ==============================
# SEASONAL PERFORMANCE
# ==============================

test["Month"] = test["DATE"].dt.month

def map_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Autumn"

test["Season"] = test["Month"].apply(map_season)

season_results = []

for season in ["Winter","Spring","Summer","Autumn"]:
    mask = test["Season"] == season
    season_mae = mean_absolute_error(
        y_test[mask], y_pred[mask]
    )
    season_results.append([season, season_mae])

season_df = pd.DataFrame(season_results, columns=["Season","MAE"])
season_df.to_csv("data/advanced_outputs/seasonal_mae.csv", index=False)

plt.figure()
plt.bar(season_df["Season"], season_df["MAE"])
plt.title("Seasonal MAE (2022 Test Set)")
plt.ylabel("MAE")
plt.xlabel("Season")
plt.tight_layout()
plt.savefig("data/advanced_figures/seasonal_mae.png", dpi=300)
plt.close()

print("\nAll advanced evaluation outputs generated successfully.")
print("Check folders:")
print("data/advanced_figures/")
print("data/advanced_outputs/")