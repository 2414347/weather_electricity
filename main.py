# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports all necessary Python libraries required
# to build the Streamlit dashboard, load trained models,
# process data, and generate visualizations.

import streamlit as st                 # Used to build interactive web applications
import pandas as pd                   # Used for data loading and manipulation
import numpy as np                    # Used for numerical operations and arrays
import joblib                         # Used to load saved machine learning objects
from tensorflow.keras.models import load_model  # Used to load trained LSTM model
import matplotlib.pyplot as plt       # Used to generate plots and charts


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# This section configures the layout and appearance of the
# Streamlit web application.
#
# page_title:
#   Sets the browser tab title.
#
# layout="wide":
#   Expands the layout to use full screen width for better
#   visualization of charts and results.

st.set_page_config(
    page_title="UK Electricity Demand Forecasting System",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================
# This section loads the processed dataset that contains
# historical electricity demand and engineered features.
#
# Steps performed:
# 1. Load CSV dataset into a pandas DataFrame.
# 2. Convert DATE column into datetime format.
# 3. Define the target variable.
# 4. Set LOOKBACK window used by LSTM model.

DATA_FILE = "data/processed/model_ready_daily_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Convert DATE column into datetime format
df["DATE"] = pd.to_datetime(df["DATE"])

# Target variable to predict
target = "daily_demand"

# Number of previous days used for LSTM prediction
LOOKBACK = 30


# ============================================================
# LOAD TRAINED MODELS AND PREPROCESSING OBJECTS
# ============================================================
# This section loads previously trained machine learning
# and deep learning models along with preprocessing objects.
#
# Objects loaded:
# - XGBoost model (tabular prediction)
# - LSTM model (sequence-based prediction)
# - Feature list (selected input variables)
# - Feature scaler (used to scale LSTM input)
# - Target scaler (used to reverse scaling)

# Load trained XGBoost model
xgb_model = joblib.load("models/xgb_multivariate.pkl")

# Load trained LSTM model
lstm_model = load_model("models/lstm_multivariate.keras")

# Load list of feature names
features = joblib.load("models/feature_list.pkl")

# Load scaler used for feature normalization
feature_scaler = joblib.load("models/lstm_feature_scaler.pkl")

# Load scaler used to reverse prediction scaling
target_scaler = joblib.load("models/lstm_target_scaler.pkl")


# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
# This section creates interactive sidebar controls
# that allow the user to:
#
# 1. Select forecasting model.
# 2. Select forecast date.
#
# The selected values are later used to generate predictions.

st.sidebar.title("Model Configuration")

# Dropdown menu to select prediction model
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["XGBoost Multivariate", "LSTM Multivariate"]
)

# Date selector to choose forecast date
selected_date = st.sidebar.date_input(
    "Select Forecast Date",

    # Default value = latest date in dataset
    value=df["DATE"].max(),

    # Restrict date range
    min_value=df["DATE"].min(),
    max_value=df["DATE"].max()
)


# ============================================================
# MAIN PAGE TITLE AND DESCRIPTION
# ============================================================
# This section displays the main title and explains
# what the application does.
#
# Markdown is used to display formatted text.

st.title("UK Daily Electricity Demand Forecasting System")

st.markdown("""
This application forecasts daily UK electricity demand using:

- Machine Learning (XGBoost)
- Deep Learning (LSTM)
- Engineered time-series features
- Historical weather integration

The system was developed following a strict chronological evaluation protocol.
""")


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
# This section defines two prediction functions:
#
# predict_xgb()
#   Uses XGBoost model to predict demand
#   using feature values of selected date.
#
# predict_lstm()
#   Uses LSTM model to predict demand
#   using previous LOOKBACK days.

def predict_xgb(date):

    # Filter row matching selected date
    row = df[df["DATE"] == pd.to_datetime(date)]

    # If date not found, return None
    if row.empty:
        return None

    # Extract feature values
    X = row[features]

    # Make prediction and return result
    return float(xgb_model.predict(X)[0])


def predict_lstm(date):

    # Find index of selected date
    idx = df[df["DATE"] == pd.to_datetime(date)].index

    # Check if enough previous days exist
    if len(idx) == 0 or idx[0] < LOOKBACK:
        return None

    # Extract integer index
    idx = idx[0]

    # Get previous LOOKBACK days
    sequence = df.iloc[idx-LOOKBACK:idx][features].copy()

    # Scale input features
    sequence[features] = feature_scaler.transform(sequence[features])

    # Convert into LSTM input format
    X = np.array([sequence.values])

    # Predict scaled output
    pred_scaled = lstm_model.predict(X, verbose=0)[0][0]

    # Convert scaled prediction to original value
    pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]

    return float(pred)


# ============================================================
# GENERATE PREDICTION
# ============================================================
# This section checks which model is selected
# and calls the corresponding prediction function.

if model_choice == "XGBoost Multivariate":
    prediction = predict_xgb(selected_date)
else:
    prediction = predict_lstm(selected_date)


# ============================================================
# DISPLAY FORECAST RESULT
# ============================================================
# This section displays the prediction result
# as a metric card.
#
# If prediction fails (insufficient data),
# a warning message is shown.

st.subheader("Forecast Result")

if prediction is None:

    st.warning(
        "Not enough historical data available for selected date."
    )

else:

    st.metric(
        label="Predicted Daily Demand (MW)",
        value=f"{prediction:,.0f}"
    )


# ============================================================
# DOWNLOAD CURRENT FORECAST
# ============================================================
# This section allows the user to download
# the predicted value as a CSV file.

st.subheader("Download Current Forecast")

if prediction is not None:

    # Create DataFrame containing prediction
    single_pred_df = pd.DataFrame({
        "DATE": [selected_date],
        "Predicted_Demand": [prediction]
    })

    # Download button
    st.download_button(
        label=f"Download Forecast for {selected_date}",
        data=single_pred_df.to_csv(index=False),
        file_name=f"forecast_{selected_date}.csv",
        mime="text/csv"
    )


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================
# This section loads precomputed evaluation metrics
# and displays them in metric cards.

st.markdown("---")
st.header("Model Evaluation (2022 Test Set)")

metrics_df = pd.read_csv(
    "data/advanced_outputs/xgb_advanced_metrics.csv"
)

# Create four columns
col1, col2, col3, col4 = st.columns(4)

# Display evaluation metrics
col1.metric("MAE", f"{metrics_df['MAE'][0]:,.0f}")
col2.metric("RMSE", f"{metrics_df['RMSE'][0]:,.0f}")
col3.metric("R²", f"{metrics_df['R2'][0]:.4f}")
col4.metric("MAPE (%)", f"{metrics_df['MAPE_%'][0]:.2f}")


# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================
# Displays top 15 most important features
# used by XGBoost model.

if model_choice == "XGBoost Multivariate":

    st.subheader("Top 15 Feature Importance")

    importance_df = pd.read_csv(
        "data/advanced_outputs/feature_importance.csv"
    ).head(15)

    fig, ax = plt.subplots()

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================
# This section displays a line plot comparing
# actual demand vs predicted demand.

st.subheader("Actual vs Predicted (2022 Test Set)")

pred_df = pd.read_csv(
    "data/advanced_outputs/xgb_test_predictions.csv"
)

fig2, ax2 = plt.subplots()

ax2.plot(
    pred_df["Actual"],
    label="Actual"
)

ax2.plot(
    pred_df["Predicted"],
    label="Predicted"
)

ax2.legend()

plt.tight_layout()

st.pyplot(fig2)


# ============================================================
# SEASONAL ERROR ANALYSIS
# ============================================================
# Displays bar chart showing seasonal
# Mean Absolute Error values.

st.subheader("Seasonal MAE (2022)")

season_df = pd.read_csv(
    "data/advanced_outputs/seasonal_mae.csv"
)

fig3, ax3 = plt.subplots()

ax3.bar(
    season_df["Season"],
    season_df["MAE"]
)

plt.tight_layout()

st.pyplot(fig3)


# ============================================================
# DOWNLOAD FULL EVALUATION RESULTS
# ============================================================
# Allows users to download complete test predictions.

st.subheader("Download Full Evaluation Data")

st.download_button(
    label="Download 2022 Test Predictions",
    data=pred_df.to_csv(index=False),
    file_name="xgb_test_predictions_2022.csv",
    mime="text/csv"
)
