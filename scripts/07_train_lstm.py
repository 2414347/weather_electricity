# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports libraries required for:
#
# os:
#   Used to create folders and manage file paths.
#
# numpy:
#   Used for numerical array operations.
#
# pandas:
#   Used to load and manipulate dataset.
#
# StandardScaler:
#   Used to normalize feature values.
#
# sklearn.metrics:
#   Used to calculate MAE and RMSE.
#
# TensorFlow / Keras:
#   Used to build and train LSTM neural networks.

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# DEFINE INPUT AND OUTPUT FILES
# ============================================================
# INPUT_FILE:
#   Feature-engineered dataset used for deep learning.
#
# OUTPUT_FILE:
#   Stores LSTM model performance results.

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/lstm_results.csv"


# ============================================================
# DEFINE LOOKBACK WINDOW
# ============================================================
# LOOKBACK defines how many past days
# are used as input for prediction.
#
# Example:
# LOOKBACK = 30 means model uses
# last 30 days to predict next day.

LOOKBACK = 30


# ============================================================
# LOAD DATASET
# ============================================================
# This section loads dataset and converts
# DATE column into datetime format.

df = pd.read_csv(INPUT_FILE)

df["DATE"] = pd.to_datetime(df["DATE"])


# ============================================================
# CHRONOLOGICAL DATA SPLIT
# ============================================================
# Dataset is split into:
#
# Train Set (2009–2019)
# Validation Set (2020–2021)
# Test Set (2022)
#
# This ensures proper time-series evaluation.

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


# Define target variable
target = "daily_demand"


# ============================================================
# DEFINE FEATURE SETS
# ============================================================
# Two LSTM configurations are used:
#
# Univariate LSTM:
#   Uses only demand values.
#
# Multivariate LSTM:
#   Uses demand + weather + calendar features.

univariate_features = [target]

multivariate_features = [

    col for col in df.columns

    if col not in ["DATE"]

]


# ============================================================
# SEQUENCE CREATION FUNCTION
# ============================================================
# LSTM requires sequential input data.
#
# This function converts tabular data into
# sequences using sliding window method.
#
# Example:
# If LOOKBACK = 30,
# model uses 30 previous days
# to predict next day's demand.

def create_sequences(data, features):

    X, y = [], []

    for i in range(LOOKBACK, len(data)):

        # Input sequence
        X.append(

            data[features]

            .iloc[i-LOOKBACK:i]

            .values

        )

        # Target value
        y.append(

            data[target]

            .iloc[i]

        )

    return np.array(X), np.array(y)


# ============================================================
# LSTM TRAINING FUNCTION
# ============================================================
# This function performs:
#
# 1. Feature scaling
# 2. Sequence creation
# 3. Model building
# 4. Model training
# 5. Prediction
# 6. Evaluation

def train_lstm(features, model_name):


    # --------------------------------------------------------
    # SCALING FEATURES
    # --------------------------------------------------------
    # Neural networks require normalized inputs.

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()


    # Fit only on training data
    feature_scaler.fit(train[features])

    target_scaler.fit(train[[target]])


    # Create scaled copies
    train_scaled = train.copy()

    val_scaled = val.copy()

    test_scaled = test.copy()


    # Apply scaling
    train_scaled[features] = feature_scaler.transform(train[features])

    val_scaled[features] = feature_scaler.transform(val[features])

    test_scaled[features] = feature_scaler.transform(test[features])


    train_scaled[target] = target_scaler.transform(train[[target]])

    val_scaled[target] = target_scaler.transform(val[[target]])

    test_scaled[target] = target_scaler.transform(test[[target]])


    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    X_train, y_train = create_sequences(

        train_scaled,

        features

    )

    X_val, y_val = create_sequences(

        val_scaled,

        features

    )

    X_test, y_test = create_sequences(

        test_scaled,

        features

    )


    # --------------------------------------------------------
    # BUILD LSTM MODEL
    # --------------------------------------------------------

    model = Sequential()

    # LSTM layer
    model.add(

        LSTM(

            64,

            input_shape=(

                X_train.shape[1],

                X_train.shape[2]

            )

        )

    )

    # Dropout layer
    model.add(

        Dropout(0.2)

    )

    # Output layer
    model.add(

        Dense(1)

    )


    # Compile model
    model.compile(

        optimizer="adam",

        loss="mse"

    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    # Stops training if validation
    # performance stops improving.

    early_stop = EarlyStopping(

        patience=10,

        restore_best_weights=True

    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=100,

        batch_size=32,

        callbacks=[early_stop],

        verbose=0

    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    val_pred_scaled = model.predict(X_val).flatten()

    test_pred_scaled = model.predict(X_test).flatten()


    # --------------------------------------------------------
    # INVERSE SCALING
    # --------------------------------------------------------

    val_pred = target_scaler.inverse_transform(

        val_pred_scaled.reshape(-1,1)

    ).flatten()

    test_pred = target_scaler.inverse_transform(

        test_pred_scaled.reshape(-1,1)

    ).flatten()


    y_val_orig = target_scaler.inverse_transform(

        y_val.reshape(-1,1)

    ).flatten()

    y_test_orig = target_scaler.inverse_transform(

        y_test.reshape(-1,1)

    ).flatten()


    # --------------------------------------------------------
    # CALCULATE PERFORMANCE
    # --------------------------------------------------------

    val_mae = mean_absolute_error(

        y_val_orig,

        val_pred

    )

    val_rmse = np.sqrt(

        mean_squared_error(

            y_val_orig,

            val_pred

        )

    )

    test_mae = mean_absolute_error(

        y_test_orig,

        test_pred

    )

    test_rmse = np.sqrt(

        mean_squared_error(

            y_test_orig,

            test_pred

        )

    )


    return [

        [model_name, "Validation", val_mae, val_rmse],

        [model_name, "Test", test_mae, test_rmse]

    ]


# ============================================================
# TRAIN BOTH LSTM MODELS
# ============================================================

results = []

results.extend(

    train_lstm(

        univariate_features,

        "LSTM_Univariate"

    )

)

results.extend(

    train_lstm(

        multivariate_features,

        "LSTM_Multivariate"

    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Dataset",

        "MAE",

        "RMSE"

    ]

)

os.makedirs(

    "data/results",

    exist_ok=True

)

results_df.to_csv(

    OUTPUT_FILE,

    index=False

)

print(results_df)
