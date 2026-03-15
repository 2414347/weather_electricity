import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

INPUT_FILE = "data/processed/model_ready_daily_dataset.csv"
OUTPUT_FILE = "data/results/lstm_results.csv"

LOOKBACK = 30

df = pd.read_csv(INPUT_FILE)
df["DATE"] = pd.to_datetime(df["DATE"])

train = df[(df["DATE"] >= "2009-01-01") & (df["DATE"] <= "2019-12-31")]
val = df[(df["DATE"] >= "2020-01-01") & (df["DATE"] <= "2021-12-31")]
test = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] <= "2022-12-31")]

target = "daily_demand"

univariate_features = [target]
multivariate_features = [col for col in df.columns if col not in ["DATE"]]


def create_sequences(data, features):
    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[features].iloc[i-LOOKBACK:i].values)
        y.append(data[target].iloc[i])
    return np.array(X), np.array(y)


def train_lstm(features, model_name):

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers on training only
    feature_scaler.fit(train[features])
    target_scaler.fit(train[[target]])

    # Scale
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    train_scaled[features] = feature_scaler.transform(train[features])
    val_scaled[features] = feature_scaler.transform(val[features])
    test_scaled[features] = feature_scaler.transform(test[features])

    train_scaled[target] = target_scaler.transform(train[[target]])
    val_scaled[target] = target_scaler.transform(val[[target]])
    test_scaled[target] = target_scaler.transform(test[[target]])

    X_train, y_train = create_sequences(train_scaled, features)
    X_val, y_val = create_sequences(val_scaled, features)
    X_test, y_test = create_sequences(test_scaled, features)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict (scaled)
    val_pred_scaled = model.predict(X_val).flatten()
    test_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform to original scale
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1,1)).flatten()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1,1)).flatten()

    y_val_orig = target_scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    val_mae = mean_absolute_error(y_val_orig, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred))

    test_mae = mean_absolute_error(y_test_orig, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))

    return [
        [model_name, "Validation", val_mae, val_rmse],
        [model_name, "Test", test_mae, test_rmse]
    ]


results = []
results.extend(train_lstm(univariate_features, "LSTM_Univariate"))
results.extend(train_lstm(multivariate_features, "LSTM_Multivariate"))

results_df = pd.DataFrame(results, columns=["Model", "Dataset", "MAE", "RMSE"])
os.makedirs("data/results", exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print(results_df)