# ==========================================================
# IMPORT REQUIRED LIBRARY
# TensorFlow is used to load and save deep learning models.
# ==========================================================
import os
import tensorflow as tf


# ==========================================================
# DEFINE PROJECT ROOT DIRECTORY
# This ensures paths work correctly regardless of where script runs.
# It moves two levels up from the current file location.
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# DEFINE MODEL INPUT AND OUTPUT PATHS
# - model_path: original .keras model file
# - output_path: converted .h5 model file
# ==========================================================
model_path = os.path.join(BASE_DIR, "models", "lstm_multivariate.keras")
output_path = os.path.join(BASE_DIR, "models", "lstm_multivariate.h5")


# ==========================================================
# DISPLAY LOADING STATUS
# Helps confirm correct file path before loading model
# ==========================================================
print("Loading model from:", model_path)


# ==========================================================
# LOAD TRAINED KERAS MODEL
# Loads the saved LSTM model from .keras format
# ==========================================================
model = tf.keras.models.load_model(model_path)


# ==========================================================
# SAVE MODEL IN HDF5 FORMAT (.h5)
# Converts .keras model to legacy .h5 format for compatibility
# ==========================================================
model.save(output_path)


# ==========================================================
# FINAL CONFIRMATION OUTPUT
# Indicates successful conversion and saved file location
# ==========================================================
print("Conversion successful")
print("Saved to:", output_path)
