import os
import tensorflow as tf

# get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "lstm_multivariate.keras")
output_path = os.path.join(BASE_DIR, "models", "lstm_multivariate.h5")

print("Loading model from:", model_path)

model = tf.keras.models.load_model(model_path)

model.save(output_path)

print("Conversion successful")
print("Saved to:", output_path)