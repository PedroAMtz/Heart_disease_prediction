import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Reading data directly from csv file

heart_data = pd.read_csv("heart.csv")

# Dividing features and labels

heart_features = heart_data.copy()
heart_labels = heart_features.pop("target")

# Convert them to numpy arrays

heart_features = np.array(heart_features)
heart_labels = np.array(heart_labels)

#Define de model

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

model.fit(heart_features, heart_labels, epochs=10)
