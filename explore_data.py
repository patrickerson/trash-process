from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd



import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


dataframe = pd.read_csv("features.csv", sep=',', header=None)
dataframe = dataframe.rename(columns={2048: "classe"})

dataframe['classe'] = pd.Categorical(dataframe['classe'])
dataframe['classe'] = dataframe.classe.cat.codes

print(dataframe['classe'])

# dataset = np.array(dataframe.groupby(dataframe.index))

class_names = ["cardboard", "glass", "paper", "plastic", "trash"]

target = dataframe.pop('classe')
dataset = tf.data.Dataset.from_tensor_slices((dataframe.values, target.values))




train_dataset = dataset.shuffle(len(dataframe)).batch(1)




def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)
model.summary()
model.save('separa_lixo.h5')

