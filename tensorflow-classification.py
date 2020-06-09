# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:22:03 2020

@author: Machachane
"""



import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds


print('\nVersion:\n', tf.__version__)
print('\nEager mode:\n', tf.executing_eagerly())
print('\nHub version:\n', hub.__version__)
print('\nGPU is', 'available' if tf.config.experimental.list_physical_devices('GPU') else 'NOT AVAILABLE')

print('\n------------------------------------------------------------------------------------------------\n')

train_data, validation_data, test_data = tfds.load(name = "imdb_reviews",
                                                  split = ('train[:60%]', ' train[60%:]', 'test'),
                                                  as_supervised = True)

print('\n------------------------------------------------------------------------------------------------\n')

train_examples_batch, train_label_batch = next(iter(train_data.batch(10)))
train_examples_batch

print('\n------------------------------------------------------------------------------------------------\n')

train_labels_batch

print('\n------------------------------------------------------------------------------------------------\n')

embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layer.Dense(16, activation='relu'))
model.add(tf.keras.layer.Dense(1))

model.summary()
