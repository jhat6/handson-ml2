# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:40:54 2022

@author: jeffa
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt



(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train_full[:50000]/255
X_val = X_train_full[50000:]/255

y_train = y_train_full[:50000]
y_val = y_train_full[50000:]


keras.backend.clear_session()
np.random.seed(6)
tf.random.set_seed(6)


def build_model(n_hidden=1, n_neurons=30, input_shape=[28, 28]):
    model=keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))    
    model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=keras.optimizers.SGD(learning_rate=.007), 
              metrics=["Accuracy"])
    return model


keras_reg =keras.wrappers.scikit_learn.KerasClassifier(build_model)

from sklearn.model_selection import RandomizedSearchCV

param_distribs = {"n_hidden": [3,4,5,6],
                 "n_neurons": np.arange(10, 300, 10)}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, n_jobs=-1)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])