# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 22:32:25 2025

@author: abul mohsin
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import os
import json
import time
import GPUtil
import platform

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'liteformer_tuning_log.json')

# Loading Data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

X_train = X_train[..., np.newaxis] if X_train.ndim == 2 else X_train
X_val   = X_val[..., np.newaxis] if X_val.ndim == 2 else X_val

input_shape = (X_train.shape[1], X_train.shape[2])

# Subsetting (5%)
subset_frac = 0.05
X_train_small = X_train[:int(subset_frac * len(X_train))]
y_train_small = y_train[:int(subset_frac * len(y_train))]
X_val_small   = X_val[:int(subset_frac * len(X_val))]
y_val_small   = y_val[:int(subset_frac * len(y_val))]

# Patch Slicing

def slice_patches(inputs, patch_size):
    feat_dim = inputs.shape[2]
    num_patches = inputs.shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [-1, num_patches, patch_size * feat_dim])

# Model Builder

def build_liteformer(hp):
    inputs = keras.Input(shape=input_shape)
    patch_size = hp.Choice('patch_size', [4, 6, 8, 12])
    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    hidden_dim = hp.Int('hidden_dim', 32, 128, step=16)
    dropout = hp.Float('dropout', 0.0, 0.3, step=0.05)

    q = layers.Dense(hidden_dim)(x)
    k = layers.Dense(hidden_dim)(x)
    v = layers.Dense(hidden_dim)(x)

    scores = tf.matmul(q, k, transpose_b=True)
    scores /= tf.math.sqrt(tf.cast(hidden_dim, tf.float32))
    attn_weights = tf.nn.softmax(scores, axis=-1)
    attn_output = tf.matmul(attn_weights, v)
    attn_output = layers.Dropout(dropout)(attn_output)

    x_proj = layers.Dense(hidden_dim)(x)
    x = layers.Add()([x_proj, attn_output])
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(12)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 5e-3, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

# Tuner Setup
project_name = 'liteformer_tuning'
tuner = kt.BayesianOptimization(
    build_liteformer,
    objective='val_loss',
    max_trials=20,
    directory=DATA_DIR,
    project_name=project_name
)

# Timing start
t0 = time.time()

# Early stopping
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run Tuning
with tf.device('/GPU:0'):
    tuner.search(
        X_train_small, y_train_small,
        validation_data=(X_val_small, y_val_small),
        epochs=50,
        batch_size=512,
        callbacks=[stop]
    )

t1 = time.time()

total_time = round((t1 - t0) / 60, 2)

# Getting Best HPs
best_hp = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hp)
total_params = model.count_params()
gpu_info = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "No GPU"

log_dict = {
    "model": "LiteFormer",
    "task": "Smart Meter Energy Forecasting",
    "tuning_type": "BayesianOptimization",
    "tuning_time_minutes": total_time,
    "best_hyperparameters": best_hp.values,
    "total_params": total_params,
    "input_shape": input_shape,
    "patch_size": best_hp.values['patch_size'],
    "gpu_used": gpu_info,
    "platform": platform.platform(),
    "log_type": "Tuning"
}

# Saving Log
with open(LOG_PATH, 'w') as f:
    json.dump(log_dict, f, indent=4)

print("Tuning complete. Log saved to:", LOG_PATH)
