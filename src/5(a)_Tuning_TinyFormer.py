# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:23:56 2025

@author: abul mohsin
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import json
import os
import time
import platform
import GPUtil

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'tinyformer_tuning_log.json')

# Loading Data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
input_shape = (X_train.shape[1], X_train.shape[2])

# 5% Subset for Tuning
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

# TinyFormer Model
def build_tinyformer(hp):
    inputs = keras.Input(shape=input_shape)
    patch_size = hp.Choice('patch_size', values=[4, 6, 8, 12])
    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    x_attn = layers.MultiHeadAttention(
        num_heads=hp.Int('num_heads', 1, 4),
        key_dim=hp.Int('hidden_dim', 16, 128, step=16),
        dropout=hp.Float('dropout', 0.0, 0.3, step=0.05)
    )(x, x)

    x = layers.Add()([x, x_attn])
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
tuner = kt.BayesianOptimization(
    build_tinyformer,
    objective='val_loss',
    max_trials=20,
    directory=DATA_DIR,
    project_name='tinyformer_tuning'
)

stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Tuning Execution
start_time = time.time()
tuner.search(
    X_train_small, y_train_small,
    validation_data=(X_val_small, y_val_small),
    epochs=50,
    batch_size=512,
    callbacks=[stop_early],
    verbose=1
)
end_time = time.time()

# Saving Best HPs
best_hp = tuner.get_best_hyperparameters(1)[0].values
hp_path = os.path.join(DATA_DIR, 'tinyformer_best_hp.json')
with open(hp_path, 'w') as f:
    json.dump(best_hp, f, indent=4)
print("Best HPs saved to:", hp_path)
print(best_hp)

# Log 
gpu_name = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "Unknown"
platform_info = platform.platform()
dummy_model = tuner.get_best_models(1)[0]

log = {
    "model": "TinyFormer",
    "task": "Smart Meter Energy Forecasting",
    "tuning_type": "BayesianOptimization",
    "tuning_time_minutes": round((end_time - start_time) / 60, 2),
    "best_hyperparameters": best_hp,
    "total_params": dummy_model.count_params(),
    "input_shape": list(input_shape),
    "patch_size": best_hp['patch_size'],
    "gpu_used": gpu_name,
    "platform": platform_info,
    "log_type": "Tuning"
}

with open(LOG_PATH, 'w') as f:
    json.dump(log, f, indent=4)

print(f"\nTuning log saved to: {LOG_PATH}")
print(json.dumps(log, indent=4))
