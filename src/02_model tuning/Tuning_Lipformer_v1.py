# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 15:37:03 2025

@author: Abul Mohsin
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import json
import os
import random
import time
import platform
import GPUtil

# For Reproducibility
tf.keras.backend.clear_session()
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'lipformer_tuning_log.json')

# Loading data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# Adding feature/channel dimension
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
input_shape = (X_train.shape[1], X_train.shape[2])

print("Train shape:", X_train.shape)
print("Val shape:  ", X_val.shape)

# Subsetting (5%) for tuning
frac = 0.05
X_train_small = X_train[:int(frac * len(X_train))]
y_train_small = y_train[:int(frac * len(y_train))]
X_val_small   = X_val[:int(frac * len(X_val))]
y_val_small   = y_val[:int(frac * len(y_val))]

print("Subset shapes:", X_train_small.shape, X_val_small.shape)

# Patch slicer
def slice_patches(inputs, patch_size):
    feat_dim = inputs.shape[2]
    num_patches = inputs.shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [-1, num_patches, patch_size * feat_dim])


# LiPFormer model
def build_lipformer(hp):
    inputs = keras.Input(shape=input_shape)

    patch_size = hp.Choice('patch_size', [4, 6, 8, 12])
    if input_shape[0] % patch_size != 0:
        raise ValueError(f"Time steps ({input_shape[0]}) must be divisible by patch size ({patch_size})")

    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    hidden_dim = hp.Int('hidden_dim', 32, 128, step=32)
    dropout = hp.Float('dropout', 0.0, 0.4, step=0.1)
    x_proj = layers.Dense(hidden_dim)(x)

    # Cross-patch lightweight attention using DepthwiseConv2D
    x_reshaped = tf.expand_dims(x_proj, axis=2)  # (B, N, 1, D)
    x_attn = layers.DepthwiseConv2D((1, 1))(x_reshaped)
    x_attn = tf.squeeze(x_attn, axis=2)  # (B, N, D)

    x = layers.Add()([x_proj, x_attn])
    x = layers.Dropout(dropout)(x)

    # lightweight feedforward layer
    if hp.Boolean('use_mlp'):
        ffn_dim = hp.Int('ff_dim', 64, 256, step=64)
        x = layers.Dense(ffn_dim, activation='gelu')(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(12, activation='linear')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 5e-3, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model


# Tuner setup
tuner = kt.BayesianOptimization(
    build_lipformer,
    objective='val_loss',
    max_trials=20,
    directory=DATA_DIR,
    project_name='lipformer_tuning'
)

# Callbacks
stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Tuning with timer
start_time = time.time()
with tf.device('/GPU:0'):
    tuner.search(
        X_train_small, y_train_small,
        validation_data=(X_val_small, y_val_small),
        epochs=50,
        batch_size=512,
        callbacks=[stop_early]
    )
end_time = time.time()
tuning_minutes = round((end_time - start_time) / 60, 2)

# Saving Logs
best_hp = tuner.get_best_hyperparameters(1)[0].values

best_model = tuner.get_best_models(1)[0]
total_params = best_model.count_params()
gpu_name = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "Unknown"
platform_info = platform.platform()

log_dict = {
    "model": "LiPFormer",
    "task": "Smart Meter Energy Forecasting",
    "tuning_type": "BayesianOptimization",
    "tuning_time_minutes": tuning_minutes,
    "best_hyperparameters": best_hp,
    "total_params": total_params,
    "input_shape": list(input_shape),
    "patch_size": best_hp["patch_size"],
    "gpu_used": gpu_name,
    "platform": platform_info,
    "log_type": "Tuning"
}

with open(LOG_PATH, 'w') as f:
    json.dump(log_dict, f, indent=4)

print("Tuning log saved to:", LOG_PATH)