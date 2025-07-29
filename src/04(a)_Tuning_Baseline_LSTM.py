# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 08:44:06 2025
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
import platform
import GPUtil

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'lstm_tuning_log.json')

# Loading data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

print(f"Train X: {X_train.shape} | Train y: {y_train.shape}")
print(f"Val   X: {X_val.shape}   | Val   y: {y_val.shape}")
print(f"Test  X: {X_test.shape}  | Test  y: {y_test.shape}")

# Reshaping for LSTM (adding feature dim)
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]
print("Input reshaped:", X_train.shape)

# Creating small subset for tuning (5%)
subset_frac = 0.05
X_train_small = X_train[:int(subset_frac * X_train.shape[0])]
y_train_small = y_train[:int(subset_frac * y_train.shape[0])]
X_val_small   = X_val[:int(subset_frac * X_val.shape[0])]
y_val_small   = y_val[:int(subset_frac * y_val.shape[0])]
print(f"Subset shapes: {X_train_small.shape} | {X_val_small.shape}")

# Model
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32)
    ))
    model.add(layers.Dropout(
        hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    ))
    model.add(layers.Dense(12))  # Output steps

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

# Tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    directory=DATA_DIR,
    project_name='lstm_retune'
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

# Tuning with timing
start_time = time.time()

with tf.device('/GPU:0'):
    tuner.search(
        X_train_small, y_train_small,
        validation_data=(X_val_small, y_val_small),
        epochs=50,
        batch_size=512,
        callbacks=[early_stop]
    )

end_time = time.time()

# Saving best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
hp_dict = {
    'units': best_hp.get('units'),
    'dropout': best_hp.get('dropout'),
    'learning_rate': best_hp.get('learning_rate')
}

hp_path = os.path.join(DATA_DIR, 'best_lstm_hp.json')
with open(hp_path, 'w') as f:
    json.dump(hp_dict, f, indent=4)

# Logging tuning results
model_tmp = build_model(best_hp)
total_params = model_tmp.count_params()
gpu_name = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "Unknown"
platform_info = platform.platform()

log = {
    "model": "LSTM",
    "task": "Smart Meter Energy Forecasting",
    "tuning_type": "BayesianOptimization",
    "tuning_time_minutes": round((end_time - start_time) / 60, 2),
    "best_hyperparameters": hp_dict,
    "total_params": total_params,
    "input_shape": list(X_train.shape[1:]),
    "sequence_length": X_train.shape[1],
    "gpu_used": gpu_name,
    "platform": platform_info,
    "log_type": "Tuning"
}

with open(LOG_PATH, 'w') as f:
    json.dump(log, f, indent=4)

print(f"\nBest HPs saved to: {hp_path}")
print(f"Tuning log saved to: {LOG_PATH}")
print(json.dumps(log, indent=4))
