# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:37:03 2025

@author: Abul Mohsin
"""

import os
import json
import time
import platform
import GPUtil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'lipformer_training_log.json')

# Loading Data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2])
print("Input shape:", input_shape)

# Loading best HPs
hp_path = os.path.join(DATA_DIR, 'lipformer_tuning_log.json')
with open(hp_path, 'r') as f:
    best_hp = json.load(f)
    
best_hps = best_hp['best_hyperparameters']
print("Best HPs:", best_hps)

patch_size     = best_hps['patch_size']
hidden_dim     = best_hps['hidden_dim']
dropout        = best_hps['dropout']
learning_rate  = best_hps['learning_rate']
ff_dim         = best_hps['ff_dim']


# Patch Slicing
def slice_patches(inputs, patch_size):
    input_shape = tf.shape(inputs)
    num_patches = input_shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [input_shape[0], num_patches, patch_size * input_shape[2]])

# Lightweight Feedforward
def linear_ffn(x, hidden_dim):
    x = layers.Dense(hidden_dim, activation='gelu')(x)
    x = layers.Dense(hidden_dim)(x)
    return x

# LiPFormer Model
def build_lipformer(hp_dict):
    inputs = keras.Input(shape=input_shape)

    patch_size    = hp_dict['patch_size']
    hidden_dim    = hp_dict['hidden_dim']
    dropout       = hp_dict['dropout']
    learning_rate = hp_dict['learning_rate']
    ff_dim        = hp_dict['ff_dim']

    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    # Linear Attention
    q = layers.Dense(hidden_dim)(x)
    k = layers.Dense(hidden_dim)(x)
    v = layers.Dense(hidden_dim)(x)

    scores = tf.matmul(q, k, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(hidden_dim, tf.float32))
    attn_weights = tf.nn.softmax(scores, axis=-1)
    attn_output = tf.matmul(attn_weights, v)
    attn_output = layers.Dropout(dropout)(attn_output)

    # Residual connection
    x_proj = layers.Dense(hidden_dim)(x)
    x = layers.Add()([x_proj, attn_output])

    # Lightweight FFN
    x = layers.Dense(ff_dim, activation='gelu')(x)
    x = layers.Dense(ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(12)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


# Training the model
model = build_lipformer(best_hps)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

start_time = time.time()
with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )
end_time = time.time()


# Saving training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(DATA_DIR, 'LipFormer_Training_History.csv'))

# Plot training curve
plt.figure()
plt.plot(hist_df['loss'], label='Train Loss')
plt.plot(hist_df['val_loss'], label='Val Loss')
plt.title('LipFormer Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, 'LipFormer_Training_Curve.png'))
plt.close()

# Evaluation
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(mse)

test_r2 = r2_score(y_test, y_pred)

# Model size
SAVEDMODEL_PATH = os.path.join(DATA_DIR, 'Final_LiPFormer_Model')
model.save(SAVEDMODEL_PATH, save_format='tf')  # TensorFlow SavedModel format

params = model.count_params()

# calculating directory size
def get_dir_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1e6  

size_mb = get_dir_size_mb(SAVEDMODEL_PATH)

# Inference latency
import timeit
sample_input = tf.convert_to_tensor(X_test[:1])
start = timeit.default_timer()
_ = model(sample_input)
end = timeit.default_timer()
inference_time = (end - start) * 1000

# GPU Info
gpu_name = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "Unknown"
platform_info = platform.platform()

# Logging
log_data = {
    "model_name": "LiPFormer_Final",
    "task": "Smart Meter Energy Forecasting",
    "tuning_type": "BayesianOptimization",
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2,
    "params": params,
    "model_size_mb": size_mb,
    "inference_time_ms": inference_time,
    "training_time_minutes": round((end_time - start_time) / 60, 2),
    "input_shape": list(X_train.shape[1:]),
    "patch_size": best_hps['patch_size'],
    "hidden_dim": best_hps['hidden_dim'],
    "dropout": best_hps['dropout'],
    "learning_rate": best_hps['learning_rate'],
    "ff_dim": best_hps['ff_dim'],
    "system_info": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "gpu_used": GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "None"
    }
}

with open(LOG_PATH, 'w') as f:
    json.dump(log_data, f, indent=4)

print(f"\n Log saved to: {LOG_PATH}")
print(json.dumps(log_data, indent=4))

# Saving final prediction CSV
np.savetxt(os.path.join(DATA_DIR, 'LipFormer_y_pred.csv'), y_pred, delimiter=",")

