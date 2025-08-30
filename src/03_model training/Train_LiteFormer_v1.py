# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:31:35 2025

@author: student
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
BASE_PATH = r'G:\My Drive\EdgeMeter_AI\Data\processed'
X_train = np.load(os.path.join(BASE_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(BASE_PATH, 'y_train.npy'))
X_val   = np.load(os.path.join(BASE_PATH, 'X_val.npy'))
y_val   = np.load(os.path.join(BASE_PATH, 'y_val.npy'))
X_test  = np.load(os.path.join(BASE_PATH, 'X_test.npy'))
y_test  = np.load(os.path.join(BASE_PATH, 'y_test.npy'))

# Reshape for model input
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2])
print("Train shape:", X_train.shape)
print("Val shape:  ", X_val.shape)

# Loading best HPs
with open(os.path.join(BASE_PATH, 'liteformer_tuning_log.json'), 'r') as f:
    best_hp = json.load(f)

best_hps = best_hp['best_hyperparameters']
print("Best HPs:", best_hps)

# Patch slicer
def slice_patches(inputs, patch_size):
    feat_dim = inputs.shape[2]
    num_patches = inputs.shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [-1, num_patches, patch_size * feat_dim])

# LiteFormer Architecture
def build_liteformer(hp_dict):
    inputs = keras.Input(shape=input_shape)

    patch_size    = hp_dict['patch_size']
    hidden_dim    = hp_dict['hidden_dim']
    dropout       = hp_dict['dropout']
    learning_rate = hp_dict['learning_rate']

    # Patch slicing
    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    # Project to hidden_dim BEFORE adding positional encoding
    x = layers.Dense(hidden_dim)(x)  # Ensure shape: (batch, num_patches, hidden_dim)

    # Learnable positional encoding
    num_patches = input_shape[0] // patch_size
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=hidden_dim)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_encoding = pos_embed(positions)  # shape: (num_patches, hidden_dim)
    x = x + pos_encoding  # Broadcasting over batch

    # Pre-attention transformation
    x = layers.Dense(hidden_dim, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)

    # Linear attention
    q = layers.Dense(hidden_dim)(x)
    k = layers.Dense(hidden_dim)(x)
    v = layers.Dense(hidden_dim)(x)
    attn_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(hidden_dim, tf.float32)), axis=-1)
    attn_output = tf.matmul(attn_weights, v)
    attn_output = layers.Dropout(dropout)(attn_output)

    # Residual connection and LayerNorm
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Lightweight Feedforward Network
    x = layers.Dense(hidden_dim * 2, activation='gelu')(x)
    x = layers.Dense(hidden_dim)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output head
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(12)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


print("Loaded HPs:", best_hp)
print("All Keys:", list(best_hp.keys()))

# Training
model = build_liteformer(best_hps)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()
with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=[early_stop],
        verbose=1
    )
end_time = time.time()
train_time = end_time - start_time

# Saving training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(BASE_PATH, 'LiteFormer_Training_History.csv'))

# Plot for training curve
plt.figure()
plt.plot(hist_df['loss'], label='Train Loss')
plt.plot(hist_df['val_loss'], label='Val Loss')
plt.title('LiteFormer Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_PATH, 'LiteFormer_Training_Curve.png'))
plt.close()

# Evaluating on test set
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(mse)

test_r2 = r2_score(y_test, y_pred)

# Model size
SAVEDMODEL_PATH = os.path.join(BASE_PATH, 'Final_LiteFormer_Model')
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

# Saving evaluation
log_data = {
    "model_name": "LiteFormer_Final",
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2,
    "params": params,
    "model_size_mb": size_mb,
    "inference_time_ms": inference_time,
    "training_time_minutes": round((end_time - start_time) / 60, 2),
    "input_shape": list(X_train.shape[1:]),
    "patch_size": best_hps.get('patch_size'),  
    "hidden_dim": best_hps.get('hidden_dim'),  
    "dropout": best_hps.get('dropout'),
    "learning_rate": best_hps.get('learning_rate'),
    "system_info": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "gpu_used": GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "None"
    }
}

with open(os.path.join(BASE_PATH, 'LiteFormer_Final_Log.json'), 'w') as f:
    json.dump(log_data, f, indent=4)

print("LiteFormer model training and evaluation complete.")

# Saving final prediction CSV
np.savetxt(os.path.join(BASE_PATH, 'LiteFormer_y_pred.csv'), y_pred, delimiter=",")

