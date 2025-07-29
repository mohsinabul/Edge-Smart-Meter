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

# Build model
def build_liteformer_from_hp(hp_dict):
    inputs = keras.Input(shape=input_shape)

    patch_size = hp_dict.get('patch_size', 8)
    hidden_dim = hp_dict.get('hidden_dim', 64)
    dropout = hp_dict.get('dropout', 0.1)
    learning_rate = hp_dict.get('learning_rate', 5e-4)

    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)
    x = layers.Dense(hidden_dim, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)

    q = layers.Dense(hidden_dim)(x)
    k = layers.Dense(hidden_dim)(x)
    v = layers.Dense(hidden_dim)(x)

    k_trans = tf.transpose(k, perm=[0, 2, 1])
    attn_weights = tf.matmul(q, k_trans)
    attn_weights = tf.nn.softmax(attn_weights / tf.math.sqrt(tf.cast(hidden_dim, tf.float32)))
    attn_output = tf.matmul(attn_weights, v)

    x = layers.Add()([x, attn_output])
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
model = build_liteformer_from_hp(best_hps)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_path = os.path.join(BASE_PATH, 'liteformer_model.h5')

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
model.save(os.path.join(BASE_PATH, 'Final_LiteFormer_Model.h5'))
params = model.count_params()
size_mb = os.path.getsize(os.path.join(BASE_PATH, 'Final_LiteFormer_Model.h5')) / 1e6

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

