# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 21:35:19 2025

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

# Loading best HPs
with open(os.path.join(BASE_PATH, 'tinyformer_tuning_log.json'), 'r') as f:
    best_hp = json.load(f)

best_hps = best_hp['best_hyperparameters']
print("Best HPs:", best_hps)

# Patch slice
def slice_patches(inputs, patch_size):
    input_shape = tf.shape(inputs)
    num_patches = input_shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [input_shape[0], num_patches, patch_size * input_shape[2]])

# Model
def build_final_tinyformer(hp_dict):
    inputs = keras.Input(shape=input_shape)
    patch_size = hp_dict['patch_size']
    x = layers.Lambda(lambda t: slice_patches(t, patch_size))(inputs)

    x_attn = layers.MultiHeadAttention(
        num_heads=hp_dict['num_heads'],
        key_dim=hp_dict['hidden_dim'],
        dropout=hp_dict['dropout']
    )(x, x)

    x = layers.Add()([x, x_attn])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(12)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_dict['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    return model

# Train
model = build_final_tinyformer(best_hps)
with open(os.path.join(BASE_PATH, 'tinyformer_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(BASE_PATH, 'tinyformer_best.h5'), save_best_only=True)
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
hist_df.to_csv(os.path.join(BASE_PATH, 'TinyFormer_Final_Training_History.csv'))

# Plot training curve
plt.figure()
plt.plot(hist_df['loss'], label='Train Loss')
plt.plot(hist_df['val_loss'], label='Val Loss')
plt.title('TinyFormer Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_PATH, 'TinyFormer_Training_Curve.png'))
plt.close()

# Evaluation on test set
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(mse)

test_r2 = r2_score(y_test, y_pred)

# Model size
SAVEDMODEL_PATH = os.path.join(BASE_PATH, 'Final_TinyFormer_Model')
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
    "model_name": "TinyFormer_Final",
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
    "num_heads": best_hps['num_heads'],
    "dropout": best_hps['dropout'],
    "learning_rate": best_hps['learning_rate'],
    "system_info": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "gpu_used": GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "None"
    }
}

with open(os.path.join(BASE_PATH, 'TinyFormer_Final_Log.json'), 'w') as f:
    json.dump(log_data, f, indent=4)

print("TinyFormer model training and evaluation complete.")

# Saving final prediction CSV
np.savetxt(os.path.join(BASE_PATH, 'TinyFormer_y_pred.csv'), y_pred, delimiter=",")

