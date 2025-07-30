# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:01:12 2025
@author: abul mohsin

"""

import os
import time
import json
import platform
import numpy as np
import GPUtil
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI'
DATA_DIR = os.path.join(BASE_PATH, 'Data', 'processed')
LOG_PATH = os.path.join(DATA_DIR, 'lstm_training_log.json')

# Loading data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))[..., np.newaxis]
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))[..., np.newaxis]
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))[..., np.newaxis]
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

input_shape = X_train.shape[1:]

# Loading best HPs
with open(os.path.join(DATA_DIR, 'lstm_tuning_log.json')) as f:
    best_hp = json.load(f)
    
best_hps = best_hp['best_hyperparameters']
print("Best HPs:", best_hps)

# Build model
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(best_hps['units']),
    layers.Dropout(best_hps['dropout']),
    layers.Dense(12)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_hps['learning_rate']),
    loss='mse',
    metrics=['mae']
)


# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

# Train
start_time = time.time()
with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
end_time = time.time()

# Saving training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(DATA_DIR, 'LSTM_Training_History.csv'))

# Plot for training curve
plt.figure()
plt.plot(hist_df['loss'], label='Train Loss')
plt.plot(hist_df['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, 'LSTM_Training_Curve.png'))
plt.close()

# Evaluation on test set
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(mse)

test_r2 = r2_score(y_test, y_pred)

# Model size
SAVEDMODEL_PATH = os.path.join(DATA_DIR, 'Final_LSTM_Model')
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
gpus = GPUtil.getGPUs()

# Final Logging
log_data = {
    "model_name": "LSTM_Final",
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
    "units": best_hps["units"],
    "dropout": best_hps["dropout"],
    "learning_rate": best_hps["learning_rate"],
    "system_info": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "gpu_used": gpus[0].name if gpus else "None"
    }
}


with open(os.path.join(DATA_DIR, 'LSTM_Final_Log.json'), 'w') as f:
    json.dump(log_data, f, indent=4)

print("LSTM model training and evaluation complete.")

# Saving final prediction CSV
np.savetxt(os.path.join(DATA_DIR, 'LSTM_y_pred.csv'), y_pred, delimiter=",")

