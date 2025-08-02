# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:13:48 2025

@author: abul mohsin
"""

import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import shap
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

# Setting GPU memory growth for stable SHAP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print("Error enabling GPU memory growth:", e)

# Paths
DATA_PATH = r'G:\My Drive\EdgeMeter_AI\Data\processed'
MODEL_PATH = r'G:\My Drive\EdgeMeter_AI\Models\Final_LSTM_Model.h5'
SHAP_DIR = os.path.join(DATA_PATH, 'SHAP', 'LSTM_Kernel')
os.makedirs(SHAP_DIR, exist_ok=True)

def patched_from_config(cls, config):
    config.pop('time_major', None)
    return cls(**config)

LSTM.from_config = classmethod(patched_from_config)


# Loading model and data
model = load_model(MODEL_PATH, compile=False)
X = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
X = X[..., np.newaxis]
X = X[:600]  # Total required: 600

# Flatten for SHAP input
X_flat = X.reshape((X.shape[0], -1))
X_sample = X_flat[:50]
background = shap.sample(X_flat[50:], 20)


# SHAP prediction wrapper
def predict_fn(x):
    x_reshaped = x.reshape((x.shape[0], 48, 1))
    return model(x_reshaped).numpy()[:, 0:1]

# KernelExplainer
explainer = shap.KernelExplainer(predict_fn, background)
shap_vals = explainer.shap_values(X_sample)

# Saving raw SHAP values
np.save(os.path.join(SHAP_DIR, 'shap_values_kernel.npy'), shap_vals)

# Feature names
feature_names = [f"t-{47 - i}" for i in range(48)]

# SHAP Summary Bar Plot
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_vals[0].T, X_sample, feature_names=feature_names, plot_type='bar', show=False)
plt.title("LSTM SHAP – KernelExplainer Output 1", fontsize=14)
plt.xlabel("Mean |SHAP| Value", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'shap_bar_output_1.png'), dpi=300)
plt.close()

# SHAP Waterfall Plot – Sample 0
exp = shap.Explanation(
    values=shap_vals[0].T[0],
    base_values=explainer.expected_value[0],
    data=X_sample[0],
    feature_names=feature_names
)
plt.figure(figsize=(12, 6))
shap.plots.waterfall(exp, show=False)
plt.title("LSTM SHAP Waterfall – Sample 0", fontsize=14)
plt.xlabel("SHAP Value Contribution", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'shap_waterfall_sample_0.png'), dpi=300)
plt.close()

# Top 10 SHAP Features Table
mean_vals = np.abs(shap_vals[0].T).mean(axis=0)
df_top = pd.DataFrame({
    'Timestep': feature_names,
    'Mean_Abs_SHAP': mean_vals
})
df_top.sort_values(by='Mean_Abs_SHAP', ascending=False).head(10).to_csv(
    os.path.join(SHAP_DIR, 'top10_shap_table.csv'), index=False
)

print("LSTM SHAP analysis complete (300/300).")
print("All results saved to:", SHAP_DIR)
