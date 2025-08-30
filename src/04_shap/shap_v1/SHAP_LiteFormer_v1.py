# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 00:05:34 2025

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
MODEL_PATH = r'G:\My Drive\EdgeMeter_AI\Models\Final_LiteFormer_Model.h5'
SHAP_DIR = os.path.join(DATA_PATH, 'SHAP', 'LiteFormer_Kernel')
os.makedirs(SHAP_DIR, exist_ok=True)

# Loading model and test data

def slice_patches(inputs, patch_size=4):
    feat_dim = inputs.shape[-1]
    num_patches = inputs.shape[1] // patch_size
    trimmed = inputs[:, :num_patches * patch_size, :]
    return tf.reshape(trimmed, [-1, num_patches, patch_size * feat_dim])

model = load_model(MODEL_PATH, compile=False, custom_objects={'slice_patches': slice_patches})

X = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
X = X[..., np.newaxis]  
X = X[:600]

# Flatten for SHAP input
X_flat = X.reshape((X.shape[0], -1))  # Shape: (600, 48)
X_sample = X_flat[:50]                # 50 test samples
background = shap.sample(X_flat[50:], 20)  # 20 background samples

# Prediction wrapper for SHAP
def predict_fn(x):
    x_reshaped = x.reshape((x.shape[0], 48, 1))
    return model(x_reshaped).numpy()[:, 0:1]

# SHAP KernelExplainer
explainer = shap.KernelExplainer(predict_fn, background)
shap_vals = explainer.shap_values(X_sample)

# Saving SHAP values
np.save(os.path.join(SHAP_DIR, 'shap_values_kernel.npy'), shap_vals)

# Feature names
feature_names = [f"t-{47 - i}" for i in range(48)]

# SHAP Summary Bar Plot
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_vals[0].T, X_sample, feature_names=feature_names, plot_type='bar', show=False)
plt.title("LiteFormer SHAP – KernelExplainer Output 1", fontsize=14)
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
plt.title("LiteFormer SHAP Waterfall – Sample 0", fontsize=14)
plt.xlabel("SHAP Value Contribution", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, 'shap_waterfall_sample_0.png'), dpi=300)
plt.close()

# Top 10 SHAP Feature Table
mean_vals = np.abs(shap_vals[0].T).mean(axis=0)
df_top = pd.DataFrame({
    'Timestep': feature_names,
    'Mean_Abs_SHAP': mean_vals
})
df_top.sort_values(by='Mean_Abs_SHAP', ascending=False).head(10).to_csv(
    os.path.join(SHAP_DIR, 'top10_shap_table.csv'), index=False
)

print("SHAP analysis complete for LiteFormer.")
print("Outputs saved to:", SHAP_DIR)
