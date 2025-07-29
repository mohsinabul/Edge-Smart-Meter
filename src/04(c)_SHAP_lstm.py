# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:10:41 2025

@author: mohsi
"""

# SHAP analysis for LSTM model – explains which input time steps affect each of the 12 outputs

import os
import json
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

# Paths
BASE_PATH = r'G:\My Drive\EdgeMeter_AI\Data\processed'
MODEL_PATH = os.path.join(BASE_PATH, 'Final_LSTM_Model.h5')
SHAP_OUT_DIR = os.path.join(BASE_PATH, 'SHAP', 'LSTM')
os.makedirs(SHAP_OUT_DIR, exist_ok=True)

x = np.load(r'G:\My Drive\EdgeMeter_AI\Data\processed\X_test.npy')
print("Shape:", x.shape)
print("First few rows:", x[:3])


# Loading test data
X_test = np.load(os.path.join(BASE_PATH, 'X_test.npy'))[..., np.newaxis]
y_test = np.load(os.path.join(BASE_PATH, 'y_test.npy'))

from keras.src.layers.rnn.lstm import LSTM

def patched_from_config(cls, config):
    config.pop('time_major', None)
    return cls(**config)

LSTM.from_config = classmethod(patched_from_config)


# Loading model
model = load_model(MODEL_PATH, compile=False)
print("LSTM model loaded.")

# different samples for background and for explanation
background = X_test[100:200]
X_shap = X_test[:100]

# Run DeepExplainer
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_shap)

# Saving raw SHAP values
np.save(os.path.join(SHAP_OUT_DIR, 'shap_values.npy'), shap_values)

# Preparing feature names with correct time order: t-47 (oldest) to t-0 (most recent)
feature_names = [f't-{47 - i}' for i in range(X_shap.shape[1])]
features = X_shap.reshape((X_shap.shape[0], X_shap.shape[1]))

# Summary plots for each of the 12 predicted time steps
for output_index in range(12):
    shap.summary_plot(
        shap_values[output_index],
        features=features,
        feature_names=feature_names,
        show=False,
        plot_type='bar'
    )
    plt.title(f'LSTM SHAP Feature Importance – Output {output_index + 1}')
    plt.savefig(os.path.join(SHAP_OUT_DIR, f'shap_bar_output_{output_index + 1}.png'))
    plt.close()

# Waterfall plots for first two samples (for local explanation)
for i in [0, 1]:
    exp = shap.Explanation(
        values=shap_values[0][i],
        base_values=explainer.expected_value[0],
        data=features[i],
        feature_names=feature_names
    )
    shap.plots.waterfall(exp, show=False)
    plt.title(f'LSTM SHAP Waterfall – Sample {i}')
    plt.savefig(os.path.join(SHAP_OUT_DIR, f'shap_waterfall_sample_{i}.png'))
    plt.close()

# Top 10 SHAP features for output step 1 (global importance)
abs_shap = np.abs(shap_values[0]).mean(axis=0)
shap_table = pd.DataFrame({
    'Timestep': feature_names,
    'Mean_Abs_SHAP': abs_shap
})
shap_table.sort_values(by='Mean_Abs_SHAP', ascending=False).head(10).to_csv(
    os.path.join(SHAP_OUT_DIR, 'top10_shap_table.csv'),
    index=False
)

print("SHAP analysis complete for LSTM.")
print(f"Summary plots saved to: {SHAP_OUT_DIR}")
