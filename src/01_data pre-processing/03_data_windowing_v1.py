# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 20:01:28 2025

@author: abul mohsin
"""

import numpy as np
import pandas as pd

# Defining a window function
def create_sliding_windows(df, input_steps=48, output_steps=12):

    X, y = [], []

    meters = df['MeterID'].unique()

    for meter in meters:
        meter_df = df[df['MeterID'] == meter].sort_values('DateTime')

        values = meter_df['Consumption'].values

        for i in range(len(values) - input_steps - output_steps):
            X.append(values[i : i + input_steps])
            y.append(values[i + input_steps : i + input_steps + output_steps])

    return np.array(X), np.array(y)

# Loading split datasets
train_df = pd.read_csv(r'G:\My Drive\EdgeMeter_AI\Data\processed\train_data.csv')
val_df   = pd.read_csv(r'G:\My Drive\EdgeMeter_AI\Data\processed\val_data.csv')
test_df  = pd.read_csv(r'G:\My Drive\EdgeMeter_AI\Data\processed\test_data.csv')

# Converting DateTime
for df in [train_df, val_df, test_df]:
    df['DateTime'] = pd.to_datetime(df['DateTime'])

# Creating windows
X_train, y_train = create_sliding_windows(train_df, input_steps=48, output_steps=12)
X_val, y_val     = create_sliding_windows(val_df, input_steps=48, output_steps=12)
X_test, y_test   = create_sliding_windows(test_df, input_steps=48, output_steps=12)


print("Train windows:", X_train.shape, y_train.shape)
print("Validation windows:", X_val.shape, y_val.shape)
print("Test windows:", X_test.shape, y_test.shape)


# Saving as .npy for modeling
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\X_train.npy', X_train)
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\y_train.npy', y_train)
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\X_val.npy', X_val)
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\y_val.npy', y_val)
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\X_test.npy', X_test)
np.save(r'G:\My Drive\EdgeMeter_AI\Data\processed\y_test.npy', y_test)