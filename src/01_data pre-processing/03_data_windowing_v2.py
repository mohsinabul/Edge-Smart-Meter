# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5 2025
@author: mohsi
"""

import os
import numpy as np
import pandas as pd

# Paths
BASE_PATH = r"C:\Users\mohsi\OneDrive\Documents\GitHub\EdgeMeter_AI\V2"
os.makedirs(BASE_PATH, exist_ok=True)

split_files = {
    "train": "train_data.csv",
    "val": "val_data.csv",
    "test": "test_data.csv",
    "demo": "demo_households.csv"
}

# Feature Columns for Windowing
features = [
    "Consumption", "Lag_1", "Lag_2", "Lag_48", "RollingMean_3",
    "hour", "day_of_week", "month", "is_weekend", "is_uk_bank_holiday",
    "usage_bin"
]

# Windowing Function
def create_sliding_windows(df, input_steps=48, output_steps=12):
    X, y = [], []
    for meter_id in df["MeterID"].unique():
        sub_df = df[df["MeterID"] == meter_id].sort_values("DateTime")
        values_X = sub_df[features].fillna(method="ffill").fillna(method="bfill").values
        values_y = sub_df["Consumption"].values
        for i in range(len(values_X) - input_steps - output_steps):
            X.append(values_X[i : i + input_steps])
            y.append(values_y[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)

# Generating and Saving Dual Windows (48-12 and 96-12)
for window in [(48, 12), (96, 12)]:
    in_steps, out_steps = window
    print(f"\n Processing Window: {in_steps} → {out_steps}")

    for split_name, file_name in split_files.items():
        df = pd.read_csv(os.path.join(BASE_PATH, file_name), parse_dates=["DateTime"])
        df = df.sort_values(["MeterID", "DateTime"])
        
        X, y = create_sliding_windows(df, input_steps=in_steps, output_steps=out_steps)
        
        np.save(os.path.join(BASE_PATH, f"X_{split_name}_{in_steps}_{out_steps}.npy"), X)
        np.save(os.path.join(BASE_PATH, f"y_{split_name}_{in_steps}_{out_steps}.npy"), y)
        
        print(f"{split_name.upper()} | X: {X.shape}, y: {y.shape} → saved as X_{split_name}_{in_steps}_{out_steps}.npy")
