# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 22:02:31 2025

@author: mohsi
"""


import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import os

# -----------------------------
# Step 1: Paths and Setup
# -----------------------------
BASE_PATH = r"G:\My Drive\EdgeMeter_AI\Data\processed"
INPUT_FILE = os.path.join(BASE_PATH, "clean_smart_meter_data.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "smart_meter_full_enriched.csv")

# -----------------------------
# Step 2: Load and Filter Smart Meter Data
# -----------------------------
df = pd.read_csv(INPUT_FILE, parse_dates=['DateTime'])
df = df.sort_values(['MeterID', 'DateTime'])

# Drop any rows with missing values (if any sneak in)
df = df.dropna(subset=['Consumption'])

# -----------------------------
# Step 3: Create Calendar Features
# -----------------------------
calendar_df = pd.DataFrame({
    'DateTime': pd.date_range(start='2013-01-01', end='2013-12-31 23:30:00', freq='30min')
})

uk_holidays = holidays.UnitedKingdom(years=2013)
calendar_df['is_uk_bank_holiday'] = calendar_df['DateTime'].dt.date.isin(set(uk_holidays.keys()))
calendar_df['day_of_week'] = calendar_df['DateTime'].dt.dayofweek
calendar_df['is_weekend'] = calendar_df['day_of_week'] >= 5
calendar_df['month'] = calendar_df['DateTime'].dt.month
calendar_df['day'] = calendar_df['DateTime'].dt.day
calendar_df['hour'] = calendar_df['DateTime'].dt.hour
calendar_df['minute'] = calendar_df['DateTime'].dt.minute
calendar_df['is_friday'] = calendar_df['day_of_week'] == 4
calendar_df['is_christmas_week'] = (calendar_df['month'] == 12) & (calendar_df['day'].between(24, 31))
calendar_df['is_halloween_week'] = (
    ((calendar_df['month'] == 10) & (calendar_df['day'] >= 28)) |
    ((calendar_df['month'] == 11) & (calendar_df['day'] <= 3))
)
calendar_df['time_segment'] = pd.cut(calendar_df['hour'],
                                     bins=[-1, 5, 11, 17, 21, 23],
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night'])

# -----------------------------
# Step 4: Merge Calendar Info
# -----------------------------
df_merged = pd.merge(df, calendar_df, on='DateTime', how='left')

# Drop rows with calendar merge failure (should be 0 after fix)
df_merged.dropna(subset=['day_of_week'], inplace=True)

# Ensure correct dtypes
bool_cols = ['is_uk_bank_holiday', 'is_weekend', 'is_friday', 'is_christmas_week', 'is_halloween_week']
for col in bool_cols:
    df_merged[col] = df_merged[col].astype(bool)

df_merged['day_of_week'] = df_merged['day_of_week'].astype(int)
df_merged['month'] = df_merged['month'].astype(int)
df_merged['day'] = df_merged['day'].astype(int)
df_merged['hour'] = df_merged['hour'].astype(int)
df_merged['minute'] = df_merged['minute'].astype(int)
df_merged['time_segment'] = df_merged['time_segment'].astype('category')

# -----------------------------
# Step 5: Add Usage Bins (0=Low to 4=High)
# -----------------------------
total_kwh = df_merged.groupby('MeterID')['Consumption'].sum().reset_index()
total_kwh.columns = ['MeterID', 'total_kwh']
total_kwh['usage_bin'] = pd.qcut(total_kwh['total_kwh'], q=5, labels=False)
df_merged = df_merged.merge(total_kwh[['MeterID', 'usage_bin']], on='MeterID', how='left')

# -----------------------------
# Step 6: Add Lag and Rolling Features
# -----------------------------
df_merged = df_merged.sort_values(['MeterID', 'DateTime'])
df_merged['Lag_1'] = df_merged.groupby('MeterID')['Consumption'].shift(1)
df_merged['Lag_2'] = df_merged.groupby('MeterID')['Consumption'].shift(2)
df_merged['Lag_48'] = df_merged.groupby('MeterID')['Consumption'].shift(48)
df_merged['RollingMean_3'] = df_merged.groupby('MeterID')['Consumption'].rolling(window=3).mean().reset_index(0, drop=True)
df_merged['is_high_usage'] = df_merged['Consumption'] > df_merged.groupby('MeterID')['Consumption'].transform('mean')

# -----------------------------
# Step 7: Save Final Output
# -----------------------------
df_merged.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Enriched dataset saved to:\n{OUTPUT_FILE}")
print(f"📏 Final shape: {df_merged.shape}")

df = df_merged
print("✅ Loaded shape:", df.shape)


######## splitting data into train, val and test 
from sklearn.model_selection import train_test_split
PROCESSED_DIR = r'C:/Users/mohsi/OneDrive/Documents/GitHub/EdgeMeter_AI/V2'

df = df.sort_values(["MeterID", "DateTime"])

# === Step 1: Summarize Unique MeterID & usage_bin ===
meter_summary = df[['MeterID', 'usage_bin']].drop_duplicates()
print(f"✅ Unique meters found: {meter_summary.shape[0]}")

# === Step 2: Pick Demo Households ===
low_house = meter_summary[meter_summary['usage_bin'] == 0].sample(1, random_state=42)
med_house = meter_summary[meter_summary['usage_bin'] == 2].sample(1, random_state=42)
high_house = meter_summary[meter_summary['usage_bin'] == 4].sample(1, random_state=42)

demo_households = pd.concat([low_house, med_house, high_house])
demo_ids = demo_households['MeterID'].tolist()

df_demo = df[df['MeterID'].isin(demo_ids)].copy()
df_demo.to_csv(os.path.join(PROCESSED_DIR, 'demo_households.csv'), index=False)

# === Step 3: Filter Out Demo Households ===
filtered_summary = meter_summary[~meter_summary['MeterID'].isin(demo_ids)]

# === Step 4: Stratified Split by usage_bin ===
train_ids, temp_ids = train_test_split(
    filtered_summary,
    test_size=0.3,
    stratify=filtered_summary['usage_bin'],
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=(2/3),  # test = 20%, val = 10%
    stratify=temp_ids['usage_bin'],
    random_state=42
)

# === Step 5: Create Full Split DataFrames ===
train_data = df[df['MeterID'].isin(train_ids['MeterID'])].copy()
val_data   = df[df['MeterID'].isin(val_ids['MeterID'])].copy()
test_data  = df[df['MeterID'].isin(test_ids['MeterID'])].copy()

# Save them
train_data.to_csv(os.path.join(PROCESSED_DIR, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(PROCESSED_DIR, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIR, 'test_data.csv'), index=False)

# === Sanity: Ensure No Leaks ===
assert not any(id_ in demo_ids for id_ in train_ids['MeterID'])
assert not any(id_ in demo_ids for id_ in val_ids['MeterID'])
assert not any(id_ in demo_ids for id_ in test_ids['MeterID'])

# === Save Summary CSVs ===
train_ids.to_csv(os.path.join(PROCESSED_DIR, 'train_summary.csv'), index=False)
val_ids.to_csv(os.path.join(PROCESSED_DIR, 'val_summary.csv'), index=False)
test_ids.to_csv(os.path.join(PROCESSED_DIR, 'test_summary.csv'), index=False)
demo_households.to_csv(os.path.join(PROCESSED_DIR, 'demo_summary.csv'), index=False)

# === Report ===
print(f" Demo households saved: {demo_ids}")
print(f" Train set size: {train_data.shape}")
print(f" Validation set size: {val_data.shape}")
print(f" Test set size: {test_data.shape}")

# confirming count of meters per bin for each split

print(" Meter count by usage_bin:")

for name, df_ids in {
    'Train': train_ids,
    'Validation': val_ids,
    'Test': test_ids,
    'Demo': demo_households
}.items():
    counts = df_ids['usage_bin'].value_counts().sort_index()
    print(f"\n{name} split:")
    print(counts)
