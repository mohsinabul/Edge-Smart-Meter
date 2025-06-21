# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 17:58:08 2025

@author: abul mohsin
"""
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Defining Google Drive paths
GDRIVE_PATH = r'G:\My Drive\EdgeMeterAI\Data'
RAW_DIR = os.path.join(GDRIVE_PATH, 'raw')
PROCESSED_DIR = os.path.join(GDRIVE_PATH, 'processed')

# Ensuring directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Downloading dataset from G-drive
file_id = '1zurl6uqUXqvtuDYCzHI22W7cSw1bY9Ap'
raw_file_path = os.path.join(RAW_DIR, 'lcl_data.csv')
if not os.path.exists(raw_file_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', raw_file_path, quiet=False)

df = pd.read_csv(raw_file_path)

# Data info
df.head()
df.info()
df.dtypes

# Converting the DateTime column
df['DateTime'] = pd.to_datetime(df['DateTime'])

df.set_index('DateTime', inplace=True)
print(df.index)

# Checking missing values per meter
missing_counts = df.isnull().sum()
print(" Missing values per meter:\n", missing_counts[missing_counts > 0])

# list of meters (columns) with NO missing values
complete_meters = missing_counts[missing_counts == 0].index.tolist()
print(f"Total meters with no missing values: {len(complete_meters)}")

# Creating a new DataFrame with only complete meters
df_clean = df[complete_meters].copy()
df_clean.reset_index(inplace=True)

# Final shape check
print("Clean dataset shape:", df_clean.shape)
print(df_clean.head())

# Converting wide to long format
df_long = df_clean.melt(id_vars='DateTime', var_name='MeterID', value_name='Consumption')

# shape
print("Reshaped to long format:", df_long.shape)
print(df_long.head())

# Saving cleaned DataSet for future use
df_long.to_csv(os.path.join(PROCESSED_DIR, 'clean_smart_meter_data.csv'), index=False)


# Calculating total consumption per household
consumption_summary = df_long.groupby('MeterID')['Consumption'].sum().reset_index()
consumption_summary.rename(columns={'Consumption': 'total_kwh'}, inplace=True)

# Bin households based on usage level (stratification buckets)
consumption_summary['usage_bin'] = pd.qcut(consumption_summary['total_kwh'], q=5, labels=False)

# Picking one from each of low (bin 0), medium (bin 2), high (bin 4)
low_house = consumption_summary[consumption_summary['usage_bin'] == 0].sample(1, random_state=42)
med_house = consumption_summary[consumption_summary['usage_bin'] == 2].sample(1, random_state=42)
high_house = consumption_summary[consumption_summary['usage_bin'] == 4].sample(1, random_state=42)

# Combining selected households
demo_households = pd.concat([low_house, med_house, high_house])
demo_ids = demo_households['MeterID'].tolist()

# Saving their data separately for smart meter simulation
df_demo = df_long[df_long['MeterID'].isin(demo_ids)].copy()
df_demo.to_csv(os.path.join(PROCESSED_DIR, 'demo_households.csv'), index=False)

# Excluding these households from splitting
filtered_consumption_summary = consumption_summary[~consumption_summary['MeterID'].isin(demo_ids)]

# Spliting into train/val/test with stratification
train_ids, temp_ids = train_test_split(
    filtered_consumption_summary,
    test_size=0.3,
    stratify=filtered_consumption_summary['usage_bin'],
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=(2/3),  # test = 20%, val = 10%
    stratify=temp_ids['usage_bin'],
    random_state=42
)

# Saving full data for each split
train_data = df_long[df_long['MeterID'].isin(train_ids['MeterID'])].copy()
val_data   = df_long[df_long['MeterID'].isin(val_ids['MeterID'])].copy()
test_data  = df_long[df_long['MeterID'].isin(test_ids['MeterID'])].copy()

train_data.to_csv(os.path.join(PROCESSED_DIR, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(PROCESSED_DIR, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIR, 'test_data.csv'), index=False)


# Ensuring no demo IDs leaked into other splits
assert not any(id_ in demo_ids for id_ in train_ids['MeterID'])
assert not any(id_ in demo_ids for id_ in val_ids['MeterID'])
assert not any(id_ in demo_ids for id_ in test_ids['MeterID'])

print(f"Demo households saved: {demo_ids}")
print(f"Train set size: {train_data.shape}")
print(f"Validation set size: {val_data.shape}")
print(f"Test set size: {test_data.shape}")

# Saving stratified summary for later analysis
train_ids.to_csv(os.path.join(PROCESSED_DIR, 'train_summary.csv'), index=False)
val_ids.to_csv(os.path.join(PROCESSED_DIR, 'val_summary.csv'), index=False)
test_ids.to_csv(os.path.join(PROCESSED_DIR, 'test_summary.csv'), index=False)
demo_households.to_csv(os.path.join(PROCESSED_DIR, 'demo_summary.csv'), index=False)
