
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 2025
Exploratory Data Analysis for Smart Meter Dataset
@author: abul mohsin
"""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Paths
base_path = r'G:\My Drive\EdgeMeter_AI\Data\processed'
clean_path = os.path.join(base_path, 'clean_smart_meter_data.csv')
demo_path = os.path.join(base_path, 'demo_households.csv')

# Loading data
df_long = pd.read_csv(clean_path)
df_long['DateTime'] = pd.to_datetime(df_long['DateTime'])
df_long['Hour'] = df_long['DateTime'].dt.hour
df_long['DayOfWeek'] = df_long['DateTime'].dt.dayofweek
df_long['Month'] = df_long['DateTime'].dt.month
df_long['Date'] = df_long['DateTime'].dt.date

# A. Total daily energy usage
plt.figure(figsize=(14, 4))
df_long.groupby('Date')['Consumption'].sum().plot()
plt.title("Total Daily Energy Consumption")
plt.ylabel("kWh")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# B. Histogram of household usage
total_usage = df_long.groupby('MeterID')['Consumption'].sum()
plt.figure(figsize=(10, 5))
sns.histplot(total_usage, bins=50, kde=True)
plt.title("Total Yearly Consumption per Household")
plt.xlabel("Total kWh")
plt.tight_layout()
plt.show()

# C. Average hourly usage (overall)
hourly_avg = df_long.groupby('Hour')['Consumption'].mean()
plt.figure(figsize=(10, 4))
hourly_avg.plot()
plt.title("Average Hourly Energy Usage")
plt.xlabel("Hour of Day")
plt.ylabel("kWh")
plt.tight_layout()
plt.show()

# C2. Weekday vs Weekend patterns
df_long['WeekPart'] = df_long['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
hourly_comparison = df_long.groupby(['Hour', 'WeekPart'])['Consumption'].mean().unstack()
hourly_comparison.plot(figsize=(10, 4), title="Weekday vs Weekend Hourly Usage")
plt.ylabel("kWh")
plt.tight_layout()
plt.show()

# D. Comparison of demo households
demo_df = pd.read_csv(demo_path)
demo_df['DateTime'] = pd.to_datetime(demo_df['DateTime'])
demo_df['Hour'] = demo_df['DateTime'].dt.hour
demo_hourly = demo_df.groupby(['Hour', 'MeterID'])['Consumption'].mean().unstack()
demo_hourly.plot(figsize=(10, 5), title="Hourly Usage of Demo Households")
plt.xlabel("Hour")
plt.ylabel("kWh")
plt.tight_layout()
plt.show()

# E. Household variance
std_usage = df_long.groupby('MeterID')['Consumption'].std()
plt.figure(figsize=(10, 5))
sns.histplot(std_usage, bins=50, kde=True)
plt.title("Household Consumption Variability (STD)")
plt.xlabel("kWh")
plt.tight_layout()
plt.show()

# F. Seasonality trends
monthly_avg = df_long.groupby('Month')['Consumption'].mean()
plt.figure(figsize=(10, 4))
monthly_avg.plot(marker='o')
plt.title("Average Monthly Energy Consumption")
plt.xlabel("Month")
plt.ylabel("kWh")
plt.tight_layout()
plt.show()

# G. Clustering households using optimal K 
hourly_profile = df_long.groupby(['MeterID', 'Hour'])['Consumption'].mean().unstack(fill_value=0)

## Normalizing the hourly profiles
scaler = StandardScaler()
profile_scaled = scaler.fit_transform(hourly_profile)

## Determining optimal K using Elbow and Silhouette
inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(profile_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(profile_scaled, labels))

# Plotting elbow and silhouette
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'bo-')
plt.title("Elbow Method (Inertia)")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'ro-')
plt.title("Silhouette Scores")
plt.xlabel("Number of clusters")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# Choosing best K based on silhouette score
best_k = K_range[sil_scores.index(max(sil_scores))]
print(f"Optimal number of clusters (K): {best_k}")

# Final KMeans with best K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(profile_scaled)

# Confirm unique clusters found
unique_clusters = sorted(set(labels))
print(f"Clusters found: {unique_clusters}")

# Assign and save
hourly_profile['Cluster'] = labels
hourly_profile.reset_index()[['MeterID', 'Cluster']].to_csv(os.path.join(base_path, 'hourly_profile_clusters.csv'), index=False)

# Plotting average profile per cluster
for cluster in unique_clusters:
    plt.figure(figsize=(10, 3))
    cluster_mean = hourly_profile[hourly_profile['Cluster'] == cluster].drop(columns='Cluster').mean()
    cluster_mean.plot()
    plt.title(f"Cluster {cluster}: Avg Hourly Profile")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.tight_layout()
    plt.show()


# H. Outlier detection
daily_usage = df_long.groupby(['MeterID', 'Date'])['Consumption'].sum().reset_index()

# IQR thresholds
q1 = daily_usage.groupby('MeterID')['Consumption'].quantile(0.25)
q3 = daily_usage.groupby('MeterID')['Consumption'].quantile(0.75)
iqr = q3 - q1
thresholds = (q3 + 1.5 * iqr).reset_index()
thresholds.columns = ['MeterID', 'Threshold']

# outliers
daily_usage = pd.merge(daily_usage, thresholds, on='MeterID')
daily_usage['IsOutlier'] = daily_usage['Consumption'] > daily_usage['Threshold']

# Outlier summary
outlier_counts = daily_usage.groupby('MeterID')['IsOutlier'].sum().reset_index(name='n_outliers')
days_count = daily_usage.groupby('MeterID').size().reset_index(name='n_days')
outlier_stats = pd.merge(outlier_counts, days_count, on='MeterID')
outlier_stats['outlier_rate'] = outlier_stats['n_outliers'] / outlier_stats['n_days']

# Total yearly consumption
total_consumption = df_long.groupby('MeterID')['Consumption'].sum().reset_index()
total_consumption.columns = ['MeterID', 'TotalConsumption']

# Merging with outlier stats
outlier_stats = pd.merge(outlier_stats, total_consumption, on='MeterID')

# Saving outlier stats
outlier_stats.to_csv(os.path.join(base_path, 'outlier_stats_per_household.csv'), index=False)

# Loading clusters
cluster_path = os.path.join(base_path, 'hourly_profile_clusters.csv')
clusters = pd.read_csv(cluster_path)

# Merging all
summary_df = pd.merge(outlier_stats, clusters, on='MeterID')

# Plot 1: Scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(data=summary_df, x='TotalConsumption', y='outlier_rate', hue='Cluster', palette='Set2')
plt.title("Outlier Rate vs Total Consumption by Cluster")
plt.xlabel("Total Yearly Consumption (kWh)")
plt.ylabel("Outlier Rate")
plt.tight_layout()
plt.show()

# Plot 2: Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=summary_df, x='Cluster', y='outlier_rate', palette='Set2')
plt.title("Distribution of Outlier Rates by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Outlier Rate")
plt.tight_layout()
plt.show()

# Plot 3: Histogram
plt.figure(figsize=(10, 5))
sns.histplot(outlier_stats['outlier_rate'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Outlier Rates Across Households")
plt.xlabel("Outlier Rate")
plt.ylabel("Number of Households")
plt.tight_layout()
plt.show()

# I. Variability vs Total Consumption Correlation
summary_stats = df_long.groupby('MeterID')['Consumption'].agg(['mean', 'std', 'sum']).reset_index()
summary_stats.rename(columns={'sum': 'total_kwh'}, inplace=True)

# Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(data=summary_stats, x='std', y='total_kwh')
plt.title("Variability (STD) vs Total Yearly Consumption")
plt.xlabel("STD of Consumption")
plt.ylabel("Total kWh")
plt.tight_layout()
plt.show()
