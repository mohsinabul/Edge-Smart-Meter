# Dataset Information

# Data Folder — EdgeMeterAI

This folder contains all raw and processed datasets used in the **EdgeMeterAI** project. The structure ensures data is well-organized for reproducibility, collaboration, and deployment in both training and real-time edge simulation.

---

## Overview of Data Workflow

1. **Raw Data Download**  
   - The raw smart meter dataset is automatically downloaded from Google Drive using `gdown`.  
   - File: `raw/lcl_data.csv`

2. **Cleaning & Reshaping**
   - Removed households with missing values to ensure data quality.
   - Converted from wide format (1 meter per column) to long format:  
     Each row = `DateTime`, `MeterID`, `Consumption`

3. **Saving Cleaned Data**
   - Output file: `processed/clean_smart_meter_data.csv`

4. **Demo Households for Simulation**
   - Selected 3 households representing:
     - Low usage
     - Medium usage
     - High usage
   - Purpose: These will be used later to simulate smart meters in real-time.
   - File: `processed/demo_households.csv`  
   - IDs saved in: `processed/demo_summary.csv`

5. **Train / Validation / Test Split**
   - Performed stratified splitting based on total annual energy consumption.
   - Ensured balanced representation across usage bins (low to high).
   - Saved full datasets per split:
     - `processed/train_data.csv`
     - `processed/val_data.csv`
     - `processed/test_data.csv`
   - Saved meter summaries:
     - `processed/train_summary.csv`
     - `processed/val_summary.csv`
     - `processed/test_summary.csv`

---

## Folder Structure (G drive)

Data/
│
├── raw/
│ └── lcl_data.csv ← Raw downloaded dataset
│
├── processed/
│ ├── clean_smart_meter_data.csv ← Fully cleaned + reshaped dataset
│ ├── demo_households.csv ← Sample households for smart meter simulation
│ ├── demo_summary.csv ← Summary stats for demo meters
│ ├── train_data.csv ← Full train set (MeterID, DateTime, Consumption)
│ ├── val_data.csv ← Full validation set
│ ├── test_data.csv ← Full test set
│ ├── train_summary.csv ← Summary of meter usage levels in train
│ ├── val_summary.csv ← Summary of meter usage levels in val
│ └── test_summary.csv ← Summary of meter usage levels in test


---

## Notes

- **Large files**: These datasets are excluded from version control (GitHub) due to their size. They are synced via Google Drive.
- **Sync location**: `G:\My Drive\EdgeMeterAI\Data`
- **Safe for Automation**: Scripted with paths and logic that support reproducibility and CI/CD workflows.

---

## Next Steps

➡️ This structured and split data is now ready to be:
- Used in EDA (`02_eda.py`)
- Transformed into windowed sequences for model training
- Fed into on-device edge simulators for real-time forecasting

---

________Maintained by: Abul Mohsin