import os
import pandas as pd
from dataset_mimic.mimic_iii_1_4 import *

# Load D_ITEMS.csv to find ITEMID for 'Potassium (Serum)'
d_items = load_D_ITEMS()

# Filter for 'Potassium (Serum)' in LABEL column
potassium_itemid = d_items[d_items['LABEL'] == 'Potassium (serum)']['ITEMID'].values[0]
print(f"ITEMID for 'Potassium (serum)': {potassium_itemid}")

# Load CHARTEVENTS data
measurements = load_CHARTEVENTS()

# Filter measurements for the specific ITEMID
potassium_measurements = measurements[measurements['ITEMID'] == potassium_itemid]

# Extract relevant columns: SUBJECT_ID, VALUENUM, and CHARTTIME
patient_data = potassium_measurements[['SUBJECT_ID', 'VALUENUM', 'CHARTTIME']]

# Select the first VALUENUM for each patient (based on the earliest CHARTTIME)
patient_data = patient_data.sort_values(by=['SUBJECT_ID', 'CHARTTIME'])  # Sort by SUBJECT_ID and CHARTTIME
single_measurement = patient_data.groupby('SUBJECT_ID').first().reset_index()

# Save the single VALUENUM data to a CSV (optional)
single_measurement.to_csv('single_potassium_measurement.csv', index=False)

# Display the first few rows of the single measurement data
print(single_measurement.head())
