import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("PHASE 1: DATA LOADING AND EXPLORATORY DATA ANALYSIS")

print("\n1. LOADING DATA FILES...")

print("Loading train_data.parquet...")
train_data = pd.read_parquet('train_data.parquet')
print(f"Train data shape: {train_data.shape}")

print("Loading test_data.parquet...")
test_data = pd.read_parquet('test_data.parquet')
print(f"Test data shape: {test_data.shape}")

print("Loading add_event.parquet...")
add_event = pd.read_parquet('add_event.parquet')
print(f"Add event data shape: {add_event.shape}")

print("Loading add_trans.parquet...")
add_trans = pd.read_parquet('add_trans.parquet')
print(f"Add trans data shape: {add_trans.shape}")

print("Loading offer_metadata.parquet...")
offer_metadata = pd.read_parquet('offer_metadata.parquet')
print(f"Offer metadata shape: {offer_metadata.shape}")

print("Loading data_dictionary.csv...")
data_dictionary = pd.read_csv('data_dictionary.csv')
print(f"Data dictionary shape: {data_dictionary.shape}")

print("\n2. BASIC INFORMATION FOR EACH DATASET")

datasets = {
    'train_data': train_data,
    'test_data': test_data,
    'add_event': add_event,
    'add_trans': add_trans,
    'offer_metadata': offer_metadata,
    'data_dictionary': data_dictionary
}

for name, df in datasets.items():
    print(f"\n{name.upper()}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    print(df.dtypes)
    print(f"First 5 rows:")
    print(df.head())
    print("-" * 30)

print("\n3. CONVERTING TIMESTAMP COLUMNS TO DATETIME")

def convert_timestamps(df, df_name):
    timestamp_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                if sample_values.min() > 1000000000:
                    timestamp_columns.append(col)
    
    print(f"\n{df_name} - Potential timestamp columns: {timestamp_columns}")
    
    for col in timestamp_columns:
        try:
            df[col] = pd.to_datetime(df[col], unit='s')
            print(f"  Converted {col} to datetime")
        except:
            try:
                df[col] = pd.to_datetime(df[col], unit='ms')
                print(f"  Converted {col} to datetime (milliseconds)")
            except:
                print(f"  Could not convert {col} to datetime")
    
    return df

for name, df in datasets.items():
    if name != 'data_dictionary':
        datasets[name] = convert_timestamps(df, name)

print("\n4. MISSING VALUES ANALYSIS")

for name, df in datasets.items():
    print(f"\n{name.upper()} - Missing Values:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found")

print("\n5. TARGET VARIABLE ANALYSIS FROM ADD_EVENT DATA")

print("Add event data columns:")
print(add_event.columns.tolist())

print("\nAdd event data sample:")
print(add_event.head(10))

if 'id4' in add_event.columns:
    print(f"\nUnique impressions (id4): {add_event['id4'].nunique()}")
    
if 'id7' in add_event.columns:
    print(f"Unique click timestamps (id7): {add_event['id7'].nunique()}")
    
    clicks = add_event[add_event['id7'].notna()]
    impressions = add_event['id4'].nunique()
    total_clicks = len(clicks)
    
    print(f"\nTotal impressions: {impressions}")
    print(f"Total clicks: {total_clicks}")
    print(f"Click-through rate: {total_clicks/impressions:.4f} ({total_clicks/impressions*100:.2f}%)")
    print(f"Class imbalance ratio (clicks:non-clicks): {total_clicks}:{impressions-total_clicks}")

print("\n6. ADDITIONAL INSIGHTS")

print("Checking for common keys between datasets...")

for name, df in datasets.items():
    if name != 'data_dictionary':
        print(f"\n{name} - Sample values from first few columns:")
        for col in df.columns[:5]:
            print(f"  {col}: {df[col].dtype}, unique values: {df[col].nunique()}")

print("\nPhase 1 EDA completed!")