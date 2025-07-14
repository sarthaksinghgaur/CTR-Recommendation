import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("PHASE 3: DATA MERGING AND PREPROCESSING")

print("Loading engineered features...")
train_feat = pd.read_parquet('train_data.parquet')
test_feat = pd.read_parquet('test_data.parquet')
add_event = pd.read_parquet('add_event.parquet')
add_trans = pd.read_parquet('add_trans.parquet')
offer_metadata = pd.read_parquet('offer_metadata.parquet')

print("\n1. MERGING ALL FEATURES")

train_feat['id2'] = train_feat['id2'].astype(int)
test_feat['id2'] = test_feat['id2'].astype(int)
train_feat['id3'] = train_feat['id3'].astype(int)
test_feat['id3'] = test_feat['id3'].astype(int)

print("Creating user features...")
add_trans['f370'] = pd.to_datetime(add_trans['f370'], errors='coerce')

user_rfm = add_trans.groupby('id2').agg(
    trans_count=('f370', 'count'),
    trans_amount_sum=('f367', 'sum'),
    trans_amount_mean=('f367', 'mean'),
    last_trans_time=('f370', 'max')
).reset_index()
user_rfm['days_since_last_trans'] = (pd.Timestamp.now() - user_rfm['last_trans_time']).dt.days

user_cat = add_trans.groupby(['id2', 'f368']).size().reset_index(name='cat_count')
user_pref_cat = user_cat.loc[user_cat.groupby('id2')['cat_count'].idxmax()][['id2', 'f368']]
user_pref_cat = user_pref_cat.rename(columns={'f368': 'most_freq_category'})

user_ind = add_trans.groupby(['id2', 'id8']).size().reset_index(name='ind_count')
user_pref_ind = user_ind.loc[user_ind.groupby('id2')['ind_count'].idxmax()][['id2', 'id8']]
user_pref_ind = user_pref_ind.rename(columns={'id8': 'most_freq_industry'})

print("Creating offer features...")
offer_feats = offer_metadata[['id3', 'f376', 'id10']].copy()
offer_feats['id3'] = offer_feats['id3'].astype(int)

add_event['is_click'] = add_event['id7'].notnull().astype(int)
offer_stats = add_event.groupby('id3').agg(
    offer_impressions=('id4', 'count'),
    offer_clicks=('is_click', 'sum')
).reset_index()
offer_stats['id3'] = offer_stats['id3'].astype(int)
offer_stats['offer_ctr'] = offer_stats['offer_clicks'] / offer_stats['offer_impressions']

offer_feats = offer_feats.merge(offer_stats, on='id3', how='left')

print("Creating user-offer interaction features...")
user_stats = add_event.groupby('id2').agg(
    user_impressions=('id4', 'count'),
    user_clicks=('is_click', 'sum')
).reset_index()
user_stats['user_ctr'] = user_stats['user_clicks'] / user_stats['user_impressions']

def merge_all_features(df):
    df = df.merge(user_rfm.drop('last_trans_time', axis=1), on='id2', how='left')
    df = df.merge(user_pref_cat, on='id2', how='left')
    df = df.merge(user_pref_ind, on='id2', how='left')
    df = df.merge(offer_feats, on='id3', how='left')
    df = df.merge(user_stats[['id2', 'user_ctr']], on='id2', how='left')
    df['industry_match'] = (df['most_freq_industry'] == df['id10']).astype(int)
    df['id4_dt'] = pd.to_datetime(df['id4'], errors='coerce')
    df['day_of_week'] = df['id4_dt'].dt.dayofweek
    df['hour_of_day'] = df['id4_dt'].dt.hour
    return df

print("Merging features into train set...")
train_merged = merge_all_features(train_feat)
print("Merging features into test set...")
test_merged = merge_all_features(test_feat)

print(f"Train shape after merging: {train_merged.shape}")
print(f"Test shape after merging: {test_merged.shape}")

print("\n2. HANDLING MISSING VALUES")

print("Missing values analysis:")
missing_train = train_merged.isnull().sum()
missing_test = test_merged.isnull().sum()

print("\nTrain missing values (top 10):")
print(missing_train[missing_train > 0].head(10))
print("\nTest missing values (top 10):")
print(missing_test[missing_test > 0].head(10))

print("\nApplying imputation strategy...")

numerical_features = ['trans_count', 'trans_amount_sum', 'trans_amount_mean', 
                     'days_since_last_trans', 'offer_impressions', 'offer_clicks', 
                     'offer_ctr', 'user_impressions', 'user_clicks', 'user_ctr', 'f376']

for col in numerical_features:
    if col in train_merged.columns:
        if 'ctr' in col or 'f376' in col:
            fill_value = train_merged[col].median()
        else:
            fill_value = 0
        
        train_merged[col] = train_merged[col].fillna(fill_value)
        test_merged[col] = test_merged[col].fillna(fill_value)

categorical_features = ['most_freq_category', 'most_freq_industry', 'id10']
for col in categorical_features:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna('Unknown')
        test_merged[col] = test_merged[col].fillna('Unknown')

temporal_features = ['day_of_week', 'hour_of_day']
for col in temporal_features:
    if col in train_merged.columns:
        fill_value = train_merged[col].mode()[0]
        train_merged[col] = train_merged[col].fillna(fill_value)
        test_merged[col] = test_merged[col].fillna(fill_value)

print("Missing values after imputation:")
print(f"Train: {train_merged.isnull().sum().sum()}")
print(f"Test: {test_merged.isnull().sum().sum()}")

print("\n3. ENCODING CATEGORICAL FEATURES")

categorical_cols = train_merged.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

label_encoders = {}
ordinal_cats = ['most_freq_category', 'most_freq_industry', 'id10']

for col in ordinal_cats:
    if col in train_merged.columns:
        le = LabelEncoder()
        combined_values = pd.concat([train_merged[col], test_merged[col]]).unique()
        le.fit(combined_values)
        
        train_merged[f'{col}_encoded'] = le.transform(train_merged[col])
        test_merged[f'{col}_encoded'] = le.transform(test_merged[col])
        label_encoders[col] = le
        
        print(f"Label encoded {col} -> {col}_encoded")

if 'id6' in train_merged.columns:
    print("One-hot encoding id6 (Placement ID)...")
    
    all_id6_values = pd.concat([train_merged['id6'], test_merged['id6']]).unique()
    
    for value in all_id6_values:
        if pd.notna(value):
            col_name = f'id6_{value}'
            train_merged[col_name] = (train_merged['id6'] == value).astype(int)
            test_merged[col_name] = (test_merged['id6'] == value).astype(int)
    
    train_merged = train_merged.drop('id6', axis=1)
    test_merged = test_merged.drop('id6', axis=1)

if 'id3' in train_merged.columns:
    print("Creating target encoding for offer_id (id3)...")
    
    train_merged['offer_id_target_encoded'] = train_merged['offer_ctr']
    test_merged['offer_id_target_encoded'] = test_merged['offer_ctr']

print("\n4. FINAL CLEANUP AND SUMMARY")

columns_to_drop = ['id4_dt', 'id4', 'id5']
for col in columns_to_drop:
    if col in train_merged.columns:
        train_merged = train_merged.drop(col, axis=1)
    if col in test_merged.columns:
        test_merged = test_merged.drop(col, axis=1)

for col in ordinal_cats:
    if col in train_merged.columns:
        train_merged = train_merged.drop(col, axis=1)
        test_merged = test_merged.drop(col, axis=1)

print(f"\nFinal shapes:")
print(f"Train: {train_merged.shape}")
print(f"Test: {test_merged.shape}")

print(f"\nFinal data types:")
print(train_merged.dtypes.value_counts())

print(f"\nSample of final features:")
feature_sample = ['id1', 'id2', 'id3', 'y', 'trans_count', 'trans_amount_sum', 
                 'offer_ctr', 'user_ctr', 'industry_match', 'day_of_week', 'hour_of_day']
available_features = [f for f in feature_sample if f in train_merged.columns]
print(train_merged[available_features].head())

print("\nSaving preprocessed data...")
train_merged.to_parquet('train_preprocessed.parquet', index=False)
test_merged.to_parquet('test_preprocessed.parquet', index=False)

print("\nPhase 3 preprocessing completed!")