import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("PHASE 2: COMPREHENSIVE FEATURE ENGINEERING")

print("Loading data...")
train = pd.read_parquet('train_data.parquet')
test = pd.read_parquet('test_data.parquet')
add_event = pd.read_parquet('add_event.parquet')
add_trans = pd.read_parquet('add_trans.parquet')
offer_metadata = pd.read_parquet('offer_metadata.parquet')

print("\n1. USER FEATURES (from add_trans.parquet)")

add_trans['f370'] = pd.to_datetime(add_trans['f370'], errors='coerce')

user_rfm = add_trans.groupby('id2').agg(
    trans_count = ('f370', 'count'),
    trans_amount_sum = ('f367', 'sum'),
    trans_amount_mean = ('f367', 'mean'),
    last_trans_time = ('f370', 'max')
).reset_index()
user_rfm['days_since_last_trans'] = (datetime.now() - user_rfm['last_trans_time']).dt.days

user_cat = add_trans.groupby(['id2', 'f368']).size().reset_index(name='cat_count')
user_pref_cat = user_cat.loc[user_cat.groupby('id2')['cat_count'].idxmax()][['id2', 'f368']]
user_pref_cat = user_pref_cat.rename(columns={'f368': 'most_freq_category'})

user_ind = add_trans.groupby(['id2', 'id8']).size().reset_index(name='ind_count')
user_pref_ind = user_ind.loc[user_ind.groupby('id2')['ind_count'].idxmax()][['id2', 'id8']]
user_pref_ind = user_pref_ind.rename(columns={'id8': 'most_freq_industry'})

def merge_user_features(df):
    df = df.merge(user_rfm.drop('last_trans_time', axis=1), on='id2', how='left')
    df = df.merge(user_pref_cat, on='id2', how='left')
    df = df.merge(user_pref_ind, on='id2', how='left')
    return df

print("\n2. OFFER FEATURES (from offer_metadata & add_event)")

offer_feats = offer_metadata[['id3', 'f376', 'id10']].copy()
offer_feats['id3'] = offer_feats['id3'].astype(int)

add_event['is_click'] = add_event['id7'].notnull().astype(int)
offer_stats = add_event.groupby('id3').agg(
    offer_impressions = ('id4', 'count'),
    offer_clicks = ('is_click', 'sum')
).reset_index()
offer_stats['id3'] = offer_stats['id3'].astype(int)
offer_stats['offer_ctr'] = offer_stats['offer_clicks'] / offer_stats['offer_impressions']

offer_feats = offer_feats.merge(offer_stats, on='id3', how='left')

print("\n3. USER-OFFER INTERACTION FEATURES")

user_stats = add_event.groupby('id2').agg(
    user_impressions = ('id4', 'count'),
    user_clicks = ('is_click', 'sum')
).reset_index()
user_stats['user_ctr'] = user_stats['user_clicks'] / user_stats['user_impressions']

train['id2'] = train['id2'].astype(int)
test['id2'] = test['id2'].astype(int)
train['id3'] = train['id3'].astype(int)
test['id3'] = test['id3'].astype(int)

def merge_all_features(df):
    df = merge_user_features(df)
    df = df.merge(offer_feats, on='id3', how='left')
    df = df.merge(user_stats[['id2', 'user_ctr']], on='id2', how='left')
    df['industry_match'] = (df['most_freq_industry'] == df['id10']).astype(int)
    df['id4_dt'] = pd.to_datetime(df['id4'], errors='coerce')
    df['day_of_week'] = df['id4_dt'].dt.dayofweek
    df['hour_of_day'] = df['id4_dt'].dt.hour
    return df

print("Merging features into train set...")
train_feat = merge_all_features(train)
print("Merging features into test set...")
test_feat = merge_all_features(test)

print("\nTrain features info:")
print(train_feat.info())
print("\nTrain features sample:")
print(train_feat.head())

print("\nTest features info:")
print(test_feat.info())
print("\nTest features sample:")
print(test_feat.head())

print("\nPhase 2 feature engineering completed!")