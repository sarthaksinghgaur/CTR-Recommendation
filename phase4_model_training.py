import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("PHASE 4: MODEL TRAINING AND VALIDATION")

print("Loading preprocessed data...")
train_data = pd.read_parquet('train_preprocessed.parquet')
test_data = pd.read_parquet('test_preprocessed.parquet')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

print("\n1. PREPARING TARGET VARIABLE")

if 'y' not in train_data.columns:
    print("Target variable 'y' not found. Creating from add_event data...")
    add_event = pd.read_parquet('add_event.parquet')
    
    add_event['is_click'] = add_event['id7'].notnull().astype(int)
    target_mapping = add_event.groupby('id4')['is_click'].max().reset_index()
    target_mapping = target_mapping.rename(columns={'id4': 'id4', 'is_click': 'y'})
    
    original_train = pd.read_parquet('train_data.parquet')
    original_train = original_train.merge(target_mapping, on='id4', how='left')
    original_train['y'] = original_train['y'].fillna(0).astype(int)
    
    train_data = train_data.merge(original_train[['id1', 'y']], on='id1', how='left')
    train_data['y'] = train_data['y'].fillna(0).astype(int)

print(f"Target variable distribution:")
print(train_data['y'].value_counts(normalize=True))

print(f"Target variable dtype: {train_data['y'].dtype}")

if train_data['y'].dtype == 'object':
    print("Converting target variable to int...")
    train_data['y'] = pd.to_numeric(train_data['y'], errors='coerce').fillna(0).astype(int)

print(f"Target variable dtype after conversion: {train_data['y'].dtype}")

print("\n2. FEATURE SELECTION")

exclude_cols = ['id1', 'id2', 'id3', 'y']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")
print(f"Feature columns: {feature_cols[:10]}...")

X = train_data[feature_cols]
y = train_data['y']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target dtype: {y.dtype}")

print("\n3. DATA SPLITTING")

print("Converting feature columns to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)

print("Splitting data randomly (80% for training, 20% for validation)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Training target dtype: {y_train.dtype}")
print(f"Validation target dtype: {y_val.dtype}")

print("\n4. MODEL TRAINING")

print("Setting up LightGBM parameters...")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'scale_pos_weight': 50
}

print("Creating LightGBM datasets...")
train_dataset = lgb.Dataset(X_train, label=y_train)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

print("Training LightGBM model...")
model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

print("\n5. MODEL EVALUATION")

print("Making predictions...")
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print("Calculating AUC scores...")
train_auc = roc_auc_score(y_train, y_train_pred)
val_auc = roc_auc_score(y_val, y_val_pred)

print(f"Training AUC: {train_auc:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

print("\nClassification Report (Validation Set):")
y_val_binary = (y_val_pred > 0.5).astype(int)
print(classification_report(y_val, y_val_binary))

print("\n6. FEATURE IMPORTANCE")

print("Getting feature importance...")
importance = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Top 20 most important features:")
print(feature_importance.head(20))

print("\nPlotting feature importance...")
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n7. SAVING MODEL AND PREDICTIONS")

print("Saving trained model...")
model.save_model('lgbm_ctr_model.txt')

print("Saving training predictions...")
train_indices = X_train.index
val_indices = X_val.index

train_predictions = pd.DataFrame({
    'id1': train_data.loc[train_indices, 'id1'],
    'id2': train_data.loc[train_indices, 'id2'],
    'id3': train_data.loc[train_indices, 'id3'],
    'actual': y_train,
    'predicted_prob': y_train_pred,
    'predicted_class': (y_train_pred > 0.5).astype(int)
})

val_predictions = pd.DataFrame({
    'id1': train_data.loc[val_indices, 'id1'],
    'id2': train_data.loc[val_indices, 'id2'],
    'id3': train_data.loc[val_indices, 'id3'],
    'actual': y_val,
    'predicted_prob': y_val_pred,
    'predicted_class': (y_val_pred > 0.5).astype(int)
})

all_predictions = pd.concat([train_predictions, val_predictions], ignore_index=True)
all_predictions.to_csv('training_predictions.csv', index=False)

print("Training predictions saved to 'training_predictions.csv'")
print("Model saved to 'lgbm_ctr_model.txt'")
print("Feature importance plot saved to 'feature_importance.png'")

print("\nPhase 4 model training completed!")