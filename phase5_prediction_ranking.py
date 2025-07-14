import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("PHASE 5: PREDICTION AND OFFER RANKING")

print("Loading trained model and test data...")
model = lgb.Booster(model_file='lgbm_ctr_model.txt')
test_data = pd.read_parquet('test_preprocessed.parquet')

print(f"Test data shape: {test_data.shape}")

print("\n1. PREPARING TEST FEATURES")

exclude_cols = ['id1', 'id2', 'id3', 'id4', 'id5']
feature_cols = [col for col in test_data.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

X_test = test_data[feature_cols]

print("Converting feature columns to numeric...")
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_test = X_test.fillna(0)

print(f"Test feature matrix shape: {X_test.shape}")

print("\n2. MAKING PREDICTIONS")

print("Generating click probability predictions...")
test_predictions = model.predict(X_test)

print(f"Prediction statistics:")
print(f"Min probability: {test_predictions.min():.6f}")
print(f"Max probability: {test_predictions.max():.6f}")
print(f"Mean probability: {test_predictions.mean():.6f}")
print(f"Std probability: {test_predictions.std():.6f}")

print("\n3. CREATING PREDICTION DATAFRAME")

predictions_df = pd.DataFrame({
    'id1': test_data['id1'],
    'id2': test_data['id2'],
    'id3': test_data['id3'],
    'predicted_ctr': test_predictions
})

print("Predictions dataframe sample:")
print(predictions_df.head(10))

print("\n4. RANKING OFFERS BY PREDICTED CTR")

print("Grouping by customer (id2) and ranking offers...")
ranked_offers = predictions_df.groupby('id2').apply(
    lambda x: x.sort_values('predicted_ctr', ascending=False)
).reset_index(drop=True)

ranked_offers['rank'] = ranked_offers.groupby('id2').cumcount() + 1

print("Ranked offers sample:")
print(ranked_offers.head(20))

print("\n5. CREATING FINAL SUBMISSION")

print("Loading submission template...")
try:
    submission_template = pd.read_csv('submission_template.csv')
    print(f"Submission template shape: {submission_template.shape}")
    print("Submission template columns:", submission_template.columns.tolist())
except FileNotFoundError:
    print("Submission template not found. Creating from scratch...")
    submission_template = pd.DataFrame(columns=['id1', 'predicted_ctr'])

print("Preparing final submission...")
final_submission = predictions_df[['id1', 'predicted_ctr']].copy()

if len(submission_template) > 0:
    final_submission = final_submission.merge(
        submission_template[['id1']], 
        on='id1', 
        how='inner'
    )

print(f"Final submission shape: {final_submission.shape}")
print("Final submission sample:")
print(final_submission.head(10))

print("\n6. SAVING OUTPUTS")

print("Saving final submission...")
final_submission.to_csv('final_submission.csv', index=False)

print("Saving detailed predictions...")
detailed_predictions = ranked_offers.copy()
detailed_predictions.to_csv('detailed_predictions.csv', index=False)

print("Saving summary statistics...")
summary_stats = {
    'total_predictions': len(predictions_df),
    'unique_customers': predictions_df['id2'].nunique(),
    'unique_offers': predictions_df['id3'].nunique(),
    'mean_predicted_ctr': predictions_df['predicted_ctr'].mean(),
    'median_predicted_ctr': predictions_df['predicted_ctr'].median(),
    'std_predicted_ctr': predictions_df['predicted_ctr'].std(),
    'min_predicted_ctr': predictions_df['predicted_ctr'].min(),
    'max_predicted_ctr': predictions_df['predicted_ctr'].max()
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv('prediction_summary.csv', index=False)

print("\n7. OUTPUT SUMMARY")

print("Files created:")
print("  - final_submission.csv: Submission file for evaluation")
print("  - detailed_predictions.csv: Detailed predictions with rankings")
print("  - prediction_summary.csv: Summary statistics")

print(f"\nSubmission statistics:")
print(f"  Total predictions: {len(final_submission)}")
print(f"  Unique customers: {final_submission['id1'].nunique()}")
print(f"  Mean predicted CTR: {final_submission['predicted_ctr'].mean():.6f}")
print(f"  Median predicted CTR: {final_submission['predicted_ctr'].median():.6f}")

print("\nSample of top-ranked offers per customer:")
top_offers = ranked_offers[ranked_offers['rank'] == 1].head(10)
print(top_offers[['id2', 'id3', 'predicted_ctr', 'rank']])

print("\nPhase 5 prediction and ranking completed!")