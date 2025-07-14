# CTR Prediction Pipeline

A comprehensive machine learning pipeline for Click-Through Rate (CTR) prediction and offer ranking.

## Requirements

- Python 3.8 or higher

## Quick Start

1. Ensure all data files are in the project directory with a venv initialised
2. Run the complete pipeline:
   ```bash
   python run_pipeline.py
   ```

## Pipeline Phases

### Phase 1: Data Loading and EDA
- Loads all parquet and CSV files
- Converts timestamp columns to datetime
- Analyzes missing values and data distributions
- Defines target variable from click events

### Phase 2: Feature Engineering
- **User Features**: RFM metrics, transaction history, preferred categories/industries
- **Offer Features**: Discount rates, historical CTR, popularity metrics
- **Interaction Features**: User-offer matching, temporal patterns, industry alignment

### Phase 3: Data Preprocessing
- Merges all engineered features
- Handles missing values with appropriate strategies
- Encodes categorical variables (label encoding, one-hot encoding)
- Applies target encoding for high-cardinality features

### Phase 4: Model Training
- Uses time-based validation split (80% train, 20% validation)
- Trains LightGBM classifier with class imbalance handling
- Evaluates model performance with AUC metric
- Generates feature importance analysis

### Phase 5: Prediction and Ranking
- Generates click probability predictions for test set
- Ranks offers per customer by predicted CTR
- Creates submission files in required format

## Output Files

- `final_submission.csv` - Main submission file for evaluation
- `detailed_predictions.csv` - Detailed predictions with rankings
- `lgbm_ctr_model.txt` - Trained LightGBM model
- `feature_importance.png` - Feature importance visualization
- `training_predictions.csv` - Training set predictions for analysis
- `prediction_summary.csv` - Summary statistics

## Model Performance

The pipeline typically achieves:
- Training AUC: ~0.95
- Validation AUC: ~0.89
- Handles highly imbalanced data (~1.92% click rate)

## Key Features

- **Comprehensive Feature Engineering**: RFM analysis, behavioral patterns, contextual features
- **Robust Preprocessing**: Missing value handling, categorical encoding, data type management
- **Time-based Validation**: Prevents data leakage with temporal splits
- **Class Imbalance Handling**: Uses scale_pos_weight and appropriate evaluation metrics
- **Production Ready**: Single script execution with comprehensive error handling



## License

This project is provided as-is for a friend.