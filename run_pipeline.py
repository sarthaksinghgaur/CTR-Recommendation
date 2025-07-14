#!/usr/bin/env python3
"""
CTR Prediction Pipeline Runner

A complete, self-contained script that sets up the environment and runs the entire
CTR prediction pipeline from data loading to final submission.

Usage:
    python run_pipeline.py

Requirements:
    - Python 3.8+
    - All data files in the same directory
    - Internet connection (for package installation)
"""

import os
import sys
import subprocess
import time
import platform
from datetime import datetime

def print_header(title):
    print(f" {title}")

def print_step(step_num, step_name):
    print(f"\n[STEP {step_num}] {step_name}")

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    print("Installing required packages...")
    
    packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0', 
        'lightgbm>=4.0.0',
        'scikit-learn>=1.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pyarrow>=10.0.0'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True, check=True)
            print(f"{package} installed")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    return True

def check_file_exists(filename, description):
    if not os.path.exists(filename):
        print(f"ERROR: {description} '{filename}' not found!")
        print(f"   Please ensure all data files are in the current directory.")
        return False
    print(f"Found {description}: {filename}")
    return True

def run_script(script_name, description):
    print(f"Running {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"{description} completed successfully!")
        print(f"   Time taken: {time.time() - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"   Error output: {e.stderr}")
        return False

def main():
    print_header("CTR PREDICTION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_step(1, "CHECKING REQUIREMENTS")
    if not check_python_version():
        sys.exit(1)
    
    print_step(2, "INSTALLING DEPENDENCIES")
    if not install_dependencies():
        print("Failed to install dependencies. Please check your internet connection.")
        sys.exit(1)
    
    print_step(3, "CHECKING DATA FILES")
    required_files = [
        ('train_data.parquet', 'Training data'),
        ('test_data.parquet', 'Test data'),
        ('add_event.parquet', 'Event data'),
        ('add_trans.parquet', 'Transaction data'),
        ('offer_metadata.parquet', 'Offer metadata'),
        ('data_dictionary.csv', 'Data dictionary'),
        ('submission_template.csv', 'Submission template')
    ]
    
    all_files_present = True
    for filename, description in required_files:
        if not check_file_exists(filename, description):
            all_files_present = False
    
    if not all_files_present:
        print("\nMissing required files. Please ensure all data files are present.")
        sys.exit(1)
    
    print("\nAll required files found!")
    
    print_step(4, "DATA LOADING AND EXPLORATORY DATA ANALYSIS")
    if not run_script('phase1_eda.py', 'Phase 1 - EDA'):
        sys.exit(1)
    
    print_step(5, "FEATURE ENGINEERING")
    if not run_script('phase2_feature_engineering.py', 'Phase 2 - Feature Engineering'):
        sys.exit(1)
    
    print_step(6, "DATA PREPROCESSING")
    if not run_script('phase3_preprocessing.py', 'Phase 3 - Preprocessing'):
        sys.exit(1)
    
    print_step(7, "MODEL TRAINING")
    if not run_script('phase4_model_training.py', 'Phase 4 - Model Training'):
        sys.exit(1)
    
    print_step(8, "PREDICTION AND RANKING")
    if not run_script('phase5_prediction_ranking.py', 'Phase 5 - Prediction and Ranking'):
        sys.exit(1)
    
    print_step(9, "CHECKING OUTPUTS")
    output_files = [
        ('final_submission.csv', 'Final submission'),
        ('detailed_predictions.csv', 'Detailed predictions'),
        ('lgbm_ctr_model.txt', 'Trained model'),
        ('feature_importance.png', 'Feature importance plot')
    ]
    
    for filename, description in output_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"{description}: {filename} ({file_size:.2f} MB)")
        else:
            print(f"{description}: {filename} (not found)")
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput files:")
    print("   - final_submission.csv: Your submission file")
    print("   - detailed_predictions.csv: Detailed predictions for analysis")
    print("   - lgbm_ctr_model.txt: Trained LightGBM model")
    print("   - feature_importance.png: Feature importance visualization")
    print("\nEnjoy, submit the final_submission.csv for evaluation my friend")

if __name__ == "__main__":
    main() 