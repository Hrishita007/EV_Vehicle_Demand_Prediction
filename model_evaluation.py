"""
Model Evaluation Script for EV Demand Forecasting
This script calculates and displays comprehensive accuracy metrics for the trained model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression model performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict : Dictionary containing all performance metrics
    """
    
    # Remove any NaN or infinite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Calculate MAPE (handle division by zero)
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    
    # Calculate Median Absolute Error
    median_ae = np.median(np.abs(y_true_clean - y_pred_clean))
    
    # Calculate RMSLE (Root Mean Squared Logarithmic Error) - useful for demand forecasting
    rmsle = np.sqrt(np.mean((np.log1p(y_pred_clean) - np.log1p(y_true_clean))**2))
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2,
        'MAPE': mape,
        'Median_AE': median_ae,
        'RMSLE': rmsle,
        'Sample_Count': len(y_true_clean)
    }
    
    return metrics


def evaluate_model(model_path, data_path, test_size=0.2):
    """
    Load model and data, then evaluate performance.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model (.pkl file)
    data_path : str
        Path to the preprocessed data (.csv file)
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    dict : Evaluation results and metrics
    """
    
    # Load model and data
    print("Loading model and data...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_columns = [
        'months_since_start', 'county_encoded', 'ev_total_lag1', 'ev_total_lag2',
        'ev_total_lag3', 'ev_total_roll_mean_3', 'ev_total_pct_change_1',
        'ev_total_pct_change_3', 'ev_growth_slope'
    ]
    
    # Check if target column exists
    target_column = 'Electric Vehicle (EV) Total'
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Using alternative column.")
        # Try to find the target column
        target_column = [col for col in df.columns if 'EV' in col and 'Total' in col][0]
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Remove rows with NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    train_metrics = calculate_metrics(y_train.values, y_pred_train)
    test_metrics = calculate_metrics(y_test.values, y_pred_test)
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }


def print_evaluation_report(eval_results):
    """
    Print a formatted evaluation report.
    
    Parameters:
    -----------
    eval_results : dict
        Results from evaluate_model()
    """
    
    print("\n" + "="*70)
    print(" "*15 + "MODEL EVALUATION REPORT")
    print("="*70)
    
    print("\n--- TRAINING SET METRICS ---")
    train_metrics = eval_results['train_metrics']
    print(f"R² Score:              {train_metrics['R2_Score']:.4f}")
    print(f"RMSE:                  {train_metrics['RMSE']:.2f}")
    print(f"MAE:                   {train_metrics['MAE']:.2f}")
    print(f"MAPE:                  {train_metrics['MAPE']:.4f} ({train_metrics['MAPE']*100:.2f}%)")
    print(f"Median Absolute Error: {train_metrics['Median_AE']:.2f}")
    print(f"RMSLE:                 {train_metrics['RMSLE']:.4f}")
    print(f"Samples:               {train_metrics['Sample_Count']}")
    
    print("\n--- TESTING SET METRICS ---")
    test_metrics = eval_results['test_metrics']
    print(f"R² Score:              {test_metrics['R2_Score']:.4f}")
    print(f"RMSE:                  {test_metrics['RMSE']:.2f}")
    print(f"MAE:                   {test_metrics['MAE']:.2f}")
    print(f"MAPE:                  {test_metrics['MAPE']:.4f} ({test_metrics['MAPE']*100:.2f}%)")
    print(f"Median Absolute Error: {test_metrics['Median_AE']:.2f}")
    print(f"RMSLE:                 {test_metrics['RMSLE']:.4f}")
    print(f"Samples:               {test_metrics['Sample_Count']}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("• R² Score:     Closer to 1.0 is better (0-1 range). Explains variance.")
    print("• RMSE/MAE:     Lower values are better. MAE in same units as target.")
    print("• MAPE:         Percentage error. Lower is better.")
    print("• Overfitting:  If test metrics much worse than train, model is overfitting.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run evaluation
    try:
        results = evaluate_model(
            model_path='forecasting_ev_model.pkl',
            data_path='preprocessed_ev_data.csv',
            test_size=0.2
        )
        print_evaluation_report(results)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Please ensure 'forecasting_ev_model.pkl' and 'preprocessed_ev_data.csv' exist.")
