#!/usr/bin/env python3
import os
import argparse
import logging
from typing import Tuple, Optional, Dict   
import json
import warnings

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Configure Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Argument Parser
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate feature importance analysis for logistic regression")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained logistic regression model (.joblib)")
    parser.add_argument("--X-train-path", type=str, required=True, help="Path to X_train.csv")
    parser.add_argument("--X-val-path", type=str, required=True, help="Path to X_val.csv")
    parser.add_argument("--output-dir", type=str, default="./reports/feature_importance", help="Directory to save outputs")
    parser.add_argument("--use-shap", action="store_true", help="Whether to use SHAP values (slower but more accurate)")
    return parser.parse_args()

# ----------------------
# Load Data & Model
# ----------------------
def load_artifacts(model_path: str, X_train_path: str, X_val_path: str) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading train data from {X_train_path}")
    X_train = pd.read_csv(X_train_path)
    
    logger.info(f"Loading validation data from {X_val_path}")
    X_val = pd.read_csv(X_val_path)
    
    return model, X_train, X_val

# ----------------------
# Generate Feature Importance
# ----------------------
def get_coefficient_importance(model: LogisticRegression, feature_names: list) -> pd.DataFrame:
    logger.info("Calculating logistic regression coefficient importance...")
    coefs = pd.Series(model.coef_[0], index=feature_names)
    return coefs.sort_values(key=lambda x: abs(x), ascending=False)

# ----------------------
# Generate SHAP Values
# ----------------------
def get_shap_values(model: LogisticRegression, X: pd.DataFrame) -> Dict[str, object]:
    logger.info("Calculating SHAP values (this may take a minute)...")
    
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # For binary classification, we focus on class 1 (default risk)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        return {
            "explainer": explainer,
            "shap_values": shap_values
        }
    except Exception as e:
        logger.error(f"SHAP calculation failed: {str(e)}")
        return {}

# ----------------------
# Plotting Functions
# ----------------------
def plot_coefficients(importance: pd.Series, output_path: str):
    logger.info(f"Plotting coefficient importance to {output_path}")
    plt.figure(figsize=(12, 8))
    importance.plot(kind='barh', title='Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_shap_summary(shap_values, output_path: str):
    logger.info(f"Plotting SHAP summary to {output_path}")
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_shap_waterfall(shap_values, X_val: pd.DataFrame, output_path: str):
    logger.info(f"Plotting SHAP waterfall to {output_path}")
    idx = 0
    
    # Create SHAP explanation for one instance
    explanation = shap._explanation.Explanation(
        values=shap_values[idx].values,
        base_values=shap_values.base_values[idx],
        data=X_val.iloc[idx].values,
        feature_names=X_val.columns.tolist()
    )
    
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ----------------------
# Save Artifacts
# ----------------------
def save_artifacts(importance: pd.Series, output_dir: str, shap_data: dict = None, X_train: pd.DataFrame = None):
    logger.info(f"Saving artifacts to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save coefficient importance
    importance.to_csv(os.path.join(output_dir, "coefficient_importance.csv"))
    
    # Save SHAP explanation (if applicable)
    if shap_data and X_train is not None:
        try:
            shap_values = shap_data['shap_values']
            
            # Use feature names from X_train
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': X_train.columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values(by='mean_abs_shap', ascending=False)
            
            shap_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"))
        except Exception as e:
            logger.error(f"SHAP artifact saving failed: {str(e)}")



def get_shap_values(model: LogisticRegression, X: pd.DataFrame) -> Dict[str, object]:
    logger.info("Calculating SHAP values (this may take a minute)...")
    
    try:
        assert isinstance(X, pd.DataFrame), "Input must be a pandas DataFrame"
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # Handle binary classification
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        return {"explainer": explainer, "shap_values": shap_values}
    except Exception as e:
        logger.error(f"SHAP calculation failed: {str(e)}")
        return {}

# ----------------------
# Main Pipeline
# ----------------------
def main():
    args = parse_args()
    
    try:
        # Setup output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Load artifacts
        model, X_train, X_val = load_artifacts(
            model_path=args.model_path,
            X_train_path=args.X_train_path,
            X_val_path=args.X_val_path
        )
        
        # Step 2: Get feature importance
        coef_importance = get_coefficient_importance(model, X_train.columns.tolist())
        
        # Step 3: Get SHAP values
        shap_data = {}
        if args.use_shap:
            shap_data = get_shap_values(model, X_train)
        
        # Step 4: Save results
        save_artifacts(coef_importance, args.output_dir, shap_data)
        
        # Step 5: Generate plots
        plot_coefficients(
            coef_importance,
            os.path.join(args.output_dir, "logreg_coefficient_importance.png")
        )
        
        if args.use_shap and shap_data:
            plot_shap_summary(
                shap_data['shap_values'],
                os.path.join(args.output_dir, "shap_summary_plot.png")
            )
            
            plot_shap_waterfall(
                shap_data['shap_values'],
                X_val,
                os.path.join(args.output_dir, "shap_waterfall_example.png")
            )
        
        logger.info("Feature importance analysis completed successfully.")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()