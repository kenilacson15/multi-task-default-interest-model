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
import xgboost as xgb

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
    parser = argparse.ArgumentParser(description="Generate feature importance analysis for XGBoost")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained XGBoost model (.model or .joblib)")
    parser.add_argument("--features-path", type=str, required=True, help="Path to processed features CSV")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output TXT file for ranked feature importances")
    return parser.parse_args()

# ----------------------
# XGBoost Feature Importance
# ----------------------
def xgb_feature_importance(model_path: str, features_path: str, output_path: str):
    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load feature names
    try:
        X = pd.read_csv(features_path)
        feature_names = list(X.columns)
        logger.info(f"Loaded {len(feature_names)} feature names from {features_path}")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return

    # Get feature importances (by gain)
    try:
        fmap = {f'f{i}': name for i, name in enumerate(feature_names)}
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        importance_dict = booster.get_score(importance_type='gain')
        # Map XGBoost feature names to actual names
        importance = [(fmap.get(k, k), v) for k, v in importance_dict.items()]
        importance.sort(key=lambda x: -x[1])
    except Exception as e:
        logger.error(f"Failed to extract feature importances: {e}")
        return

    # Write to TXT file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("Ranked Feature Importances (by gain):\n")
            for i, (feat, val) in enumerate(importance, 1):
                f.write(f"{i}. {feat}: {val:.6f}\n")
        logger.info(f"Feature importance report written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write feature importance TXT: {e}")

# ----------------------
# Main Pipeline
# ----------------------
def main():
    args = parse_args()
    xgb_feature_importance(args.model_path, args.features_path, args.output_path)

if __name__ == "__main__":
    main()