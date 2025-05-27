import os
import json
from datetime import datetime
import xgboost as xgb
import joblib

# Paths (edit as needed)
MODEL_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\dl_multitask_baseline"
METRICS_PATH = os.path.join(MODEL_DIR, 'dl_validation_metrics.json')
LOG_PATH = os.path.join(MODEL_DIR, 'performance_metrics_log.txt')

def log_metrics(metrics_path, log_path):
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_lines = [
        f"=== Model Performance Log: {now} ===",
        f"Default Accuracy:   {metrics.get('default_accuracy', 'N/A'):.4f}",
        f"Default AUC:        {metrics.get('default_auc', 'N/A'):.4f}",
        f"Interest MAE:       {metrics.get('interest_mae', 'N/A'):.4f}",
        f"Interest MSE:       {metrics.get('interest_mse', 'N/A'):.4f}",
        ""
    ]
    with open(log_path, 'a') as f:
        f.write('\n'.join(log_lines))
    print(f"Metrics logged to {log_path}")

def log_feature_importance_xgb(model_path, log_path, features_path=None):
    """
    Logs ranked XGBoost feature importances (by gain) to a TXT file.
    """
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")
        return
    # Try to get feature names from features_path if provided
    feature_names = None
    if features_path and os.path.exists(features_path):
        try:
            import pandas as pd
            X = pd.read_csv(features_path)
            feature_names = list(X.columns)
        except Exception as e:
            print(f"Could not load feature names: {e}")
    # Get feature importance
    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    fmap = {f'f{i}': name for i, name in enumerate(feature_names)} if feature_names else None
    importance_dict = booster.get_score(importance_type='gain')
    # Map XGBoost feature names to actual names if available
    importance = [(fmap.get(k, k) if fmap else k, v) for k, v in importance_dict.items()]
    importance.sort(key=lambda x: -x[1])
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_lines = [
        f"=== XGBoost Feature Importance Log: {now} ===",
        "Ranked Feature Importances (by gain):"
    ]
    for i, (feat, val) in enumerate(importance, 1):
        log_lines.append(f"{i}. {feat}: {val:.6f}")
    log_lines.append("")
    with open(log_path, 'a') as f:
        f.write('\n'.join(log_lines))
    print(f"XGBoost feature importances logged to {log_path}")

if __name__ == "__main__":
    log_metrics(METRICS_PATH, LOG_PATH)
    # Add XGBoost feature importance logging
    XGB_MODEL_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\saved_models\xgb_classifier.model"
    FEATURES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
    XGB_LOG_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\model-results\feature_importance.txt"
    log_feature_importance_xgb(XGB_MODEL_PATH, XGB_LOG_PATH, FEATURES_PATH)
