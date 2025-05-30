import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, roc_curve
import json
import matplotlib.pyplot as plt
import sys
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress XGBoost serialization warning if desired
warnings.filterwarnings("ignore", category=UserWarning, message=".*If you are loading a serialized model.*")

# Paths (update as needed)
FEATURE_NAMES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\feature_names.npy"
FEATURES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\processed_features.csv"
TARGETS_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\targets.csv"
MODEL_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\saved_models\xgb_classifier.model"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\xgb_validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # 1. Load feature names and align columns
    feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
    if feature_names.ndim > 1:
        feature_names = feature_names.flatten()
    feature_names = list(feature_names)
    X_val_raw = pd.read_csv(FEATURES_PATH)
    missing = set(feature_names) - set(X_val_raw.columns)
    extra = set(X_val_raw.columns) - set(feature_names)
    if missing:
        logger.error(f"Missing columns in validation data: {sorted(list(missing))}")
        sys.exit(1)
    if extra:
        logger.warning(f"Extra columns in validation data (will be ignored): {sorted(list(extra))}")
    X_val = X_val_raw.reindex(columns=feature_names, fill_value=0)
    logger.info(f"Validation features aligned. Shape: {X_val.shape}")

    # 2. Load targets
    y_val = pd.read_csv(TARGETS_PATH)
    y_def_val = y_val['loan_status']
    y_int_val = y_val['loan_int_rate']

    # 3. Load model
    model = joblib.load(MODEL_PATH)
    # --- XGBoost serialization best practice ---
    # If you see a warning about model serialization, it means the model was saved with joblib/pickle.
    # For best compatibility, save and load XGBoost models using Booster.save_model() and Booster.load_model().
    # Example (for training script):
    #   model.save_model('xgb_classifier.json')
    # Example (for validation):
    #   model = xgb.Booster()
    #   model.load_model('xgb_classifier.json')
    # If you retrain, consider switching to this approach for future-proofing.
    if isinstance(model, xgb.Booster):
        logger.info("Model loaded as XGBoost Booster (future-proofed).")
    else:
        logger.info("Model loaded via joblib. If you see version warnings, consider saving/loading with Booster.save_model/load_model.")

    # 4. Predict
    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    y_def_pred_prob = model.predict(dval)
    y_def_pred = (y_def_pred_prob > 0.5).astype(int)

    # 5. Classification metrics
    acc = accuracy_score(y_def_val, y_def_pred)
    auc = roc_auc_score(y_def_val, y_def_pred_prob)

    # 6. Regression metrics (if you have a separate regression model, load and predict here)
    # For demonstration, we use the same model for regression if multi-task, else skip.
    mae = mean_absolute_error(y_int_val, y_def_pred_prob)  # Replace with regression prediction if available
    mse = mean_squared_error(y_int_val, y_def_pred_prob)  # Replace with regression prediction if available

    metrics = {
        'default_accuracy': float(acc),
        'default_auc': float(auc),
        'interest_mae': float(mae),
        'interest_mse': float(mse)
    }
    with open(os.path.join(OUTPUT_DIR, 'xgb_validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Validation metrics: {metrics}')

    # 7. ROC Curve
    fpr, tpr, _ = roc_curve(y_def_val, y_def_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Default Prediction (Validation)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_default.png'))
    plt.close()

except Exception as e:
    logger.exception(f"Validation pipeline failed: {e}")
    sys.exit(1)
