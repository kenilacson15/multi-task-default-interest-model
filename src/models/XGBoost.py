import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error, r2_score
)
from xgboost import plot_importance
import matplotlib.pyplot as plt
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostBaseline:
    def __init__(self, features_path, targets_path, output_dir="models"):
        """Initialize paths and configuration"""
        self.features_path = features_path
        self.targets_path = targets_path
        self.output_dir = output_dir
        self.model = None
        self.best_params = {}
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.plot_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.model_dir = os.path.join(self.output_dir, "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        """Load and validate feature and target data"""
        try:
            if not os.path.exists(self.features_path):
                logger.error(f"Features file not found: {self.features_path}")
                return False
            if not os.path.exists(self.targets_path):
                logger.error(f"Targets file not found: {self.targets_path}")
                return False
            self.X = pd.read_csv(self.features_path)
            self.y = pd.read_csv(self.targets_path)
            if self.X.empty or self.y.empty:
                logger.error("Loaded features or targets are empty.")
                return False
            if len(self.X) != len(self.y):
                logger.error(f"Features and targets have different lengths: {len(self.X)} vs {len(self.y)}")
                return False
            logger.info(f"Loaded {self.X.shape[1]} features and {len(self.X)} samples")
            return True
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data error: {str(e)}")
            return False
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Data loading failed: {str(e)}")
            return False

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data with stratification for classification target"""
        try:
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                logger.error("Data not loaded. Call load_data() first.")
                return False
            if 'loan_status' not in self.y.columns or 'loan_int_rate' not in self.y.columns:
                logger.error("Target columns missing in targets file.")
                return False
            self.y_class = self.y['loan_status']
            self.y_reg = self.y['loan_int_rate']
            if self.y_class.nunique() < 2:
                logger.error("Classification target must have at least two classes.")
                return False
            self.X_train, self.X_test, self.y_train_class, self.y_test_class = train_test_split(
                self.X, self.y_class,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y_class
            )
            _, _, self.y_train_reg, self.y_test_reg = train_test_split(
                self.X, self.y_reg,
                test_size=test_size,
                random_state=random_state
            )
            logger.info(f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            return True
        except ValueError as e:
            logger.error(f"Data preparation ValueError: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Data preparation failed: {str(e)}")
            return False

    def train_model(self, early_stopping_rounds=20):
        """Train XGBoost model with early stopping"""
        try:
            if not hasattr(self, 'X_train') or not hasattr(self, 'y_train_class'):
                logger.error("Training data not prepared. Call prepare_data() first.")
                return False
            if self.y_train_class.nunique() < 2:
                logger.error("Training labels must have at least two classes.")
                return False
            dtrain_class = xgb.DMatrix(self.X_train, label=self.y_train_class)
            dtest_class = xgb.DMatrix(self.X_test, label=self.y_test_class)
            class_counts = self.y_train_class.value_counts()
            if 0 not in class_counts or 1 not in class_counts:
                logger.error("Both classes (0 and 1) must be present in training labels.")
                return False
            scale_pos_weight = class_counts[0] / class_counts[1]
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'device': 'cpu'
            }
            cv_results = xgb.cv(
                params,
                dtrain_class,
                num_boost_round=1000,
                early_stopping_rounds=early_stopping_rounds,
                folds=StratifiedKFold(n_splits=5),
                metrics=("auc"),
                seed=42
            )
            best_rounds = cv_results.shape[0]
            logger.info(f"Best boosting rounds: {best_rounds}")
            self.model = xgb.train(
                params,
                dtrain_class,
                num_boost_round=best_rounds
            )
            try:
                joblib.dump(self.model, os.path.join(self.model_dir, 'xgb_classifier.model'))
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
            return True
        except xgb.core.XGBoostError as e:
            logger.error(f"XGBoost error during training: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Model training failed: {str(e)}")
            return False

    def evaluate_model(self):
        """Evaluate model performance with multiple metrics"""
        try:
            if self.model is None:
                logger.error("Model not trained. Call train_model() first.")
                return None
            dtest = xgb.DMatrix(self.X_test)
            y_pred_proba = self.model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            metrics = {}
            try:
                metrics['AUC'] = roc_auc_score(self.y_test_class, y_pred_proba)
            except Exception as e:
                logger.warning(f"AUC calculation failed: {str(e)}")
                metrics['AUC'] = None
            try:
                metrics['Accuracy'] = accuracy_score(self.y_test_class, y_pred)
                metrics['Precision'] = precision_score(self.y_test_class, y_pred)
                metrics['Recall'] = recall_score(self.y_test_class, y_pred)
                metrics['F1'] = f1_score(self.y_test_class, y_pred)
            except Exception as e:
                logger.warning(f"Classification metric calculation failed: {str(e)}")
            logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
            try:
                plt.figure(figsize=(12, 8))
                plot_importance(self.model, max_num_features=20, importance_type='gain')
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir, "feature_importance.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"Feature importance plot failed: {str(e)}")
            return metrics
        except Exception as e:
            logger.exception(f"Model evaluation failed: {str(e)}")
            return None

    def run_pipeline(self):
        """Run complete pipeline with error handling"""
        logger.info("Starting XGBoost baseline pipeline")
        
        steps = [
            ("Data Loading", self.load_data),
            ("Data Preparation", self.prepare_data),
            ("Model Training", self.train_model),
            ("Model Evaluation", self.evaluate_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Executing step: {step_name}")
            result = step_func()
            if not result:
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
                
        logger.info("XGBoost baseline pipeline completed successfully")
        return True

if __name__ == "__main__":
    # Define paths
    features_path = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
    targets_path = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\targets.csv"
    
    # Initialize and run pipeline
    pipeline = XGBoostBaseline(features_path, targets_path)
    pipeline.run_pipeline()