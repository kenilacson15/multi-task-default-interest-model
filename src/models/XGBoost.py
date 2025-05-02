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
            self.X = pd.read_csv(self.features_path)
            self.y = pd.read_csv(self.targets_path)
            
            # Validate data shapes
            if len(self.X) != len(self.y):
                raise ValueError("Features and targets have different lengths")
                
            logger.info(f"Loaded {self.X.shape[1]} features and {len(self.X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return False

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data with stratification for classification target"""
        try:
            # Separate classification and regression targets
            self.y_class = self.y['loan_status']
            self.y_reg = self.y['loan_int_rate']
            
            # Stratified split for classification
            self.X_train, self.X_test, self.y_train_class, self.y_test_class = train_test_split(
                self.X, self.y_class,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y_class
            )
            
            # Regular split for regression
            _, _, self.y_train_reg, self.y_test_reg = train_test_split(
                self.X, self.y_reg,
                test_size=test_size,
                random_state=random_state
            )
            
            logger.info(f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return False

    def train_model(self, early_stopping_rounds=20):
        """Train XGBoost model with early stopping"""
        try:
            # Convert to DMatrix for better performance
            dtrain_class = xgb.DMatrix(self.X_train, label=self.y_train_class)
            dtest_class = xgb.DMatrix(self.X_test, label=self.y_test_class)
            
            # Calculate class weights
            class_counts = self.y_train_class.value_counts()
            scale_pos_weight = class_counts[0] / class_counts[1]
            
            # Default parameters for classification
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'device': 'cpu'
            }
            
            # Cross-validation for early stopping
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
            
            # Train final model
            self.model = xgb.train(
                params,
                dtrain_class,
                num_boost_round=best_rounds
            )
            
            # Save model
            joblib.dump(self.model, os.path.join(self.model_dir, 'xgb_classifier.model'))
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False

    def evaluate_model(self):
        """Evaluate model performance with multiple metrics"""
        try:
            dtest = xgb.DMatrix(self.X_test)
            y_pred_proba = self.model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Classification metrics
            metrics = {
                'AUC': roc_auc_score(self.y_test_class, y_pred_proba),
                'Accuracy': accuracy_score(self.y_test_class, y_pred),
                'Precision': precision_score(self.y_test_class, y_pred),
                'Recall': recall_score(self.y_test_class, y_pred),
                'F1': f1_score(self.y_test_class, y_pred)
            }
            
            logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
            # Feature importance
            plt.figure(figsize=(12, 8))
            plot_importance(self.model, max_num_features=20, importance_type='gain')
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "feature_importance.png"))
            plt.close()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
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