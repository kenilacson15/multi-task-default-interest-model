import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix, roc_curve
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

    def hyperparameter_search(self, n_iter=20, cv=3, random_state=42):
        """Perform RandomizedSearchCV for XGBoost hyperparameters"""
        try:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            from xgboost import XGBClassifier
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0.5, 1, 2, 5],
            }
            xgb_clf = XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=self.y_train_class.value_counts()[0] / self.y_train_class.value_counts()[1],
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='auc',
                tree_method='hist',
                n_jobs=-1
            )
            search = RandomizedSearchCV(
                xgb_clf,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=cv,
                verbose=1,
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(self.X_train, self.y_train_class)
            self.best_params = search.best_params_
            logger.info(f"Best hyperparameters: {self.best_params}")
            return self.best_params
        except Exception as e:
            logger.error(f"Hyperparameter search failed: {str(e)}")
            return None

    def train_model(self, early_stopping_rounds=20):
        """Train XGBoost model with early stopping and best params if available"""
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
            # Use best params if available
            params = self.best_params.copy() if self.best_params else {}
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42
            })
            # Remove unused parameters for xgb.train
            params.pop('n_estimators', None)
            params.pop('device', None)
            evals = [(dtrain_class, 'train'), (dtest_class, 'eval')]
            self.model = xgb.train(
                params,
                dtrain_class,
                num_boost_round=self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else 1000,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50
            )
            try:
                joblib.dump(self.model, os.path.join(self.model_dir, 'xgb_classifier.model'))
                with open(os.path.join(self.model_dir, 'xgb_best_params.json'), 'w') as f:
                    import json
                    json.dump(params, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save model or params: {str(e)}")
            return True
        except xgb.core.XGBoostError as e:
            logger.error(f"XGBoost error during training: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Model training failed: {str(e)}")
            return False

    def evaluate_model(self):
        """Evaluate model performance with multiple metrics and plots, including threshold tuning."""
        try:
            if self.model is None:
                logger.error("Model not trained. Call train_model() first.")
                return None
            dtest = xgb.DMatrix(self.X_test)
            y_pred_proba = self.model.predict(dtest)
            # Threshold tuning for best F1
            from sklearn.metrics import f1_score, precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(self.y_test_class, y_pred_proba)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1s)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            logger.info(f"Best threshold for F1: {best_threshold:.4f}")
            y_pred = (y_pred_proba > best_threshold).astype(int)
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
                metrics['Best_Threshold'] = best_threshold
            except Exception as e:
                logger.warning(f"Classification metric calculation failed: {str(e)}")
            logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
            # Confusion matrix
            try:
                cm = confusion_matrix(self.y_test_class, y_pred)
                plt.figure(figsize=(6, 5))
                plt.imshow(cm, cmap='Blues', interpolation='nearest')
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir, "confusion_matrix.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"Confusion matrix plot failed: {str(e)}")
            # ROC curve
            try:
                fpr, tpr, _ = roc_curve(self.y_test_class, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["AUC"]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir, "roc_curve.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"ROC curve plot failed: {str(e)}")
            # Feature importance
            try:
                plt.figure(figsize=(12, 8))
                plot_importance(self.model, max_num_features=20, importance_type='gain')
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir, "feature_importance.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"Feature importance plot failed: {str(e)}")
            # Save metrics
            try:
                with open(os.path.join(self.plot_dir, "metrics.json"), 'w') as f:
                    import json
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                logger.warning(f"Saving metrics failed: {str(e)}")
            return metrics
        except Exception as e:
            logger.exception(f"Model evaluation failed: {str(e)}")
            return None

    def run_pipeline(self):
        """Run complete pipeline with error handling and optimization"""
        logger.info("Starting XGBoost baseline pipeline")
        steps = [
            ("Data Loading", self.load_data),
            ("Data Preparation", self.prepare_data),
            ("Hyperparameter Search", self.hyperparameter_search),
            ("Model Training", self.train_model),
            ("Model Evaluation", self.evaluate_model)
        ]
        for step_name, step_func in steps:
            logger.info(f"Executing step: {step_name}")
            result = step_func()
            if not result and step_name != "Hyperparameter Search":
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