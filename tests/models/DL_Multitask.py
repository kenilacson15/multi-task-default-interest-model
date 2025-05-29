import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths (edit as needed)
DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\processed_features.csv"
TARGET_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\targets.csv"
MODEL_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\dl_multitask_baseline"
FEATURE_NAMES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\feature_names.npy"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load and preprocess data
def load_and_preprocess(feature_path, target_path):
    try:
        X = pd.read_csv(feature_path)
        y = pd.read_csv(target_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}. Please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    # Targets
    try:
        y_default = y['loan_status']  # 0/1
        y_interest = y['loan_int_rate']  # regression
    except KeyError as e:
        raise KeyError(f"Missing expected column in targets: {e}")
    # Train/val split
    try:
        X_train, X_val, y_def_train, y_def_val, y_int_train, y_int_val = train_test_split(
            X, y_default, y_interest, test_size=0.2, random_state=SEED, stratify=y_default)
    except Exception as e:
        raise RuntimeError(f"Error during train/validation split: {e}")
    # Scale features
    scaler = StandardScaler()
    try:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    except Exception as e:
        raise RuntimeError(f"Error during feature scaling: {e}")
    return (X_train, X_val, y_def_train, y_def_val, y_int_train, y_int_val, scaler)

# 2. Build model
def build_model(input_dim):
    try:
        inputs = Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        # Default head
        def_branch = layers.Dense(32, activation='relu')(x)
        def_out = layers.Dense(1, activation='sigmoid', name='default_output')(def_branch)
        # Interest head
        int_branch = layers.Dense(32, activation='relu')(x)
        int_out = layers.Dense(1, activation='linear', name='int_rate_output')(int_branch)
        model = Model(inputs=inputs, outputs=[def_out, int_out])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'default_output': 'binary_crossentropy', 'int_rate_output': 'mse'},
            metrics={'default_output': 'accuracy', 'int_rate_output': 'mae'}
        )
    except Exception as e:
        raise RuntimeError(f"Error building or compiling the model: {e}")
    return model

# 3. Train model
def train_model(model, X_train, y_def_train, y_int_train, X_val, y_def_val, y_int_val):
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'), save_best_only=True, monitor='val_loss')
    ]
    try:
        history = model.fit(
            X_train, {'default_output': y_def_train, 'int_rate_output': y_int_train},
            validation_data=(X_val, {'default_output': y_def_val, 'int_rate_output': y_int_val}),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=2
        )
    except Exception as e:
        raise RuntimeError(f"Error during model training: {e}")
    return history

# 4. Main
if __name__ == "__main__":
    try:
        X_train, X_val, y_def_train, y_def_val, y_int_train, y_int_val, scaler = load_and_preprocess(DATA_PATH, TARGET_PATH)
        # Load pre-trained model instead of building/training a new one
        model_path = os.path.join(MODEL_DIR, 'best_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
        model = keras.models.load_model(model_path)
        print(model.summary())
        # --- Defensive check for feature shape consistency (using fixed validation data) ---
        X_val_raw = pd.read_csv(DATA_PATH)
        try:
            if os.path.exists(FEATURE_NAMES_PATH):
                feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
                if feature_names.ndim > 1:
                    feature_names = feature_names.flatten()
                X_val_df = X_val_raw.reindex(columns=feature_names, fill_value=0)
                X_val = X_val_df.values
                extra_cols = set(X_val_raw.columns) - set(feature_names)
                missing_cols = set(feature_names) - set(X_val_raw.columns)
                col_order_mismatch = list(X_val_raw.columns) != list(feature_names)
                if extra_cols:
                    print(f"[DEBUG] Extra columns in validation not in training: {sorted(list(extra_cols))}")
                if missing_cols:
                    print(f"[DEBUG] Missing columns in validation that were in training: {sorted(list(missing_cols))}")
                if col_order_mismatch:
                    print(f"[DEBUG] Column order mismatch detected between validation and training features.")
                    print(f"[DEBUG] Validation columns: {list(X_val_raw.columns)}")
                    print(f"[DEBUG] Training feature_names: {list(feature_names)}")
            else:
                print("Warning: feature_names.npy not found. Please save feature names during training for strict consistency.")
                X_val = X_val_raw.values
        except Exception as e:
            print(f"[ERROR] Exception during feature alignment: {e}")
            print(f"[DEBUG] X_val_raw shape: {X_val_raw.shape}")
            print(f"[DEBUG] feature_names_path: {FEATURE_NAMES_PATH}")
            raise
        # Load targets for validation
        val_targets = pd.read_csv(r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data\targets.csv")
        y_def_val = val_targets['loan_status']
        y_int_val = val_targets['loan_int_rate']
        # Check shape before prediction with more diagnostics
        if X_val.shape[1] != model.input_shape[1]:
            print(f"[ERROR] Feature shape mismatch: model expects {model.input_shape[1]}, got {X_val.shape[1]}")
            print(f"[DEBUG] X_val columns: {list(X_val_raw.columns)}")
            if os.path.exists(FEATURE_NAMES_PATH):
                print(f"[DEBUG] feature_names: {list(feature_names)}")
            raise ValueError(f"Input feature shape mismatch: model expects {model.input_shape[1]}, got {X_val.shape[1]}")
        # Import dependencies for metrics and plotting
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, roc_curve
        import matplotlib.pyplot as plt
        import json
        # Predict on validation set
        y_def_pred_prob, y_int_pred = model.predict(X_val)
        y_def_pred = (y_def_pred_prob > 0.5).astype(int)
        # Classification metrics
        acc = accuracy_score(y_def_val, y_def_pred)
        auc = roc_auc_score(y_def_val, y_def_pred_prob)
        # Regression metrics
        mae = mean_absolute_error(y_int_val, y_int_pred)
        mse = mean_squared_error(y_int_val, y_int_pred)
        # Save metrics
        metrics = {
            'default_accuracy': float(acc),
            'default_auc': float(auc),
            'interest_mae': float(mae),
            'interest_mse': float(mse)
        }
        with open(os.path.join(MODEL_DIR, 'dl_validation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Validation metrics:', metrics)
        # --- Logging ---
        log_path = os.path.join(MODEL_DIR, 'performance_metrics_log.txt')
        from datetime import datetime
        with open(log_path, 'a') as logf:
            logf.write(f"\n=== DL Model Validation Performance Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            for metric, value in metrics.items():
                logf.write(f"{metric}: {value}\n")
            logf.write("\n")
        # --- Visualization ---
        # 1. Training curves (skip if not available)
        # 2. ROC Curve for default
        fpr, tpr, _ = roc_curve(y_def_val, y_def_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Default Prediction')
        plt.legend()
        plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_default.png'))
        plt.close()
        # 3. Scatter plot for interest rate prediction
        plt.figure()
        plt.scatter(y_int_val, y_int_pred, alpha=0.5)
        plt.xlabel('True Interest Rate')
        plt.ylabel('Predicted Interest Rate')
        plt.title('Interest Rate Prediction (Validation)')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'int_rate_scatter.png'))
        plt.close()
        print(f"Performance metrics and plots saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
