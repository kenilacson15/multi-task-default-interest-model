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
DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
TARGET_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\targets.csv"
MODEL_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\models\dl_multitask_baseline"
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
        model = build_model(X_train.shape[1])
        print(model.summary())
        history = train_model(model, X_train, y_def_train, y_int_train, X_val, y_def_val, y_int_val)
        # Save scaler
        import joblib
        try:
            joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
        except Exception as e:
            print(f"Warning: Could not save scaler: {e}")
        print(f"Model and scaler saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
    else:
        # --- Performance Metrics Report ---
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
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
        # --- Visualization ---
        # 1. Training curves
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['default_output_accuracy'], label='Train Acc')
        plt.plot(history.history['val_default_output_accuracy'], label='Val Acc')
        plt.title('Default Classification Accuracy')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history.history['int_rate_output_mae'], label='Train MAE')
        plt.plot(history.history['val_int_rate_output_mae'], label='Val MAE')
        plt.title('Interest Rate MAE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
        plt.close()
        # 2. ROC Curve for default
        from sklearn.metrics import roc_curve
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
