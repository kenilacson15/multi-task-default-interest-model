import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

# Configuration
FEATURES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
TARGETS_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\targets.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\model-results"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and merge features and targets"""
    features = pd.read_csv(FEATURES_PATH)
    targets = pd.read_csv(TARGETS_PATH)
    return features, targets['loan_status']

def train_model(X_train, y_train):
    """Train logistic regression model with class balancing"""
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Generate evaluation metrics and visualizations"""
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    return report, roc_auc

def feature_importance(model, features, output_dir):
    """Generate feature importance report and visualization"""
    importance = pd.DataFrame({
        'feature': features.columns,
        'coefficient': model.coef_[0],
        'abs_importance': np.abs(model.coef_[0])
    })
    importance = importance.sort_values('abs_importance', ascending=False)

    # Save to text file
    with open(os.path.join(output_dir, 'feature_importance.txt'), 'w') as f:
        f.write("Feature Importance Report\n")
        f.write("="*50 + "\n")
        f.write(importance.to_string(index=False))

    # Visualize top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='coefficient',
        y='feature',
        data=importance.head(20),
        palette='viridis'
    )
    plt.title('Top 20 Features by Coefficient Magnitude')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def main():
    # Load and prepare data
    X, y = load_data()
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    report, roc_auc = evaluate_model(model, X_test, y_test)

    # Generate feature importance
    feature_importance(model, X, OUTPUT_DIR)

    # Save performance report
    with open(os.path.join(OUTPUT_DIR, 'performance_report.txt'), 'w') as f:
        f.write("Model Performance Report\n")
        f.write("="*50 + "\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nFeature Importance file and visualizations saved in directory.")

if __name__ == "__main__":
    main()