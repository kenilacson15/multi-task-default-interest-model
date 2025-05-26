import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

# Paths
FEATURES_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
TARGETS_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\targets.csv"
REPORT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\eda_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
features = pd.read_csv(FEATURES_PATH)
targets = pd.read_csv(TARGETS_PATH)

# Use only classification target for feature importance
if 'loan_status' not in targets.columns:
    raise ValueError("Target column 'loan_status' not found in targets.csv")
y = targets['loan_status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)

# 1. Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)

# 2. Logistic Regression Coefficient Importance
lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
lr_importances = pd.Series(np.abs(lr.coef_[0]), index=features.columns).sort_values(ascending=False)

# 3. AUC for quick sanity check
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# 4. Correlation heatmap (top 20 features by RF importance)
top_features = rf_importances.head(20).index.tolist()
plt.figure(figsize=(12, 10))
sns.heatmap(features[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap (Top 20 Features by RF Importance)')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'feature_corr_heatmap.png'))
plt.close()

# 5. Save feature importance reports
rf_importances.to_csv(os.path.join(REPORT_DIR, 'rf_feature_importance.csv'))
lr_importances.to_csv(os.path.join(REPORT_DIR, 'lr_feature_importance.csv'))

# 6. Plot feature importances
plt.figure(figsize=(10, 7))
rf_importances.head(20).plot(kind='barh', title='Random Forest Feature Importance (Top 20)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'rf_feature_importance.png'))
plt.close()

plt.figure(figsize=(10, 7))
lr_importances.head(20).plot(kind='barh', title='Logistic Regression Coefficient Importance (Top 20)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'lr_feature_importance.png'))
plt.close()

# 7. Generate summary report
def write_report():
    report_path = os.path.join(REPORT_DIR, 'feature_engineering_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write('Feature Engineering Quality Report\n')
        f.write('='*60 + '\n')
        f.write(f"Random Forest Test AUC: {rf_auc:.4f}\n")
        f.write(f"Logistic Regression Test AUC: {lr_auc:.4f}\n\n")
        f.write('Top 10 Features by Random Forest Importance:\n')
        for i, (feat, imp) in enumerate(rf_importances.head(10).items(), 1):
            f.write(f"{i:2d}. {feat:40s} {imp:.4f}\n")
        f.write('\nTop 10 Features by Logistic Regression Coefficient:\n')
        for i, (feat, imp) in enumerate(lr_importances.head(10).items(), 1):
            f.write(f"{i:2d}. {feat:40s} {imp:.4f}\n")
        f.write('\nCorrelation heatmap saved as feature_corr_heatmap.png\n')
        f.write('Feature importance plots and CSVs saved in eda_reports/\n')
        f.write('\nRecommendations:\n')
        f.write('- Investigate highly correlated features for redundancy.\n')
        f.write('- Consider removing or combining low-importance features.\n')
        f.write('- Review top features for domain sense and leakage.\n')
        f.write('- Use SHAP for deeper interpretability if needed.\n')
write_report()

print(f"Feature engineering quality report and plots saved in {REPORT_DIR}")
