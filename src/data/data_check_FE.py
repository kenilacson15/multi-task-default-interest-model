import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

# Set up paths
features_path = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\processed_features.csv"
targets_path = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer\targets.csv"

output_dir = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\eda_reports"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
features = pd.read_csv(features_path)
targets = pd.read_csv(targets_path)

# 1. Basic Data Structure Analysis
def basic_structure_analysis(df, name):
    print(f"\n{name} Dataset Structure:")
    print("-" * 50)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nData Types:")
    print(df.dtypes)
    return df.describe(include='all').T

# 2. Missing Values Analysis (FUNCTION NAME FIXED HERE)
def missing_values_report(df, name):
    print(f"\n{name} Missing Values Analysis:")
    print("-" * 50)
    missing = df.isnull().sum()
    percent_missing = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_percent': percent_missing})
    missing_df = missing_df[missing_df['missing_count'] > 0]
    if not missing_df.empty:
        print(missing_df.sort_values(by='missing_percent', ascending=False))
    else:
        print("No missing values found.")
    return missing_df  # Fixed indentation

# 3. Duplicate Analysis
def duplicate_analysis(df, name):
    duplicates = df.duplicated().sum()
    print(f"\n{name} Duplicate Rows: {duplicates}")
    return duplicates

# 4. Numerical Features Analysis
def numerical_analysis(df):
    print("\nNumerical Features Analysis:")
    print("-" * 50)
    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) > 0:
        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, col in enumerate(numerical_cols[:4]):
            sns.histplot(df[col], ax=axes[i//2, i%2], kde=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numerical_distributions.png'))
        plt.close()
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Variance analysis
        var_threshold = VarianceThreshold(threshold=0.1)
        var_threshold.fit(df[numerical_cols])
        low_variance = numerical_cols[~var_threshold.get_support()].tolist()
        print("\nLow Variance Numerical Features:", low_variance)
        return low_variance
    return []

# 5. Categorical Features Analysis
def categorical_analysis(df):
    print("\nCategorical Features Analysis:")
    print("-" * 50)
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            print(f"\n{col} Unique Values ({len(df[col].unique())}):")
            print(df[col].value_counts(normalize=True).head(10))
        return categorical_cols
    return []

# 6. Target Analysis
def target_analysis(targets):
    print("\nTarget Variables Analysis:")
    print("-" * 50)
    for col in targets.columns:
        print(f"\n{col} Distribution:")
        if targets[col].dtype in ['int64', 'float64']:
            sns.histplot(targets[col], kde=True)
            plt.title(f'{col} Distribution')
            plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
            plt.close()
        else:
            print(targets[col].value_counts(normalize=True))
        print("\nMissing Values:", targets[col].isnull().sum())

# 7. Feature-Target Relationship Analysis
def feature_target_analysis(features, targets):
    print("\nFeature-Target Relationships:")
    print("-" * 50)
    relationships = {}
    for target_col in targets.columns:
        if pd.api.types.is_numeric_dtype(targets[target_col]):
            # Numerical target - correlation
            correlations = features.corrwith(targets[target_col])
            relationships[target_col] = correlations.sort_values(ascending=False).head(5)
        else:
            # Categorical target - ANOVA or Chi-square
            pass  # Implement based on specific requirements
    print(relationships)
    return relationships

# Run analyses
print("=== DATA QUALITY AND STRUCTURE REPORT ===")
print("=" * 50)

# Features analysis
features_desc = basic_structure_analysis(features, "Features")
features_missing = missing_values_report(features, "Features")  # UPDATED FUNCTION NAME HERE
features_dupes = duplicate_analysis(features, "Features")
num_low_var = numerical_analysis(features)
cat_features = categorical_analysis(features)

# Targets analysis
targets_desc = basic_structure_analysis(targets, "Targets")
target_missing = missing_values_report(targets, "Targets")  # UPDATED FUNCTION NAME HERE
targets_dupes = duplicate_analysis(targets, "Targets")
target_analysis(targets)

# Combined analysis
if features.shape[0] == targets.shape[0]:
    print("\nFeatures and Targets have matching row counts")
else:
    print("\nWarning: Features and Targets have mismatched row counts")

feature_target_corr = feature_target_analysis(features, targets)

# Save summary report
with open(os.path.join(output_dir, 'data_quality_report.txt'), 'w') as f:
    f.write("=== DATA QUALITY AND STRUCTURE REPORT ===\n")
    f.write(f"Features Shape: {features.shape}\n")
    f.write(f"Targets Shape: {targets.shape}\n")
    f.write(f"Missing Features Columns: {features_missing.shape[0]}\n")
    f.write(f"Missing Targets Columns: {target_missing.shape[0]}\n")
    f.write(f"Low Variance Numerical Features: {num_low_var}\n")
    f.write(f"Duplicate Rows in Features: {features_dupes}\n")
    f.write(f"Duplicate Rows in Targets: {targets_dupes}\n")
    f.write(f"Categorical Features: {list(cat_features)}\n")
    f.write("Top Feature-Target Correlations:\n")
    for target, corrs in feature_target_corr.items():
        f.write(f"\n{target}:\n{str(corrs)}\n")

print(f"\nReport saved to: {output_dir}")