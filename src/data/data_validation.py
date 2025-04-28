import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paths
DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\capped_credit_risk.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\exploratory-data-analysis\data_quality_validation"

# Ensure output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 1. Load data
def load_data(path):
    df = pd.read_csv(path)
    return df

# 2. Outlier detection using IQR method
def detect_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    pct = len(outliers) / len(df) * 100
    return pct, lower, upper

# 3. Skewness check and transform suggestion
def check_skewness(df, col):
    skew = df[col].skew()
    suggestion = None
    if abs(skew) > 1:
        if (df[col] > 0).all():
            suggestion = 'Consider log or Box-Cox transform'
        else:
            suggestion = 'Consider Yeo-Johnson transform'
    return skew, suggestion

# 4. Class imbalance
def check_class_imbalance(df, target):
    counts = df[target].value_counts(normalize=True) * 100
    return counts.to_dict()

# 5. Target definition validation
def validate_targets(df):
    report = []
    # loan_status
    unique_status = df['loan_status'].unique()
    counts_status = df['loan_status'].value_counts().to_dict()
    report.append((unique_status, counts_status))
    # loan_int_rate
    stats_rate = df['loan_int_rate'].describe().to_dict()
    return report, stats_rate

# 6. Generate report
def generate_report(df):
    lines = []
    lines.append('# Data Quality Validation Report\n')
    # Outliers
    lines.append('## Outlier Detection (IQR method)')
    for col in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']:
        pct, low, high = detect_outliers(df, col)
        action = 'Review and cap or drop' if pct > 1 else 'No major outliers'
        lines.append(f'- **{col}**: {pct:.2f}% outliers outside [{low:.2f}, {high:.2f}]. {action}.')
    lines.append('')

    # Skewness
    lines.append('## Distributional Skewness')
    for col in ['person_income', 'loan_amnt']:
        skew, suggestion = check_skewness(df, col)
        sug_text = suggestion if suggestion else 'No transform required'
        lines.append(f'- **{col}**: skewness={skew:.2f}. {sug_text}.')
    lines.append('')

    # Class imbalance
    lines.append('## Class Imbalance')
    imbalance = check_class_imbalance(df, 'loan_status')
    for cls, pct in imbalance.items():
        lines.append(f'- Class {cls}: {pct:.2f}%')
    lines.append('Suggestion: Use stratified sampling or class weights in model training.')
    lines.append('')

    # Target definitions
    lines.append('## Target Definition Validation')
    report_targets, stats_rate = validate_targets(df)
    unique_status, counts_status = report_targets[0]
    lines.append(f'- **loan_status** unique values: {list(unique_status)}, counts: {counts_status}')
    lines.append('- Confirm that 0 = paid, 1 = default/delinquency; map any extra values appropriately.')
    lines.append(f'- **loan_int_rate** summary: {stats_rate}')
    lines.append('- Decide whether to predict raw APR or classify into high/low bins (e.g., above/below median).')

    # Write markdown report
    report_path = os.path.join(OUTPUT_DIR, 'data_quality_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Report saved to {report_path}')

# 7. Generate visualizations for skew and outliers
def generate_plots(df):
    # Histograms with outlier thresholds
    for col in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']:
        pct, low, high = detect_outliers(df, col)
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.axvline(low, color='red', linestyle='--')
        plt.axvline(high, color='red', linestyle='--')
        plt.title(f'{col} distribution with IQR bounds')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{col}_outlier_hist.png'))
        plt.close()

    # Skewness suggestion plots
    for col in ['person_income', 'loan_amnt']:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} skewness: {df[col].skew():.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{col}_skew_hist.png'))
        plt.close()

    print(f'Plots saved to {OUTPUT_DIR}')

# Main execution
def main():
    ensure_dir(OUTPUT_DIR)
    df = load_data(DATA_PATH)
    generate_report(df)
    generate_plots(df)
    print('Data quality validation complete.')

if __name__ == '__main__':
    main()
