import os
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt to import tabulate support for markdown; fallback if missing
try:
    # pandas uses tabulate internally for DataFrame.to_markdown
    import tabulate  # noqa: F401
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    filename=f"eda_{timestamp}.log", 
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def save_fig(fig, filepath):
    """Save a Matplotlib figure to disk and close it."""
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    logger.info(f"Saved figure: {filepath}")


def df_to_md(df: pd.DataFrame, **kwargs) -> str:
    """Convert a DataFrame to markdown table; fallback to plain string if tabulate missing."""
    try:
        return df.to_markdown(**kwargs)
    except Exception:
        logger.warning("to_markdown failed; falling back to to_string().")
        return df.to_string(**kwargs)


def write_report_line(report_path, text=""):  # pylint: disable=missing-function-docstring
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(text + "\n")


def main(input_file: str, output_dir: str):
    # Prepare output structure
    ensure_dir(output_dir)
    figs_dir = os.path.join(output_dir, 'figures')
    ensure_dir(figs_dir)
    report_file = os.path.join(output_dir, 'exploratory_report.md')
    # Initialize report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Exploratory Data Analysis Report\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Data shape: {df.shape}")

    # Section 1: Data Overview
    write_report_line(report_file, '## 1. Data Overview')
    write_report_line(report_file, f"- Number of rows: {df.shape[0]}")
    write_report_line(report_file, f"- Number of columns: {df.shape[1]}")
    write_report_line(report_file, '### 1.1 Column Types and Missing Values')
    col_info = df.dtypes.to_frame('dtype')
    col_info['missing'] = df.isnull().sum()
    write_report_line(report_file, df_to_md(col_info))

    # Section 2: Missing Value Analysis
    write_report_line(report_file, '## 2. Missing Value Analysis')
    missing = df.isnull().mean() * 100
    write_report_line(report_file, 'Percentage missing per column:')
    write_report_line(report_file, df_to_md(missing.to_frame('percent_missing')))  
    fig = plt.figure()
    missing.plot.bar(title='Missing Values (%)')
    save_fig(fig, os.path.join(figs_dir, 'missing_values.png'))
    write_report_line(report_file, '![Missing Values](figures/missing_values.png)')

    # Section 3: Descriptive Statistics
    write_report_line(report_file, '## 3. Descriptive Statistics (Numeric Features)')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    write_report_line(report_file, df_to_md(desc))

    # Section 4: Categorical Feature Analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    write_report_line(report_file, '## 4. Categorical Feature Distribution')
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        write_report_line(report_file, f"### {col}")
        write_report_line(report_file, df_to_md(vc.to_frame('count')))
        fig = plt.figure()
        sns.countplot(y=col, data=df, order=vc.index)
        plt.title(f'Distribution of {col}')
        save_fig(fig, os.path.join(figs_dir, f'{col}_countplot.png'))
        write_report_line(report_file, f'![{col} distribution](figures/{col}_countplot.png)')

    # Section 5: Target Variable Analysis
    write_report_line(report_file, '## 5. Target Variable (loan_status) Analysis')
    target = 'loan_status'
    write_report_line(report_file, f"Distribution of `{target}`:")
    ts = df[target].value_counts(normalize=True) * 100
    write_report_line(report_file, df_to_md(ts.to_frame('percent')))
    fig = plt.figure()
    ts.plot.pie(autopct='%1.1f%%', title='Loan Status')
    save_fig(fig, os.path.join(figs_dir, 'target_distribution.png'))
    write_report_line(report_file, '![Target Distribution](figures/target_distribution.png)')

    # Section 6: Feature vs Target Relationships
    write_report_line(report_file, '## 6. Feature vs Target Relationships')
    for col in num_cols:
        if col == target:
            continue
        fig = plt.figure()
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'{col} by {target}')
        save_fig(fig, os.path.join(figs_dir, f'{col}_by_{target}.png'))
        write_report_line(report_file, f'### {col} by {target}')
        write_report_line(report_file, f'![{col} by {target}](figures/{col}_by_{target}.png)')

    # Section 7: Correlation Analysis
    write_report_line(report_file, '## 7. Correlation Analysis')
    corr = df[num_cols].corr()
    write_report_line(report_file, '### Correlation Matrix')
    write_report_line(report_file, df_to_md(corr))
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    save_fig(fig, os.path.join(figs_dir, 'correlation_heatmap.png'))
    write_report_line(report_file, '![Correlation Heatmap](figures/correlation_heatmap.png)')

    # Section 8: Outlier Detection Summary
    write_report_line(report_file, '## 8. Outlier Detection')
    outlier_summary = []
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        pct = len(outliers) / len(df) * 100
        outlier_summary.append((col, len(outliers), f"{pct:.2f}%"))
    outlier_df = pd.DataFrame(outlier_summary, columns=['feature','count','percent'])
    write_report_line(report_file, df_to_md(outlier_df, index=False))

    logger.info("EDA completed successfully.")
    print(f"EDA report and figures saved in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive EDA for cleaned_credit_risk.csv')
    parser.add_argument('--input_file', type=str, default=r'C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\cleaned_credit_risk.csv', help='Path to cleaned CSV file')
    parser.add_argument('--output_dir', type=str, default=r'C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\exploratory-data-analysis', help='Directory to save EDA results')
    args = parser.parse_args()
    main(args.input_file, args.output_dir)
