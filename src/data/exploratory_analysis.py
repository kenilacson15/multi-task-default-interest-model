import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import logging
import argparse


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from the given path.
    """
    logging.info(f"Loading data from {path}")
    try:
        df = pd.read_csv(path)
    except Exception:
        # Try whitespace/tab-delimited fallback
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    logging.info(f"Data shape: {df.shape}")
    return df


def create_output_dir(path: str):
    """
    Create the output directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created output directory {path}")
    else:
        logging.info(f"Output directory {path} already exists")


def save_summary(df: pd.DataFrame, output_dir: str):
    """
    Save descriptive statistics to CSV.
    """
    desc = df.describe(include='all').T
    desc.to_csv(os.path.join(output_dir, 'data_summary.csv'))
    logging.info("Saved summary statistics to data_summary.csv")


def missing_values_report(df: pd.DataFrame, output_dir: str):
    """
    Save missing values count per column.
    """
    missing = df.isnull().sum()
    missing.to_csv(os.path.join(output_dir, 'missing_values.csv'), header=['missing_count'])
    logging.info("Saved missing values report to missing_values.csv")




def plot_numeric(df: pd.DataFrame, numeric_cols: list, output_dir: str):
    """
    Generate histograms and boxplots for numeric columns.
    """
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(os.path.join(output_dir, f'hist_{col}.png'))
        plt.close()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(output_dir, f'boxplot_{col}.png'))
        plt.close()
    logging.info("Saved numeric variable plots")



def plot_categorical(df: pd.DataFrame, cat_cols: list, output_dir: str):

    """
    Generate countplots for categorical columns.
    """
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        order = df[col].value_counts().index
        sns.countplot(x=df[col], order=order)
        plt.title(f'Countplot of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'count_{col}.png'))
        plt.close()
    logging.info("Saved categorical variable plots")



def correlation_heatmap(df: pd.DataFrame, numeric_cols: list, output_dir: str):


    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    logging.info("Saved correlation heatmap")



def generate_profile_report(df: pd.DataFrame, output_dir: str):
    """
    Generate an interactive HTML profiling report (requires ydata-profiling).
    """
    profile = ProfileReport(df, title="Pandas Profiling Report", minimal=False)
    profile.to_file(os.path.join(output_dir, "profile_report.html"))
    logging.info("Saved interactive profile report to profile_report.html")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive EDA for credit risk dataset"
    )
    parser.add_argument(
        "--input_path", type=str,
        default=r"C:\Users\Ken Ira Talingting\Desktop\Survival Analysis of Time-to-Default\data\raw\credit_risk_dataset.csv",
        help="Path to the raw CSV dataset"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"C:\Users\Ken Ira Talingting\Desktop\Survival Analysis of Time-to-Default\data\exploratory-data-analysis",
        help="Directory where EDA outputs will be saved"
    )
    args = parser.parse_args()

    setup_logging()
    create_output_dir(args.output_dir)
    df = load_data(args.input_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    save_summary(df, args.output_dir)
    missing_values_report(df, args.output_dir)
    plot_numeric(df, numeric_cols, args.output_dir)
    plot_categorical(df, cat_cols, args.output_dir)
    correlation_heatmap(df, numeric_cols, args.output_dir)

    

if __name__ == "__main__":
    main()



